# src/train_phase5C_prefixcache_delta_residual_v1.py
from __future__ import annotations

import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ---------------- utils ----------------
def now(): return time.perf_counter()
def cuda_sync(d):
    if d.type == "cuda": torch.cuda.synchronize()

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# ---------------- cutpoints ----------------
CUTPOINTS = {
    0: ("stem", -1),
    1: ("layer1", 0), 2: ("layer1", 1),
    3: ("layer2", 0), 4: ("layer2", 1),
    5: ("layer3", 0), 6: ("layer3", 1),
    7: ("layer4", 0), 8: ("layer4", 1),
}

def split_resnet18(model, boundary):
    lname, bidx = CUTPOINTS[boundary]

    class Prefix(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            x = self.m.conv1(x); x = self.m.bn1(x)
            x = self.m.relu(x); x = self.m.maxpool(x)
            if lname == "stem": return x
            for ln in ["layer1","layer2","layer3","layer4"]:
                for bi, blk in enumerate(getattr(self.m, ln)):
                    x = blk(x)
                    if ln == lname and bi == bidx: return x
            raise RuntimeError()

    class Suffix(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, feat):
            x = feat; passed = False
            for ln in ["layer1","layer2","layer3","layer4"]:
                for bi, blk in enumerate(getattr(self.m, ln)):
                    if not passed:
                        if (ln, bi) == (lname, bidx): passed = True
                        continue
                    x = blk(x)
            x = self.m.avgpool(x)
            x = torch.flatten(x, 1)
            return self.m.fc(x)

    return Prefix(model), Suffix(model)

# ---------------- drift ----------------
def drift_stats(ref, cur, eps=1e-12):
    r = ref.view(ref.size(0), -1)
    c = cur.view(cur.size(0), -1)
    d = c - r
    rel = torch.norm(d, dim=1) / (torch.norm(r, dim=1) + eps)
    cos = (r * c).sum(1) / (torch.norm(r, dim=1)*torch.norm(c, dim=1) + eps)
    return {
        "rel_p95": float(torch.quantile(rel, 0.95)),
        "cos_p05": float(torch.quantile(cos, 0.05)),
        "delta_energy": float(torch.norm(d) / torch.norm(r))
    }

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--transition_epoch", type=int, default=170)
    ap.add_argument("--probe_every", type=int, default=5)
    ap.add_argument("--tau_rel", type=float, default=0.05)
    ap.add_argument("--tau_cos", type=float, default=0.995)
    ap.add_argument("--tau_energy", type=float, default=0.10)
    ap.add_argument("--suffix_lr", type=float, default=0.03)
    ap.add_argument("--log_path", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    prefix, suffix = split_resnet18(model, args.boundary_idx)
    prefix.to(device); suffix.to(device)

    # data
    tf = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10("data", train=True, download=True, transform=tf)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    probe_idx = np.random.choice(len(trainset), 2048, replace=False)
    probe_loader = DataLoader(Subset(trainset, probe_idx), batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    cached_feat = None
    ref_probe_feat = None
    transitioned = False

    for epoch in range(1, args.epochs+1):
        t0 = now()
        log = {}

        if not transitioned and epoch >= args.transition_epoch:
            transitioned = True
            for p in prefix.parameters(): p.requires_grad = False
            cached_feat = torch.cat(
                [prefix(x.to(device)).cpu().detach() for x,_ in trainloader]
            )
            ref_probe_feat = torch.cat(
                [prefix(x.to(device)).cpu().detach() for x,_ in probe_loader]
            )
            optimizer = optim.SGD(
                [p for p in suffix.parameters() if p.requires_grad],
                lr=args.suffix_lr, momentum=0.9
            )
            log["transitioned"] = True

        if transitioned and (epoch - args.transition_epoch) % args.probe_every == 0:
            cur_probe = torch.cat(
                [prefix(x.to(device)).cpu().detach() for x,_ in probe_loader]
            )
            stats = drift_stats(ref_probe_feat, cur_probe)
            log.update(stats)

            if stats["rel_p95"] <= args.tau_rel and stats["cos_p05"] >= args.tau_cos:
                mode = "reuse"
            elif stats["delta_energy"] <= args.tau_energy:
                mode = "delta"
                delta = cur_probe - ref_probe_feat
                ref_probe_feat = cur_probe
            else:
                mode = "refresh"
                cached_feat = torch.cat(
                    [prefix(x.to(device)).cpu().detach() for x,_ in trainloader]
                )
                ref_probe_feat = cur_probe
            log["reuse_mode"] = mode

        ep_s = now() - t0
        log.update({"epoch": epoch, "epoch_time_s": ep_s})

        with open(args.log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        print(f"Epoch {epoch:03d} | mode={log.get('reuse_mode','train')} | t={ep_s:.2f}s")

    print("Done")

if __name__ == "__main__":
    main()
