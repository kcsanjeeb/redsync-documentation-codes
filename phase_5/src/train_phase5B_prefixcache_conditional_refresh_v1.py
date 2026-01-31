# src/train_phase5B_prefixcache_conditional_refresh_v1.py
from __future__ import annotations

import argparse
import os
import time
import json
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# -------------------------
# Utilities
# -------------------------
def now() -> float:
    return time.perf_counter()


def cuda_sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Cutpoints
# -------------------------
CUTPOINTS = {
    0: ("stem", -1),
    1: ("layer1", 0),
    2: ("layer1", 1),
    3: ("layer2", 0),
    4: ("layer2", 1),
    5: ("layer3", 0),
    6: ("layer3", 1),
    7: ("layer4", 0),
    8: ("layer4", 1),
}


def split_resnet18(model: nn.Module, boundary_idx: int):
    assert boundary_idx in CUTPOINTS
    lname, bidx = CUTPOINTS[boundary_idx]

    class Prefix(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = self.m.conv1(x)
            x = self.m.bn1(x)
            x = self.m.relu(x)
            x = self.m.maxpool(x)
            if lname == "stem":
                return x
            for ln in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, ln)
                for bi, blk in enumerate(layer):
                    x = blk(x)
                    if ln == lname and bi == bidx:
                        return x
            raise RuntimeError("prefix boundary not reached")

    class Suffix(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, feat):
            x = feat
            passed = False
            for ln in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, ln)
                for bi, blk in enumerate(layer):
                    if not passed:
                        if (ln, bi) == (lname, bidx):
                            passed = True
                        continue
                    x = blk(x)
            x = self.m.avgpool(x)
            x = torch.flatten(x, 1)
            return self.m.fc(x)

    return Prefix(model), Suffix(model)


def freeze_prefix(model: nn.Module, boundary_idx: int):
    lname, bidx = CUTPOINTS[boundary_idx]
    prefixes = ["conv1", "bn1"]
    if lname != "stem":
        for ln in ["layer1", "layer2", "layer3", "layer4"]:
            if ln == lname:
                for bi in range(bidx + 1):
                    prefixes.append(f"{ln}.{bi}")
                break
            prefixes += [f"{ln}.0", f"{ln}.1"]

    for name, p in model.named_parameters():
        if any(name.startswith(pr) for pr in prefixes):
            p.requires_grad = False


# -------------------------
# Data
# -------------------------
def make_cifar_loaders(data_dir, batch_size, num_workers, aug):
    if aug:
        tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        tr = transforms.Compose([transforms.ToTensor()])

    te = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=tr)
    testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=te)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trainloader, testloader


def make_probe_loader(data_dir, probe_size, batch_size, num_workers, seed):
    ds = datasets.CIFAR10(data_dir, train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=probe_size, replace=False).tolist()
    subset = Subset(ds, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def compute_feats(prefix, loader, device):
    prefix.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(device)
        cuda_sync_if_needed(device)
        f = prefix(xb)
        cuda_sync_if_needed(device)
        out.append(f.detach().cpu().float())
    return torch.cat(out, dim=0)


def drift_metrics(ref, cur, eps=1e-12):
    r = ref.view(ref.size(0), -1)
    c = cur.view(cur.size(0), -1)
    d = c - r
    rnorm = torch.norm(r, dim=1)
    dnorm = torch.norm(d, dim=1)
    rel = dnorm / (rnorm + eps)
    cos = (r * c).sum(1) / (rnorm * torch.norm(c, dim=1) + eps)
    return {
        "delta_rel_l2_p95": float(torch.quantile(rel, 0.95)),
        "cosine_p05": float(torch.quantile(cos, 0.05)),
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--transition_epoch", type=int, default=170)
    ap.add_argument("--probe_every", type=int, default=5)
    ap.add_argument("--probe_size", type=int, default=2048)
    ap.add_argument("--probe_batch", type=int, default=256)
    ap.add_argument("--tau_rel", type=float, default=0.05)
    ap.add_argument("--tau_cos", type=float, default=0.995)
    ap.add_argument("--suffix_lr", type=float, default=0.03)

    ap.add_argument("--log_path", required=True)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    prefix, suffix = split_resnet18(model, args.boundary_idx)
    prefix.to(device)
    suffix.to(device)

    trainloader, testloader = make_cifar_loaders(
        args.data_dir, args.batch_size, args.num_workers, aug=True
    )
    probe_loader = make_probe_loader(
        args.data_dir, args.probe_size, args.probe_batch, args.num_workers, args.seed
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    cached_feats = None
    cached_labels = None
    ref_feats = None
    transitioned = False

    for epoch in range(1, args.epochs + 1):
        t0 = now()
        refresh = False
        drift_log = {}

        if (not transitioned) and epoch >= args.transition_epoch:
            transitioned = True
            freeze_prefix(model, args.boundary_idx)

            ref_feats = compute_feats(prefix, probe_loader, device)
            cached_feats = compute_feats(prefix, trainloader, device)
            cached_labels = torch.cat([y for _, y in trainloader], dim=0)

            optimizer = optim.SGD(
                [p for p in suffix.parameters() if p.requires_grad],
                lr=args.suffix_lr, momentum=0.9, weight_decay=5e-4
            )

            drift_log["transitioned"] = True

        if transitioned and (epoch - args.transition_epoch) % args.probe_every == 0:
            cur_feats = compute_feats(prefix, probe_loader, device)
            m = drift_metrics(ref_feats, cur_feats)
            drift_log.update(m)

            if m["delta_rel_l2_p95"] > args.tau_rel or m["cosine_p05"] < args.tau_cos:
                refresh = True
                ref_feats = cur_feats
                cached_feats = compute_feats(prefix, trainloader, device)
                drift_log["cache_refreshed"] = True
            else:
                drift_log["cache_refreshed"] = False

        # Training
        if transitioned:
            suffix.train()
            logits = suffix(cached_feats.to(device))
            loss = criterion(logits, cached_labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.train()
            for xb, yb in trainloader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        ep_s = now() - t0

        row = {
            "epoch": epoch,
            "epoch_time_s": ep_s,
            "transitioned": transitioned,
            **drift_log,
        }

        with open(args.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(f"Epoch {epoch:03d} | time={ep_s:.2f}s | refresh={drift_log.get('cache_refreshed', False)}")

    print("Done")


if __name__ == "__main__":
    main()
