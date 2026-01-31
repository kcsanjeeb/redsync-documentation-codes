# src/train_phase5A_prefix_drift_probe_v1.py
from __future__ import annotations

import argparse
import os
import time
import json
from typing import Dict, Tuple, List, Optional

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
# Cutpoints for ResNet-18 (torchvision naming)
# boundary_idx -> output after that point
# 0: stem output
# 1: layer1[0] output
# 2: layer1[1] output
# 3: layer2[0] output
# 4: layer2[1] output
# 5: layer3[0] output
# 6: layer3[1] output
# 7: layer4[0] output
# 8: layer4[1] output
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


def split_resnet18_prefix(model: nn.Module, boundary_idx: int) -> nn.Module:
    """
    Build a prefix forward module that returns activations at boundary_idx.
    Uses the same underlying model modules.
    """
    assert boundary_idx in CUTPOINTS, f"Unknown boundary_idx={boundary_idx}"
    layer_name, block_idx = CUTPOINTS[boundary_idx]

    class Prefix(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x):
            # stem
            x = self.m.conv1(x)
            x = self.m.bn1(x)
            x = self.m.relu(x)
            x = self.m.maxpool(x)

            if layer_name == "stem":
                return x

            layers = ["layer1", "layer2", "layer3", "layer4"]
            for lname in layers:
                layer = getattr(self.m, lname)
                for bi, blk in enumerate(layer):
                    x = blk(x)
                    if lname == layer_name and bi == block_idx:
                        return x

            raise RuntimeError("Boundary not reached in Prefix.forward")

    return Prefix(model)


def make_cifar_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    aug: bool,
) -> Tuple[DataLoader, DataLoader]:
    if aug:
        tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        tr = transforms.Compose([transforms.ToTensor()])

    te = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tr)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=te)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, testloader


def make_probe_loader(
    data_dir: str,
    probe_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """
    Probe set must be deterministic: no augmentation.
    We sample a fixed subset of training indices.
    """
    ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(probe_size, len(ds)), replace=False).tolist()
    subset = Subset(ds, idx)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def eval_full(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def compute_prefix_feats(prefix: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Returns [N, ...] CPU float32 tensor of prefix activations for probe set.
    """
    prefix.eval()
    feat_list: List[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        cuda_sync_if_needed(device)
        feats = prefix(xb)
        cuda_sync_if_needed(device)
        feats_cpu = feats.detach().to("cpu").to(torch.float32)
        feat_list.append(feats_cpu)
    return torch.cat(feat_list, dim=0).contiguous()


def drift_metrics(ref: torch.Tensor, cur: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute drift metrics between reference and current features.
    Operates per-sample on flattened vectors.
    """
    assert ref.shape == cur.shape, f"shape mismatch: {ref.shape} vs {cur.shape}"
    r = ref.view(ref.shape[0], -1)
    c = cur.view(cur.shape[0], -1)
    d = c - r

    # L2 per sample
    r_norm = torch.norm(r, p=2, dim=1)
    d_norm = torch.norm(d, p=2, dim=1)
    rel = d_norm / (r_norm + eps)

    # cosine similarity per sample
    dot = (r * c).sum(dim=1)
    c_norm = torch.norm(c, p=2, dim=1)
    cos = dot / (r_norm * c_norm + eps)

    # summarize
    def mean(x): return float(x.mean().item())
    def p95(x): return float(torch.quantile(x, 0.95).item())
    def p99(x): return float(torch.quantile(x, 0.99).item())

    out = {
        "delta_l2_mean": mean(d_norm),
        "delta_l2_p95": p95(d_norm),
        "delta_l2_p99": p99(d_norm),
        "delta_rel_l2_mean": mean(rel),
        "delta_rel_l2_p95": p95(rel),
        "cosine_mean": mean(cos),
        "cosine_p05": float(torch.quantile(cos, 0.05).item()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()

    # data/training
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    # optimizer
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    # drift probe
    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--transition_epoch", type=int, default=170,
                    help="Epoch at which we snapshot reference prefix activations for drift tracking.")
    ap.add_argument("--probe_every_epochs", type=int, default=5,
                    help="After transition_epoch, measure drift every N epochs.")
    ap.add_argument("--probe_size", type=int, default=2048,
                    help="Number of deterministic training samples used for drift measurement.")
    ap.add_argument("--probe_batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    # logging
    ap.add_argument("--log_path", type=str, required=True)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-friendly ResNet-18
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    prefix = split_resnet18_prefix(model, args.boundary_idx).to(device)

    trainloader, testloader = make_cifar_loaders(
        args.data_dir, args.batch_size, args.num_workers, aug=True
    )
    probe_loader = make_probe_loader(
        args.data_dir,
        probe_size=args.probe_size,
        batch_size=args.probe_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ref_feats: Optional[torch.Tensor] = None
    ref_shape: Optional[List[int]] = None

    for epoch in range(1, args.epochs + 1):
        t0 = now()

        # normal training (full model) — this is a measurement run
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in trainloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * yb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += yb.size(0)

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = 100.0 * train_correct / max(1, train_total)
        test_acc = eval_full(model, testloader, device)

        # snapshot ref feats at transition epoch
        drift_log: Dict[str, float] = {}
        if epoch == args.transition_epoch:
            ref_feats = compute_prefix_feats(prefix, probe_loader, device)
            ref_shape = list(ref_feats.shape)
            drift_log.update({
                "drift_ref_epoch": int(epoch),
                "probe_size": int(ref_feats.shape[0]),
                "probe_feat_shape": ref_shape,
            })

        # measure drift periodically after transition
        if ref_feats is not None and epoch > args.transition_epoch:
            if (epoch - args.transition_epoch) % max(1, args.probe_every_epochs) == 0:
                cur_feats = compute_prefix_feats(prefix, probe_loader, device)
                m = drift_metrics(ref_feats, cur_feats)
                drift_log.update(m)
                drift_log.update({
                    "drift_measured": True,
                    "drift_epoch": int(epoch),
                    "probe_feat_shape": ref_shape,
                })

        ep_s = now() - t0
        lr_cur = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": int(epoch),
            "epoch_time_s": float(ep_s),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "lr": float(lr_cur),

            # probe config (for paper reproducibility)
            "boundary_idx": int(args.boundary_idx),
            "transition_epoch": int(args.transition_epoch),
            "probe_every_epochs": int(args.probe_every_epochs),
            "probe_size_arg": int(args.probe_size),
            "probe_batch_size": int(args.probe_batch_size),
            "seed": int(args.seed),
        }
        row.update(drift_log)

        with open(args.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if drift_log.get("drift_measured", False):
            print(
                f"Epoch {epoch:03d} | train_acc={train_acc:.2f} | test_acc={test_acc:.2f} "
                f"| time={ep_s:.2f}s | drift_l2_mean={drift_log.get('delta_l2_mean', 0.0):.6f} "
                f"| cos_mean={drift_log.get('cosine_mean', 0.0):.6f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | train_acc={train_acc:.2f} | test_acc={test_acc:.2f} "
                f"| time={ep_s:.2f}s"
            )

    print("Done.")


if __name__ == "__main__":
    main()
