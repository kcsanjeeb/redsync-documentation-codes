# src/train_phase5C_prefixcache_delta_residual_v2.py
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models


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
# Cutpoints (ResNet-18)
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


def split_resnet18(model: nn.Module, boundary_idx: int) -> Tuple[nn.Module, nn.Module]:
    assert boundary_idx in CUTPOINTS
    layer_name, block_idx = CUTPOINTS[boundary_idx]

    class Prefix(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = self.m.conv1(x)
            x = self.m.bn1(x)
            x = self.m.relu(x)
            x = self.m.maxpool(x)  # CIFAR: Identity()

            if layer_name == "stem":
                return x

            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, lname)
                for bi, blk in enumerate(layer):
                    x = blk(x)
                    if lname == layer_name and bi == block_idx:
                        return x

            raise RuntimeError("Boundary not reached in Prefix.forward")

    class Suffix(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, feat):
            x = feat
            passed = False
            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, lname)
                for bi, blk in enumerate(layer):
                    if not passed:
                        if lname == layer_name and bi == block_idx:
                            passed = True
                        continue
                    x = blk(x)
            x = self.m.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.m.fc(x)
            return x

    return Prefix(model), Suffix(model)


def prefix_param_prefixes(boundary_idx: int) -> List[str]:
    assert boundary_idx in CUTPOINTS
    layer_name, block_idx = CUTPOINTS[boundary_idx]

    prefixes = ["conv1", "bn1"]

    if layer_name == "stem":
        return prefixes

    order = ["layer1", "layer2", "layer3", "layer4"]
    for lname in order:
        if lname == layer_name:
            for bi in range(block_idx + 1):
                prefixes.append(f"{lname}.{bi}")
            break
        else:
            prefixes.append(f"{lname}.0")
            prefixes.append(f"{lname}.1")
    return prefixes


def freeze_prefix_params(model: nn.Module, boundary_idx: int) -> Dict[str, int]:
    prefs = prefix_param_prefixes(boundary_idx)
    frozen_tensors = 0
    frozen_params = 0

    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefs):
            if p.requires_grad:
                p.requires_grad = False
                frozen_tensors += 1
                frozen_params += p.numel()

    return {"frozen_param_tensors": frozen_tensors, "frozen_param_count": frozen_params}


# -------------------------
# Data helpers
# -------------------------
def make_train_loader(data_dir: str, batch_size: int, num_workers: int, aug: bool) -> DataLoader:
    if aug:
        tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        tr = transforms.Compose([transforms.ToTensor()])

    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tr)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def make_eval_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    te = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=te)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def make_probe_loader(data_dir: str, probe_size: int, probe_batch: int, num_workers: int, seed: int) -> DataLoader:
    te = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=te)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(probe_size, len(ds)), replace=False).tolist()
    subset = Subset(ds, idx)
    return DataLoader(subset, batch_size=probe_batch, shuffle=False, num_workers=num_workers, pin_memory=True)


class CachedFeatDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        assert feats.shape[0] == labels.shape[0]
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, i):
        return self.feats[i], self.labels[i]


@torch.no_grad()
def eval_full(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def compute_prefix_feats(prefix: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute prefix feats for entire loader dataset order (as yielded).
    Returns feats CPU float32, labels CPU long, and compute time.
    """
    prefix.eval()
    t0 = now()
    feats_list, labels_list = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        cuda_sync_if_needed(device)
        f = prefix(xb)
        cuda_sync_if_needed(device)
        feats_list.append(f.detach().to("cpu").to(torch.float32))
        labels_list.append(yb.detach().to("cpu"))
    feats = torch.cat(feats_list, dim=0).contiguous()
    labels = torch.cat(labels_list, dim=0).contiguous()
    return feats, labels, (now() - t0)


def drift_metrics(ref: torch.Tensor, cur: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compare probe features.
    """
    r = ref.view(ref.shape[0], -1)
    c = cur.view(cur.shape[0], -1)
    d = c - r

    r_norm = torch.norm(r, p=2, dim=1)
    d_norm = torch.norm(d, p=2, dim=1)
    rel = d_norm / (r_norm + eps)

    cos = (r * c).sum(1) / (r_norm * torch.norm(c, p=2, dim=1) + eps)

    out = {
        "delta_rel_l2_mean": float(rel.mean().item()),
        "delta_rel_l2_p95": float(torch.quantile(rel, 0.95).item()),
        "cosine_mean": float(cos.mean().item()),
        "cosine_p05": float(torch.quantile(cos, 0.05).item()),
        # global energy ratio (one scalar)
        "delta_energy_ratio": float(torch.norm(d).item() / (torch.norm(r).item() + eps)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--transition_epoch", type=int, default=170)
    ap.add_argument("--probe_every", type=int, default=5)

    # thresholds
    ap.add_argument("--tau_rel", type=float, default=0.05)
    ap.add_argument("--tau_cos", type=float, default=0.995)
    ap.add_argument("--tau_energy", type=float, default=0.10, help="delta mode if energy_ratio <= tau_energy")

    # suffix optimizer
    ap.add_argument("--suffix_lr", type=float, default=0.03)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_path", type=str, required=True)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    prefix, suffix = split_resnet18(model, args.boundary_idx)
    prefix.to(device)
    suffix.to(device)

    # loaders
    train_aug_loader = make_train_loader(args.data_dir, args.batch_size, args.num_workers, aug=True)
    train_noaug_loader = make_train_loader(args.data_dir, args.batch_size, args.num_workers, aug=False)
    test_loader = make_eval_loader(args.data_dir, args.batch_size, args.num_workers)
    probe_loader = make_probe_loader(args.data_dir, probe_size=2048, probe_batch=256, num_workers=args.num_workers, seed=args.seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    transitioned = False

    # base cache and delta cache (CPU)
    F_base: Optional[torch.Tensor] = None
    F_delta: Optional[torch.Tensor] = None
    cached_labels: Optional[torch.Tensor] = None

    # probe reference
    ref_probe: Optional[torch.Tensor] = None

    for epoch in range(1, args.epochs + 1):
        ep_t0 = now()
        log: Dict[str, object] = {"epoch": int(epoch)}

        # ---- normal training before transition ----
        if not transitioned and epoch < args.transition_epoch:
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for xb, yb in train_aug_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)

            log["train_loss"] = float(loss_sum / max(1, total))
            log["train_acc"] = float(100.0 * correct / max(1, total))
            log["test_acc"] = float(eval_full(model, test_loader, device))
            log["transitioned"] = False
            log["epoch_time_s"] = float(now() - ep_t0)

            with open(args.log_path, "a") as f:
                f.write(json.dumps(log) + "\n")

            print(f"Epoch {epoch:03d} | mode=train | time={log['epoch_time_s']:.2f}s | test_acc={log['test_acc']:.2f}")
            continue

        # ---- transition (build base cache + freeze prefix) ----
        if not transitioned and epoch >= args.transition_epoch:
            transitioned = True

            freeze_stats = freeze_prefix_params(model, args.boundary_idx)
            log.update({"transitioned": True, "transition_epoch": int(epoch), **freeze_stats})

            # probe reference at transition
            ref_probe, _, probe_build_s = compute_prefix_feats(prefix, probe_loader, device)
            log["probe_ref_build_s"] = float(probe_build_s)

            # base cache for entire train set (no aug)
            F_base, cached_labels, cache_build_s = compute_prefix_feats(prefix, train_noaug_loader, device)
            F_delta = None
            log["base_cache_build_s"] = float(cache_build_s)
            log["base_cache_shape"] = list(F_base.shape)
            log["base_cache_bytes"] = int(F_base.numel() * F_base.element_size())

            # new optimizer for suffix only
            suffix_params = [p for p in suffix.parameters() if p.requires_grad]
            if len(suffix_params) == 0:
                raise RuntimeError("Suffix has no trainable params after prefix freezing. Check boundary_idx / freeze logic.")
            optimizer = optim.SGD(suffix_params, lr=args.suffix_lr, momentum=0.9, weight_decay=5e-4)

        # ---- probe drift + decide mode ----
        reuse_mode = "reuse"
        refresh = False
        do_delta = False

        if (epoch - args.transition_epoch) % max(1, args.probe_every) == 0:
            cur_probe, _, probe_s = compute_prefix_feats(prefix, probe_loader, device)
            m = drift_metrics(ref_probe, cur_probe)
            log.update(m)
            log["probe_eval_s"] = float(probe_s)

            safe = (m["delta_rel_l2_p95"] <= args.tau_rel) and (m["cosine_p05"] >= args.tau_cos)
            if safe:
                reuse_mode = "reuse"
            else:
                # moderate drift: try delta residual if energy small
                if m["delta_energy_ratio"] <= args.tau_energy:
                    reuse_mode = "delta"
                    do_delta = True
                else:
                    reuse_mode = "refresh"
                    refresh = True

            log["reuse_mode"] = reuse_mode

            # advance reference to current probe (important: keeps deltas local, not cumulative)
            ref_probe = cur_probe

        # ---- apply mode to cached features ----
        if refresh:
            # rebuild base cache from current frozen prefix (no aug)
            F_base, cached_labels, cache_s = compute_prefix_feats(prefix, train_noaug_loader, device)
            F_delta = None
            log["refresh_cache_s"] = float(cache_s)
            log["refresh_cache_bytes"] = int(F_base.numel() * F_base.element_size())

        elif do_delta:
            # compute a delta cache (full dataset) and keep base
            cur_full, _, delta_s = compute_prefix_feats(prefix, train_noaug_loader, device)
            # delta = current - base
            F_delta = (cur_full - F_base).contiguous()
            log["delta_build_s"] = float(delta_s)
            log["delta_bytes"] = int(F_delta.numel() * F_delta.element_size())

        # training uses base (+ delta if present)
        feats_used = F_base if F_delta is None else (F_base + F_delta)
        cached_ds = CachedFeatDataset(feats_used, cached_labels)
        cached_loader = DataLoader(cached_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # ---- suffix-only training epoch ----
        suffix.train()
        loss_sum, correct, total = 0.0, 0, 0
        for feat_cpu, y_cpu in cached_loader:
            feat = feat_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = suffix(feat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        log["train_loss"] = float(loss_sum / max(1, total))
        log["train_acc"] = float(100.0 * correct / max(1, total))
        log["test_acc"] = float(eval_full(model, test_loader, device))

        log["epoch_time_s"] = float(now() - ep_t0)

        with open(args.log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        print(
            f"Epoch {epoch:03d} | mode={log.get('reuse_mode','reuse')} | time={log['epoch_time_s']:.2f}s "
            f"| test_acc={log['test_acc']:.2f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
