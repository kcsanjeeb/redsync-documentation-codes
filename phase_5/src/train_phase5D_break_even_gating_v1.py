# src/train_phase5D_break_even_gating_v1.py
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


# -------------------------
# Timing utils
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
            raise RuntimeError("Boundary not reached")

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
    frozen_tensors, frozen_params = 0, 0
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefs):
            if p.requires_grad:
                p.requires_grad = False
                frozen_tensors += 1
                frozen_params += p.numel()
    return {"frozen_param_tensors": frozen_tensors, "frozen_param_count": frozen_params}


# -------------------------
# Data + cache dataset
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


def make_test_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    te = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=te)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


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
def build_full_prefix_cache(prefix: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build full-dataset cache of prefix feats on CPU (float32).
    Returns: feats_cpu_fp32, labels_cpu, build_time_s
    """
    prefix.eval()
    t0 = now()
    feats_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        cuda_sync_if_needed(device)
        f = prefix(xb)
        cuda_sync_if_needed(device)
        feats_list.append(f.detach().to("cpu").to(torch.float32))
        y_list.append(yb.detach().to("cpu"))

    feats = torch.cat(feats_list, dim=0).contiguous()
    labels = torch.cat(y_list, dim=0).contiguous()
    return feats, labels, (now() - t0)


# -------------------------
# Break-even gate state (EMA)
# -------------------------
class GateEMA:
    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self.prefix_ms: Optional[float] = None
        self.cache_ms: Optional[float] = None

    def update(self, prefix_ms: Optional[float], cache_ms: Optional[float]):
        if prefix_ms is not None:
            self.prefix_ms = prefix_ms if self.prefix_ms is None else (self.alpha * prefix_ms + (1 - self.alpha) * self.prefix_ms)
        if cache_ms is not None:
            self.cache_ms = cache_ms if self.cache_ms is None else (self.alpha * cache_ms + (1 - self.alpha) * self.cache_ms)

    def decide(self, amort_cache_ms: float, margin_ratio: float) -> Tuple[str, Dict[str, float]]:
        """
        Decide CACHE vs COMPUTE using EMA estimates:
          saved = prefix_ms - (cache_ms + amort_cache_ms)
          use cache if saved/prefix_ms >= margin_ratio
        """
        prefix_ms = float(self.prefix_ms) if self.prefix_ms is not None else float("nan")
        cache_ms = float(self.cache_ms) if self.cache_ms is not None else float("nan")
        cost_cache = cache_ms + float(amort_cache_ms)

        if not np.isfinite(prefix_ms) or not np.isfinite(cache_ms) or prefix_ms <= 0:
            # if no measurements yet, default to CACHE after transition
            return "CACHE", {
                "gate_prefix_ms_ema": prefix_ms,
                "gate_cache_ms_ema": cache_ms,
                "gate_amort_cache_ms": float(amort_cache_ms),
                "gate_saved_ms": float("nan"),
                "gate_saved_ratio": float("nan"),
            }

        saved = prefix_ms - cost_cache
        saved_ratio = saved / prefix_ms

        mode = "CACHE" if saved_ratio >= margin_ratio else "COMPUTE"
        return mode, {
            "gate_prefix_ms_ema": prefix_ms,
            "gate_cache_ms_ema": cache_ms,
            "gate_amort_cache_ms": float(amort_cache_ms),
            "gate_saved_ms": float(saved),
            "gate_saved_ratio": float(saved_ratio),
        }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--transition_epoch", type=int, default=170)

    # gate controls
    ap.add_argument("--gate_calib_batches", type=int, default=10, help="batches used to measure prefix/cache time per epoch")
    ap.add_argument("--gate_alpha", type=float, default=0.2, help="EMA smoothing")
    ap.add_argument("--gate_margin", type=float, default=0.10, help="require saved_ratio >= margin to use cache")

    # suffix optimizer
    ap.add_argument("--suffix_lr", type=float, default=0.03)

    ap.add_argument("--seed", type=int, default=0)
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

    prefix, suffix = split_resnet18(model, args.boundary_idx)
    prefix.to(device)
    suffix.to(device)

    # loaders:
    # - before transition: augmented training
    # - after transition: we use NO-AUG for both CACHE and COMPUTE modes for fair comparison with cached features
    train_aug = make_train_loader(args.data_dir, args.batch_size, args.num_workers, aug=True)
    train_noaug = make_train_loader(args.data_dir, args.batch_size, args.num_workers, aug=False)
    test_loader = make_test_loader(args.data_dir, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    transitioned = False

    # cache state
    F_cache: Optional[torch.Tensor] = None
    y_cache: Optional[torch.Tensor] = None
    cache_build_s: float = 0.0

    gate = GateEMA(alpha=args.gate_alpha)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = now()
        log: Dict[str, object] = {"epoch": int(epoch)}

        # ---------------------
        # Phase: pre-transition
        # ---------------------
        if (not transitioned) and epoch < args.transition_epoch:
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for xb, yb in train_aug:
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

            print(f"Epoch {epoch:03d} | pre | time={log['epoch_time_s']:.2f}s | test_acc={log['test_acc']:.2f}")
            continue

        # ---------------------
        # Transition once
        # ---------------------
        if (not transitioned) and epoch >= args.transition_epoch:
            transitioned = True
            freeze_stats = freeze_prefix_params(model, args.boundary_idx)
            log.update({"transitioned": True, "transition_epoch": int(epoch), **freeze_stats})

            # build full cache (no-aug)
            F_cache, y_cache, cache_build_s = build_full_prefix_cache(prefix, train_noaug, device)
            log["cache_build_s"] = float(cache_build_s)
            log["cache_bytes"] = int(F_cache.numel() * F_cache.element_size())
            log["cache_shape"] = list(F_cache.shape)

            # switch optimizer to suffix-only
            suffix_params = [p for p in suffix.parameters() if p.requires_grad]
            if len(suffix_params) == 0:
                raise RuntimeError("Suffix has no trainable params after prefix freezing. Check boundary_idx/freezing.")
            optimizer = optim.SGD(suffix_params, lr=args.suffix_lr, momentum=0.9, weight_decay=5e-4)

        # ---------------------
        # Post-transition: break-even gating per epoch
        # ---------------------
        assert F_cache is not None and y_cache is not None

        num_batches = len(train_noaug)
        epochs_left = max(1, args.epochs - epoch + 1)
        # amortize cache build over remaining epochs (fair cost attribution)
        amort_cache_ms = (cache_build_s / (epochs_left * max(1, num_batches))) * 1000.0

        # calibration measurements this epoch
        calib_batches = max(1, int(args.gate_calib_batches))

        # measure prefix forward (COMPUTE) on first calib_batches
        prefix_ms_sum = 0.0
        cache_ms_sum = 0.0
        measured = 0

        # We'll iterate a calibration slice of the no-aug loader
        with torch.no_grad():
            prefix.eval()
        suffix.train()

        for bi, (xb, yb) in enumerate(train_noaug):
            if bi >= calib_batches:
                break

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # ---- measure prefix compute time (prefix forward only) ----
            cuda_sync_if_needed(device)
            t1 = now()
            feat = prefix(xb)
            cuda_sync_if_needed(device)
            prefix_ms_sum += (now() - t1) * 1000.0

            # ---- measure cache transfer time (CPU->GPU only) ----
            # take matching number of cached feats (slice is fine for time)
            feat_cpu = F_cache[bi * args.batch_size : (bi + 1) * args.batch_size]
            cuda_sync_if_needed(device)
            t2 = now()
            _ = feat_cpu.to(device, non_blocking=True)
            cuda_sync_if_needed(device)
            cache_ms_sum += (now() - t2) * 1000.0

            measured += 1

        if measured > 0:
            gate.update(prefix_ms_sum / measured, cache_ms_sum / measured)

        mode, gate_log = gate.decide(amort_cache_ms=amort_cache_ms, margin_ratio=args.gate_margin)
        log.update(gate_log)
        log["gate_mode"] = mode

        # ---------------------
        # Train epoch in chosen mode
        # ---------------------
        loss_sum, correct, total = 0.0, 0, 0

        if mode == "CACHE":
            # train on cached features
            cached_ds = CachedFeatDataset(F_cache, y_cache)
            cached_loader = DataLoader(cached_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

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

        else:
            # COMPUTE mode: compute prefix on-the-fly, but do NOT backprop through prefix
            prefix.eval()
            for xb, yb in train_noaug:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.no_grad():
                    feat = prefix(xb)

                optimizer.zero_grad(set_to_none=True)
                logits = suffix(feat)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)

        log["train_loss"] = float(loss_sum / max(1, total))
        log["train_acc"] = float(100.0 * correct / max(1, total))
        log["test_acc"] = float(eval_full(model, test_loader, device))
        log["epoch_time_s"] = float(now() - ep_t0)

        with open(args.log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        print(
            f"Epoch {epoch:03d} | post | gate={mode} | time={log['epoch_time_s']:.2f}s "
            f"| saved_ratio={log.get('gate_saved_ratio', float('nan')):.3f} | test_acc={log['test_acc']:.2f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
