# src/train_phase5E_break_even_gating_packed_v1.py
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src.feature_cache_packed import PackedTensorCache


def now() -> float:
    return time.perf_counter()


def cuda_sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            x = self.m.maxpool(x)
            if layer_name == "stem":
                return x
            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, lname)
                for bi, blk in enumerate(layer):
                    x = blk(x)
                    if lname == layer_name and bi == block_idx:
                        return x
            raise RuntimeError()

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

    def decide(self, amort_cache_ms: float, margin_ratio: float):
        prefix_ms = float(self.prefix_ms) if self.prefix_ms is not None else float("nan")
        cache_ms = float(self.cache_ms) if self.cache_ms is not None else float("nan")
        cost_cache = cache_ms + float(amort_cache_ms)
        if not np.isfinite(prefix_ms) or not np.isfinite(cache_ms) or prefix_ms <= 0:
            return "CACHE", {"gate_saved_ratio": float("nan"), "gate_saved_ms": float("nan")}
        saved = prefix_ms - cost_cache
        ratio = saved / prefix_ms
        return ("CACHE" if ratio >= margin_ratio else "COMPUTE"), {"gate_saved_ratio": float(ratio), "gate_saved_ms": float(saved)}


@torch.no_grad()
def build_packed_cache(prefix: nn.Module, loader: DataLoader, device: torch.device, cache: PackedTensorCache) -> Tuple[float, int]:
    """
    Store per-batch features into PackedTensorCache using raw packing.
    Keyed by global batch index.
    Returns: build_time_s, total_bytes
    """
    prefix.eval()
    t0 = now()
    total_bytes = 0
    for bi, (xb, _) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        cuda_sync_if_needed(device)
        feat = prefix(xb)
        cuda_sync_if_needed(device)
        log = cache.capture(f"batch_{bi}", feat)
        total_bytes += int(log.get("packed_bytes", 0))
    return (now() - t0), total_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--transition_epoch", type=int, default=170)

    ap.add_argument("--suffix_lr", type=float, default=0.03)

    ap.add_argument("--gate_calib_batches", type=int, default=10)
    ap.add_argument("--gate_alpha", type=float, default=0.2)
    ap.add_argument("--gate_margin", type=float, default=0.10)

    ap.add_argument("--pin_memory", action="store_true", help="Pin CPU tensors in cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_path", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    prefix, suffix = split_resnet18(model, args.boundary_idx)
    prefix.to(device); suffix.to(device)

    train_aug = make_train_loader(args.data_dir, args.batch_size, args.num_workers, aug=True)
    train_noaug = make_train_loader(args.data_dir, args.batch_size, args.num_workers, aug=False)
    test_loader = make_test_loader(args.data_dir, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    transitioned = False
    cache_build_s = 0.0
    packed_cache: Optional[PackedTensorCache] = None
    gate = GateEMA(alpha=args.gate_alpha)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = now()
        log: Dict[str, object] = {"epoch": int(epoch)}

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

        if (not transitioned) and epoch >= args.transition_epoch:
            transitioned = True
            log["transitioned"] = True
            log["transition_epoch"] = int(epoch)
            log.update(freeze_prefix_params(model, args.boundary_idx))

            # build packed cache (no-aug)
            packed_cache = PackedTensorCache(pin_memory=args.pin_memory, max_entries=100000)
            cache_build_s, total_bytes = build_packed_cache(prefix, train_noaug, device, packed_cache)
            log["cache_build_s"] = float(cache_build_s)
            log["cache_total_bytes"] = int(total_bytes)
            log["cache_batches"] = int(len(train_noaug))

            # suffix-only optimizer
            suffix_params = [p for p in suffix.parameters() if p.requires_grad]
            if len(suffix_params) == 0:
                raise RuntimeError("Suffix has no trainable params after prefix freezing.")
            optimizer = optim.SGD(suffix_params, lr=args.suffix_lr, momentum=0.9, weight_decay=5e-4)

        assert packed_cache is not None

        num_batches = len(train_noaug)
        epochs_left = max(1, args.epochs - epoch + 1)
        amort_cache_ms = (cache_build_s / (epochs_left * max(1, num_batches))) * 1000.0
        log["gate_amort_cache_ms"] = float(amort_cache_ms)

        # calibration
        calib_batches = max(1, int(args.gate_calib_batches))
        prefix_ms_sum = 0.0
        cache_ms_sum = 0.0
        measured = 0

        prefix.eval()
        for bi, (xb, _) in enumerate(train_noaug):
            if bi >= calib_batches:
                break

            xb = xb.to(device, non_blocking=True)

            # prefix compute time
            cuda_sync_if_needed(device)
            t1 = now()
            _ = prefix(xb)
            cuda_sync_if_needed(device)
            prefix_ms_sum += (now() - t1) * 1000.0

            # packed cache load time (unpack + H2D)
            cuda_sync_if_needed(device)
            t2 = now()
            _ = packed_cache.load(f"batch_{bi}", device)["act"]
            cuda_sync_if_needed(device)
            cache_ms_sum += (now() - t2) * 1000.0

            measured += 1

        if measured > 0:
            gate.update(prefix_ms_sum / measured, cache_ms_sum / measured)

        mode, g = gate.decide(amort_cache_ms=amort_cache_ms, margin_ratio=args.gate_margin)
        log["gate_mode"] = mode
        log["gate_prefix_ms_ema"] = float(gate.prefix_ms) if gate.prefix_ms is not None else float("nan")
        log["gate_cache_ms_ema"] = float(gate.cache_ms) if gate.cache_ms is not None else float("nan")
        log.update(g)

        # training epoch
        suffix.train()
        loss_sum, correct, total = 0.0, 0, 0

        if mode == "CACHE":
            # use packed cache per batch (no dataset tensor materialization)
            for bi, (_, yb) in enumerate(train_noaug):
                yb = yb.to(device, non_blocking=True)
                feat = packed_cache.load(f"batch_{bi}", device)["act"]

                optimizer.zero_grad(set_to_none=True)
                logits = suffix(feat)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        else:
            # compute prefix on the fly
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
