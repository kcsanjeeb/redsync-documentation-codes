# src/train_phase4_prefixcache_transition_v2_sched.py
from __future__ import annotations

import argparse
import os
import time
import json
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models


def now() -> float:
    return time.perf_counter()


def cuda_sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# -------------------------
# Cutpoints for ResNet-18 (torchvision-style naming)
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


def split_resnet18(model: nn.Module, boundary_idx: int) -> Tuple[nn.Module, nn.Module]:
    """
    Returns (prefix_forward_module, suffix_forward_module).
    These modules route through the SAME underlying model for forward only.

    IMPORTANT:
      Do NOT freeze via prefix.parameters() because that would freeze the whole model.
      Freeze via model.named_parameters() and name prefixes (see freeze_prefix_params()).
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
            x = self.m.maxpool(x)  # CIFAR config will set this to Identity()

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

    class Suffix(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, feat):
            x = feat
            layer_name, block_idx = CUTPOINTS[boundary_idx]

            layers = ["layer1", "layer2", "layer3", "layer4"]
            passed = False
            for lname in layers:
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
    """
    Returns name prefixes of parameters that belong to the prefix up to boundary_idx.
    Works for torchvision ResNet naming: conv1, bn1, layer1.0, layer1.1, layer2.0, ...
    """
    assert boundary_idx in CUTPOINTS, f"Unknown boundary_idx={boundary_idx}"
    layer_name, block_idx = CUTPOINTS[boundary_idx]

    prefixes = ["conv1", "bn1"]  # stem params

    if layer_name == "stem":
        return prefixes

    order = ["layer1", "layer2", "layer3", "layer4"]
    for lname in order:
        if lname == layer_name:
            for bi in range(block_idx + 1):
                prefixes.append(f"{lname}.{bi}")
            break
        else:
            # ResNet-18 has 2 blocks per layer: 0 and 1
            prefixes.append(f"{lname}.0")
            prefixes.append(f"{lname}.1")

    return prefixes


def freeze_prefix_params(model: nn.Module, boundary_idx: int) -> Dict[str, int]:
    """
    Freeze only parameters that belong to the prefix (by name prefixes).
    Returns stats for logging.
    """
    prefs = prefix_param_prefixes(boundary_idx)

    frozen_tensors = 0
    frozen_params = 0

    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefs):
            if p.requires_grad:
                p.requires_grad = False
                frozen_tensors += 1
                frozen_params += p.numel()

    return {
        "frozen_param_tensors": int(frozen_tensors),
        "frozen_param_count": int(frozen_params),
    }


class CachedFeatureDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        assert feats.shape[0] == labels.shape[0]
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


@torch.no_grad()
def build_feature_cache(
    prefix: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Builds a full-dataset cache of prefix features on CPU.
    """
    prefix.eval()
    t0 = now()

    feat_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        cuda_sync_if_needed(device)
        feats = prefix(xb)
        cuda_sync_if_needed(device)

        feats_cpu = feats.detach().to("cpu")
        if out_dtype is not None:
            feats_cpu = feats_cpu.to(out_dtype)

        feat_list.append(feats_cpu)
        y_list.append(yb.detach().to("cpu"))

    feats_all = torch.cat(feat_list, dim=0).contiguous()
    y_all = torch.cat(y_list, dim=0).contiguous()

    build_s = now() - t0
    bytes_total = feats_all.numel() * feats_all.element_size()

    log = {
        "cache_build_s": float(build_s),
        "cache_feat_shape": list(feats_all.shape),
        "cache_feat_dtype": str(feats_all.dtype),
        "cache_feat_bytes": int(bytes_total),
        "cache_num_samples": int(feats_all.shape[0]),
    }
    return feats_all, y_all, log


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


def make_suffix_scheduler(
    optimizer: optim.Optimizer,
    sched_name: str,
    epochs_after_transition: int,
    step_size: int,
    gamma: float,
):
    sched_name = (sched_name or "none").lower()
    if sched_name == "none":
        return None
    if sched_name == "cosine":
        # smooth decay to 0 by the end of suffix-only stage
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs_after_transition))
    if sched_name == "step":
        # drop lr by gamma every step_size epochs after transition
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, step_size), gamma=float(gamma))
    raise ValueError(f"Unknown --suffix_sched={sched_name}. Use: none|cosine|step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    # Transition controls
    ap.add_argument("--bootstrap_epochs", type=int, default=120)
    ap.add_argument("--transition_epoch", type=int, default=150)
    ap.add_argument("--boundary_idx", type=int, default=3)

    # Suffix optimizer config
    ap.add_argument("--suffix_lr", type=float, default=0.01, help="Suffix optimizer LR after transition")

    # NEW: suffix scheduler
    ap.add_argument("--suffix_sched", type=str, default="none", choices=["none", "cosine", "step"],
                    help="Suffix LR scheduler after transition (none|cosine|step)")
    ap.add_argument("--suffix_step_size", type=int, default=10, help="StepLR step_size (epochs) after transition")
    ap.add_argument("--suffix_gamma", type=float, default=0.3, help="StepLR gamma after transition")

    # Cache dtype
    ap.add_argument("--cache_dtype", type=str, default="fp32", choices=["fp32", "fp16"],
                    help="Cached feature dtype on CPU (fp32 is safest; fp16 needs AMP/autocast in suffix)")

    # Logging
    ap.add_argument("--log_path", type=str, required=True)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-friendly ResNet-18 (torchvision backbone adapted)
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device)

    trainloader, testloader = make_cifar_loaders(
        args.data_dir, args.batch_size, args.num_workers, aug=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    transitioned = False
    cached_train_loader: Optional[DataLoader] = None
    suffix_scheduler = None

    prefix, suffix = split_resnet18(model, args.boundary_idx)
    prefix.to(device)
    suffix.to(device)

    cache_out_dtype = torch.float32 if args.cache_dtype == "fp32" else torch.float16

    for epoch in range(1, args.epochs + 1):
        ep_t0 = now()
        transition_log: Dict = {}

        # Transition block
        if (not transitioned) and (epoch >= args.transition_epoch):
            transitioned = True

            freeze_stats = freeze_prefix_params(model, args.boundary_idx)

            # Build cache with deterministic transform (no random augmentation)
            cache_build_loader, _ = make_cifar_loaders(
                args.data_dir, args.batch_size, args.num_workers, aug=False
            )

            feats, labels, cache_log = build_feature_cache(
                prefix, cache_build_loader, device, out_dtype=cache_out_dtype
            )

            cached_ds = CachedFeatureDataset(feats, labels)
            cached_train_loader = DataLoader(
                cached_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            suffix_params = [p for p in suffix.parameters() if p.requires_grad]
            if len(suffix_params) == 0:
                raise RuntimeError("Suffix has no trainable params; check freezing logic.")

            optimizer = optim.SGD(
                suffix_params,
                lr=float(args.suffix_lr),
                momentum=0.9,
                weight_decay=5e-4,
            )

            epochs_after = max(0, args.epochs - epoch)
            suffix_scheduler = make_suffix_scheduler(
                optimizer=optimizer,
                sched_name=args.suffix_sched,
                epochs_after_transition=epochs_after,
                step_size=args.suffix_step_size,
                gamma=args.suffix_gamma,
            )

            transition_log.update({
                "transitioned": True,
                "transition_epoch": int(epoch),
                "boundary_idx": int(args.boundary_idx),
                "suffix_lr": float(args.suffix_lr),
                "suffix_sched": str(args.suffix_sched),
                "suffix_step_size": int(args.suffix_step_size),
                "suffix_gamma": float(args.suffix_gamma),
                "cache_dtype": str(args.cache_dtype),
                **freeze_stats,
                **cache_log,
            })

        # Train
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        if transitioned and cached_train_loader is not None:
            suffix.train()
            use_autocast = (device.type == "cuda" and args.cache_dtype == "fp16")
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_autocast)

            for feat_cpu, y_cpu in cached_train_loader:
                feat = feat_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    logits = suffix(feat)
                    loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * y.size(0)
                train_correct += (logits.argmax(1) == y).sum().item()
                train_total += y.size(0)

            # Step scheduler once per epoch in suffix-only stage
            if suffix_scheduler is not None:
                suffix_scheduler.step()

        else:
            model.train()
            for xb, yb in trainloader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = 100.0 * train_correct / max(1, train_total)

        test_acc = eval_full(model, testloader, device)

        ep_s = now() - ep_t0

        # Current LR for logging
        cur_lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0

        row = {
            "epoch": int(epoch),
            "epoch_time_s": float(ep_s),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "transitioned": bool(transitioned),
            "boundary_idx": int(args.boundary_idx),
            "bootstrap_epochs": int(args.bootstrap_epochs),
            "lr": float(cur_lr),
        }
        row.update(transition_log)

        with open(args.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch:03d} | train_acc={train_acc:.2f} | test_acc={test_acc:.2f} "
            f"| time={ep_s:.2f}s | transitioned={transitioned} | lr={cur_lr:.5f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
