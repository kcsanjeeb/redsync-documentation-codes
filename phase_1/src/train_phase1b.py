import argparse
import time
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .data import cifar10_loaders
from .models.resnet_cifar import resnet56_cifar
from .utils import set_seed, set_determinism, JsonlLogger
from .egeria.modules import build_resnet56_modules


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct, total = 0, 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def _collect_module_activations(modules, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Run sequentially through modules and collect outputs after each module.
    Returns list of CPU tensors, one per module.
    """
    acts = []
    h = x
    for m in modules:
        h = m(h)
        acts.append(h.detach().cpu())
    return acts


def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """
    Cosine similarity between two activation tensors, computed on flattened vectors.
    Returns None if tensors are empty.
    """
    a1 = a.reshape(-1).float()
    b1 = b.reshape(-1).float()
    if a1.numel() == 0 or b1.numel() == 0:
        return None
    return F.cosine_similarity(a1, b1, dim=0).item()


def _mean_ignore_none(xs: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in xs if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _depth_buckets(n: int):
    """
    Split module indices into 3 contiguous depth buckets (shallow/mid/deep).
    This is index-based; it’s reproducible and verifiable even if names differ.
    """
    if n <= 3:
        return list(range(n)), [], []
    a = n // 3
    b = 2 * (n // 3)
    shallow = list(range(0, a))
    mid = list(range(a, b))
    deep = list(range(b, n))
    return shallow, mid, deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_path", type=str, default="runs/phase1b_measure/train.jsonl")

    # Probe batch for functional stability measurements
    ap.add_argument("--probe_bs", type=int, default=32)

    args = ap.parse_args()

    set_seed(args.seed)
    set_determinism(args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size)

    # Model + module decomposition
    model = resnet56_cifar(num_classes=10).to(device)
    modules = build_resnet56_modules(model)
    n_mod = len(modules)

    shallow_idx, mid_idx, deep_idx = _depth_buckets(n_mod)

    # Fixed probe batch ONCE (consistent epoch-to-epoch comparisons)
    probe_x, _ = next(iter(train_loader))
    probe_x = probe_x[: args.probe_bs].contiguous()

    ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    logger = JsonlLogger(args.log_path)

    # Previous epoch activations on probe batch
    prev_acts: Optional[List[torch.Tensor]] = None
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", ncols=100)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        scheduler.step()
        epoch_time = time.time() - t0

        # ---- Phase-1B: functional stability on fixed probe batch ----
        model.eval()
        with torch.no_grad():
            acts = _collect_module_activations(modules, probe_x.to(device, non_blocking=True))

        if prev_acts is None:
            cos_list: List[Optional[float]] = [None] * n_mod
        else:
            cos_list = [_cosine_flat(a, b) for a, b in zip(acts, prev_acts)]

        prev_acts = acts
        model.train()
        # ------------------------------------------------------------

        # Depth summaries (optional but very useful for Phase-1B narrative)
        shallow_cos = _mean_ignore_none([cos_list[i] for i in shallow_idx]) if shallow_idx else None
        mid_cos = _mean_ignore_none([cos_list[i] for i in mid_idx]) if mid_idx else None
        deep_cos = _mean_ignore_none([cos_list[i] for i in deep_idx]) if deep_idx else None
        all_cos = _mean_ignore_none(cos_list)

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)

        row: Dict[str, Any] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_acc,
            "total_modules": n_mod,
            "probe_bs": args.probe_bs,

            # Main Phase-1B signal: per-module cosine similarity (epoch t vs t-1)
            "act_cos_sim_modules": cos_list,

            # Helpful summaries for docs (shallow/mid/deep by module index thirds)
            "act_cos_mean_all": all_cos,
            "act_cos_mean_shallow": shallow_cos,
            "act_cos_mean_mid": mid_cos,
            "act_cos_mean_deep": deep_cos,

            # Bucket definitions so readers can reproduce exactly
            "bucket_shallow_idx": shallow_idx,
            "bucket_mid_idx": mid_idx,
            "bucket_deep_idx": deep_idx,
        }

        logger.log(row)

        print(
            f"Epoch {epoch:03d} | test_acc={test_acc*100:.2f}% | time={epoch_time:.2f}s | "
            f"cos(all)={None if all_cos is None else f'{all_cos:.4f}'} | "
            f"cos(deep)={None if deep_cos is None else f'{deep_cos:.4f}'}"
        )


if __name__ == "__main__":
    main()
