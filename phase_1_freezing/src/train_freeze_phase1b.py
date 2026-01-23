# src/train_freeze_phase1b.py
#
# Phase 1b (fixed): Egeria-style freezing with a CPU reference model,
# but WITHOUT stalling the GPU every N steps.
#
# Key fix vs your previous version:
# - We compute plasticity ONLY once per epoch
# - We use a small fixed "probe batch" (e.g., 32 samples) captured at startup
# - We update the CPU reference periodically (every K epochs)
#
# Run:
#   python -m src.train_freeze_phase1b \
#     --epochs 200 --batch_size 128 \
#     --bootstrap_epochs 60 \
#     --ref_update_every_epochs 5 \
#     --probe_bs 32 \
#     --W 20 --slope_T 1e-4 --required_hits 2 \
#     --max_freeze_modules 20 \
#     --log_path runs/phase1b_freeze/train.jsonl

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import cifar10_loaders
from .models.resnet_cifar import resnet56_cifar
from .utils import set_seed, set_determinism, JsonlLogger
from .egeria.modules import (
    build_resnet56_modules,
    freeze_module,
    forward_through_modules,
)
from .egeria.controller import (
    FreezeController,
    make_cpu_reference,
    plasticity_l2,
)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_path", type=str, default="runs/phase1b_freeze/train.jsonl")

    # Egeria-ish knobs
    ap.add_argument("--bootstrap_epochs", type=int, default=60)     # no freezing early
    ap.add_argument("--ref_update_every_epochs", type=int, default=5)
    ap.add_argument("--probe_bs", type=int, default=32)            # probe batch size
    ap.add_argument("--W", type=int, default=20)                   # history window
    ap.add_argument("--slope_T", type=float, default=1e-4)         # slope threshold
    ap.add_argument("--required_hits", type=int, default=2)        # consecutive "stable" windows
    ap.add_argument("--max_freeze_modules", type=int, default=20)  # safety cap
    args = ap.parse_args()

    set_seed(args.seed)
    set_determinism(args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size)

    # Grab a fixed probe batch ONCE (small) to make plasticity checks cheap & consistent
    probe_x, _ = next(iter(train_loader))
    probe_x = probe_x[: args.probe_bs].contiguous().cpu()  # keep probe on CPU by default

    # Model + modules
    model = resnet56_cifar(num_classes=10).to(device)
    modules = build_resnet56_modules(model)

    # Controller + CPU reference model
    controller = FreezeController(W=args.W, slope_T=args.slope_T, required_hits=args.required_hits)
    ref_model = make_cpu_reference(model)
    ref_modules = build_resnet56_modules(ref_model)

    ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    logger = JsonlLogger(args.log_path)

    frozen_upto = -1
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Periodically refresh reference model from the current training model
        if epoch % args.ref_update_every_epochs == 0:
            ref_model = make_cpu_reference(model)
            ref_modules = build_resnet56_modules(ref_model)

        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", ncols=110)
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

        # ---- Plasticity check ONCE PER EPOCH (cheap, avoids stalling training) ----
        last_pval = None
        last_slope = None
        last_hits = None
        froze_this_epoch = False

        if epoch >= args.bootstrap_epochs:
            next_idx = frozen_upto + 1
            can_freeze_more = (next_idx < len(modules)) and (next_idx < args.max_freeze_modules)
            if can_freeze_more:
                boundary_idx = next_idx

                # Train model activation on GPU for probe batch
                model.eval()
                with torch.no_grad():
                    _, act_train = forward_through_modules(
                        modules, probe_x.to(device, non_blocking=True), stop_at=boundary_idx
                    )

                # Reference activation on CPU for probe batch
                with torch.no_grad():
                    _, act_ref = forward_through_modules(ref_modules, probe_x, stop_at=boundary_idx)

                last_pval = plasticity_l2(act_train.detach().cpu(), act_ref)
                info = controller.update(last_pval)

                if info.get("ready", False):
                    last_slope = info.get("slope", None)
                    last_hits = info.get("hit_count", None)

                if info.get("ready", False) and info.get("freeze_now", False):
                    frozen_upto += 1
                    freeze_module(modules[frozen_upto])
                    froze_this_epoch = True
                    # reset history for next module
                    controller = FreezeController(W=args.W, slope_T=args.slope_T, required_hits=args.required_hits)

                model.train()

        # -------------------------------------------------------------------------

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)

        logger.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_acc,
            "frozen_modules": frozen_upto + 1,
            "total_modules": len(modules),
            "froze_this_epoch": froze_this_epoch,
            "last_plasticity": last_pval,
            "last_slope": last_slope,
            "last_hit_count": last_hits,
            "probe_bs": args.probe_bs,
            "W": args.W,
            "slope_T": args.slope_T,
            "required_hits": args.required_hits,
            "bootstrap_epochs": args.bootstrap_epochs,
            "ref_update_every_epochs": args.ref_update_every_epochs,
        })

        print(
            f"Epoch {epoch:03d} | test_acc={test_acc*100:.2f}% | "
            f"time={epoch_time:.2f}s | frozen={frozen_upto+1}/{len(modules)} "
            f"| probe_p={None if last_pval is None else f'{last_pval:.5f}'} "
            f"| slope={None if last_slope is None else f'{last_slope:.3e}'} "
            f"| froze={froze_this_epoch}"
        )


if __name__ == "__main__":
    main()
