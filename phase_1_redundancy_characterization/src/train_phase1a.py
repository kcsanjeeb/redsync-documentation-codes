import argparse
import time

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
    correct, total = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def _flatten_params(module: nn.Module) -> torch.Tensor:
    """Return a 1D CPU tensor of all params in a module (detached)."""
    ps = [p.detach().reshape(-1).cpu() for p in module.parameters()]
    if len(ps) == 0:
        return torch.empty(0)
    return torch.cat(ps, dim=0)


@torch.no_grad()
def _collect_module_activations(modules, x: torch.Tensor):
    """
    Run sequentially through modules and collect outputs after each module.
    Returns a list of CPU tensors (one per module).
    """
    acts = []
    h = x
    for m in modules:
        h = m(h)
        acts.append(h.detach().cpu())
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_path", type=str, default="runs/phase1a_measure/train.jsonl")

    # Optional: probe batch activation stability logging (not required for Phase-1A)
    ap.add_argument("--log_probe_cosine", action="store_true")
    ap.add_argument("--probe_bs", type=int, default=32)

    args = ap.parse_args()

    set_seed(args.seed)
    set_determinism(args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size)

    model = resnet56_cifar(num_classes=10).to(device)
    modules = build_resnet56_modules(model)

    ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    logger = JsonlLogger(args.log_path)

    # Fixed probe batch ONCE (only if activation cosine is enabled)
    probe_x = None
    if args.log_probe_cosine:
        px, _ = next(iter(train_loader))
        probe_x = px[: args.probe_bs].contiguous()

    # Previous epoch snapshots for ΔW and optional cosine
    prev_params = [_flatten_params(m) for m in modules]
    prev_acts = None

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

        # ---- Phase-1A: per-module relative weight delta ----
        rel_deltas = []
        for i, m in enumerate(modules):
            cur = _flatten_params(m)
            prev = prev_params[i]
            if prev.numel() == 0 or cur.numel() == 0:
                rel = 0.0
            else:
                denom = prev.norm(p=2).item() + 1e-12
                rel = (cur.sub(prev).norm(p=2).item()) / denom
            rel_deltas.append(rel)
            prev_params[i] = cur
        # ---------------------------------------------------

        # Optional: probe-batch activation cosine stability (epoch-to-epoch)
        act_cos = None
        if args.log_probe_cosine:
            model.eval()
            with torch.no_grad():
                acts = _collect_module_activations(modules, probe_x.to(device, non_blocking=True))
            if prev_acts is None:
                act_cos = [None] * len(acts)
            else:
                cos_list = []
                for a, b in zip(acts, prev_acts):
                    a1 = a.reshape(-1).float()
                    b1 = b.reshape(-1).float()
                    if a1.numel() == 0 or b1.numel() == 0:
                        cos_list.append(None)
                    else:
                        cos_list.append(F.cosine_similarity(a1, b1, dim=0).item())
                act_cos = cos_list
            prev_acts = acts
            model.train()

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_acc,
            "total_modules": len(modules),
            "rel_weight_delta_modules": rel_deltas,
        }

        if args.log_probe_cosine:
            row.update(
                {
                    "probe_bs": args.probe_bs,
                    "act_cos_sim_modules": act_cos,
                }
            )

        logger.log(row)

        print(
            f"Epoch {epoch:03d} | test_acc={test_acc*100:.2f}% | "
            f"time={epoch_time:.2f}s"
        )


if __name__ == "__main__":
    main()
