import argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import cifar10_loaders
from .models.resnet_cifar import resnet56_cifar
from .utils import set_seed, set_determinism, JsonlLogger
from .egeria.modules import build_resnet56_modules, freeze_module

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_path", type=str, default="runs/phase1a_freeze/train.jsonl")

    # Phase1a controls
    ap.add_argument("--freeze_start_epoch", type=int, default=80)
    ap.add_argument("--freeze_every", type=int, default=5)  # freeze one module every N epochs

    args = ap.parse_args()

    set_seed(args.seed)
    set_determinism(args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size)

    model = resnet56_cifar(num_classes=10).to(device)
    ce = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    logger = JsonlLogger(args.log_path)

    modules = build_resnet56_modules(model)
    frozen_upto = -1  # index of last frozen module

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Phase1a: deterministic freezing schedule (TEMP)
        if epoch >= args.freeze_start_epoch:
            should_freeze = ((epoch - args.freeze_start_epoch) % args.freeze_every == 0)
            if should_freeze and frozen_upto + 1 < len(modules):
                frozen_upto += 1
                freeze_module(modules[frozen_upto])

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

            global_step += 1

        scheduler.step()
        epoch_time = time.time() - t0

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
        })

        print(f"Epoch {epoch:03d} | test_acc={test_acc*100:.2f}% | "
              f"time={epoch_time:.2f}s | frozen={frozen_upto+1}/{len(modules)}")

if __name__ == "__main__":
    main()
