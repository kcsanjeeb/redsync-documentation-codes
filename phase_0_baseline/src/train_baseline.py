import argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import cifar10_loaders
from src.models.resnet_cifar import resnet56_cifar
from src.utils import set_seed, set_determinism, JsonlLogger

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
    ap.add_argument("--log_path", type=str, default="runs/baseline/train.jsonl")
    ap.add_argument("--no_aug", action="store_true", help="Disable data augmentation (deterministic preprocessing)")
    args = ap.parse_args()

    set_seed(args.seed)
    set_determinism(args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size, no_aug=args.no_aug)

    model = resnet56_cifar(num_classes=10).to(device)
    ce = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # CIFAR ResNet typically uses multi-step decay (e.g., 100, 150)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    logger = JsonlLogger(args.log_path)

    best_acc = 0.0
    global_step = 0

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

            global_step += 1
            if global_step % 50 == 0:
                pbar.set_postfix(loss=running_loss/total, acc=correct/total, lr=optimizer.param_groups[0]["lr"])

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
        })

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% | "
              f"test_loss={test_loss:.4f}, test_acc={test_acc*100:.2f}% | "
              f"time={epoch_time:.2f}s | best={best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
