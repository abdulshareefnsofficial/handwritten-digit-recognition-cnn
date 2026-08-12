"""
Training pipeline for Handwritten Digit Recognition CNN.

Usage:
    python train.py --epochs 10 --batch-size 64 --lr 0.001
"""

import argparse
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.model import DigitCNN
from src.dataset import get_dataloaders
from src.utils import set_seed, get_device, save_checkpoint, plot_training_history


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Executes single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for numerical stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Operating device: {device}")

    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=not args.no_augment
    )

    print(f"[INFO] Dataset split: {len(train_loader.dataset)} train | {len(val_loader.dataset)} validation | {len(test_loader.dataset)} test samples.")

    # Model, Loss, Optimizer, Scheduler
    model = DigitCNN(num_classes=10, dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0
    checkpoint_path = f"{args.save_dir}/best_model.pth"

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[SUMMARY] Training completed in {total_time:.2f} seconds.")
    print(f"[SUMMARY] Best Validation Accuracy: {best_val_acc:.2f}%")

    # Evaluate best checkpoint on test dataset
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"[EVALUATION] Final Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    # Plot metrics
    plot_training_history(history, save_path=f"{args.save_dir}/training_history.png")
    print(f"[INFO] History chart saved to: {args.save_dir}/training_history.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for MNIST Handwritten Digit Classification")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for dataset storage")
    parser.add_argument("--save-dir", type=str, default="./artifacts", help="Output directory for checkpoints/plots")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
