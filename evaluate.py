"""
Evaluation script for trained MNIST CNN model.
Generates classification report, confusion matrix, and misclassification visualization.

Usage:
    python evaluate.py --model-path artifacts/best_model.pth
"""

import argparse
import os
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.model import DigitCNN
from src.dataset import get_dataloaders
from src.utils import get_device, load_checkpoint


@torch.no_grad()
def run_evaluation(model_path: str, data_dir: str, save_dir: str) -> None:
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Load test dataset
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=128, num_workers=0, augment=False)

    # Instantiate and load model
    model = DigitCNN().to(device)
    checkpoint = load_checkpoint(model_path, model, device=device)
    print(f"[INFO] Loaded checkpoint trained for {checkpoint.get('epoch', 'N/A')} epochs.")
    model.eval()

    all_preds = []
    all_targets = []
    misclassified_images = []
    misclassified_preds = []
    misclassified_targets = []

    for images, targets in test_loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        _, preds = outputs.max(1)

        preds_cpu = preds.cpu().numpy()
        targets_cpu = targets.numpy()

        all_preds.extend(preds_cpu)
        all_targets.extend(targets_cpu)

        # Collect misclassified examples
        incorrect_mask = (preds_cpu != targets_cpu)
        if np.any(incorrect_mask):
            for img, p, t in zip(images[incorrect_mask], preds_cpu[incorrect_mask], targets_cpu[incorrect_mask]):
                if len(misclassified_images) < 16:
                    misclassified_images.append(img.squeeze(0).numpy())
                    misclassified_preds.append(p)
                    misclassified_targets.append(t)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Quantitative summary
    accuracy = (all_preds == all_targets).mean() * 100.0
    print(f"\n==========================================")
    print(f"       MODEL EVALUATION RESULTS           ")
    print(f"==========================================")
    print(f"Overall Test Accuracy: {accuracy:.2f}%\n")
    print(classification_report(all_targets, all_preds, digits=4))

    os.makedirs(save_dir, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = [str(i) for i in range(10)]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix - MNIST Digit Recognition",
           ylabel="True Label",
           xlabel="Predicted Label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix heatmap to: {cm_path}")

    # 2. Misclassifications Plot
    if misclassified_images:
        fig, axes = plt.subplots(4, 4, figsize=(8, 8), dpi=300)
        fig.suptitle("Sample Misclassified Digits (True vs Predicted)", fontsize=12, fontweight="bold")
        for i, ax in enumerate(axes.flat):
            if i < len(misclassified_images):
                # Denormalize image for rendering
                img = misclassified_images[i] * 0.3081 + 0.1307
                ax.imshow(img, cmap="gray")
                ax.set_title(f"True: {misclassified_targets[i]} | Pred: {misclassified_preds[i]}",
                             fontsize=9, color="red")
            ax.axis("off")
        plt.tight_layout()
        misc_path = os.path.join(save_dir, "misclassified_samples.png")
        plt.savefig(misc_path)
        plt.close(fig)
        print(f"[INFO] Saved misclassified samples grid to: {misc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trained MNIST CNN Model")
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data", help="MNIST data directory")
    parser.add_argument("--save-dir", type=str, default="./artifacts", help="Report output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.model_path, args.data_dir, args.save_dir)
