"""
Inference script for classifying single handwritten digit images.

Usage:
    python predict.py --image path/to/digit.png --model-path artifacts/best_model.pth
"""

import argparse
import os
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from src.model import DigitCNN
from src.dataset import MNIST_MEAN, MNIST_STD
from src.utils import get_device, load_checkpoint


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Loads image, converts to 28x28 grayscale tensor matching MNIST dataset format.
    Automatically handles color inversion (MNIST standard: light digit on dark background).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load as grayscale
    img = Image.open(image_path).convert("L")

    # Auto-invert if image background is light (e.g., black pen on white paper)
    stat = np.array(img)
    if np.mean(stat) > 127:
        img = ImageOps.invert(img)

    # Resize to 28x28 with anti-aliasing
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_arr = np.array(img, dtype=np.float32) / 255.0

    # Normalize with standard MNIST statistics
    img_norm = (img_arr - MNIST_MEAN) / MNIST_STD

    # Convert to Tensor (B, C, H, W) -> (1, 1, 28, 28)
    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_digit(image_path: str, model_path: str) -> Tuple[int, np.ndarray]:
    device = get_device()
    tensor = preprocess_image(image_path).to(device)

    model = DigitCNN().to(device)
    load_checkpoint(model_path, model, device=device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_digit = int(np.argmax(probabilities))
    return predicted_digit, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict handwritten digit from image file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pth", help="Path to trained model checkpoint")
    args = parser.parse_args()

    pred, probs = predict_digit(args.image, args.model_path)
    
    print(f"\n==========================================")
    print(f"       HANDWRITTEN DIGIT PREDICTION       ")
    print(f"==========================================")
    print(f"Input Image : {args.image}")
    print(f"Predicted   : DIGIT {pred}")
    print(f"Confidence  : {probs[pred] * 100:.2f}%\n")
    print("Class Probabilities:")
    for digit, prob in enumerate(probs):
        bar = "█" * int(prob * 20)
        print(f"  Digit {digit}: {prob * 100:5.2f}% {bar}")


if __name__ == "__main__":
    main()
