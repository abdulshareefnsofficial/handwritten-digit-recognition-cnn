"""
Master Orchestration Script for Handwritten Digit Recognition System.

Commands:
    python main.py --action train       : Train the CNN model on MNIST dataset
    python main.py --action evaluate    : Compute metrics, plots, and confusion matrix
    python main.py --action gui         : Launch interactive drawing GUI application
    python main.py --action predict --image digit.png : Predict digit from single image file
"""

import argparse
import sys
from train import run_training, parse_args as parse_train_args
from evaluate import run_evaluation
from predict import predict_digit


def main():
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition System (CNN)")
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "evaluate", "gui", "predict"],
        default="train",
        help="Action to perform: train, evaluate, gui, or predict"
    )
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to input image for prediction")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--save-dir", type=str, default="./artifacts", help="Directory for artifacts")

    args, unknown = parser.parse_known_args()

    if args.action == "train":
        train_args = parse_train_args()
        run_training(train_args)

    elif args.action == "evaluate":
        run_evaluation(args.model_path, args.data_dir, args.save_dir)

    elif args.action == "predict":
        if not args.image:
            print("Error: --image argument is required for prediction action.")
            sys.exit(1)
        pred, probs = predict_digit(args.image, args.model_path)
        print(f"Predicted Digit: {pred} (Confidence: {probs[pred]*100:.2f}%)")

    elif args.action == "gui":
        from gui_app import DigitRecognizerGUI
        import tkinter as tk
        root = tk.Tk()
        app = DigitRecognizerGUI(root, args.model_path)
        root.mainloop()


if __name__ == "__main__":
    main()
