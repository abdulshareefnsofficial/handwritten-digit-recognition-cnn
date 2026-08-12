"""
Interactive Tkinter Drawing Application for Real-Time Handwritten Digit Recognition.

Usage:
    python gui_app.py --model-path artifacts/best_model.pth
"""

import argparse
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from src.model import DigitCNN
from src.dataset import MNIST_MEAN, MNIST_STD
from src.utils import get_device, load_checkpoint


class DigitRecognizerGUI:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root = root
        self.root.title("Handwritten Digit Recognition - CNN")
        self.root.geometry("640 x 480")
        self.root.resizable(False, False)

        self.model_path = model_path
        self.device = get_device()
        self.model = None
        self._load_model()

        # Canvas properties
        self.canvas_size = 280
        self.brush_size = 18

        # PIL Image matching canvas drawing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self._build_ui()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Model Error",
                f"Checkpoint file not found at: {self.model_path}\nPlease train the model first using train.py."
            )
            return

        self.model = DigitCNN().to(self.device)
        load_checkpoint(self.model_path, self.model, device=self.device)
        self.model.eval()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        # Main layout frame
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Canvas + Control Buttons
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=5)

        title_label = ttk.Label(left_frame, text="Draw a Digit (0-9):", font=("Segoe UI", 12, "bold"))
        title_label.pack(anchor=tk.W, pady=(0, 5))

        self.canvas = tk.Canvas(
            left_frame, width=self.canvas_size, height=self.canvas_size, bg="black", cursor="pencil"
        )
        self.canvas.pack(pady=5)

        self.canvas.bind("<B1-Motion>", self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        predict_btn = ttk.Button(btn_frame, text="Recognize", command=self._predict)
        predict_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        clear_btn = ttk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas)
        clear_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        # Right Column: Prediction & Probability Bars
        right_frame = ttk.Frame(main_frame, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.pred_label = ttk.Label(
            right_frame, text="Prediction: --", font=("Segoe UI", 18, "bold"), foreground="#0066cc"
        )
        self.pred_label.pack(anchor=tk.W, pady=(0, 5))

        self.conf_label = ttk.Label(
            right_frame, text="Confidence: --%", font=("Segoe UI", 11)
        )
        self.conf_label.pack(anchor=tk.W, pady=(0, 15))

        prob_header = ttk.Label(right_frame, text="Class Probabilities:", font=("Segoe UI", 10, "bold"))
        prob_header.pack(anchor=tk.W, pady=(0, 5))

        # Class probability progress bars
        self.prob_bars = []
        self.prob_labels = []

        for i in range(10):
            row_frame = ttk.Frame(right_frame)
            row_frame.pack(fill=tk.X, pady=2)

            lbl = ttk.Label(row_frame, text=f"Digit {i}:", width=8, font=("Consolas", 10))
            lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row_frame, orient="horizontal", mode="determinate", maximum=100)
            pbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            pct_lbl = ttk.Label(row_frame, text="0.0%", width=6, font=("Consolas", 9))
            pct_lbl.pack(side=tk.RIGHT)

            self.prob_bars.append(pbar)
            self.prob_labels.append(pct_lbl)

    def _on_draw(self, event) -> None:
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _on_release(self, event) -> None:
        self._predict()

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Prediction: --", foreground="#0066cc")
        self.conf_label.config(text="Confidence: --%")
        for pbar, pct_lbl in zip(self.prob_bars, self.prob_labels):
            pbar["value"] = 0
            pct_lbl.config(text="0.0%")

    def _predict(self) -> None:
        if self.model is None:
            return

        # Preprocess PIL image for CNN
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img_resized, dtype=np.float32) / 255.0

        # Check if canvas is empty
        if np.max(arr) == 0:
            return

        norm_arr = (arr - MNIST_MEAN) / MNIST_STD
        tensor = torch.tensor(norm_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_digit = int(np.argmax(probs))
        confidence = probs[pred_digit] * 100.0

        self.pred_label.config(text=f"Prediction: Digit {pred_digit}", foreground="#008800" if confidence > 80 else "#cc6600")
        self.conf_label.config(text=f"Confidence: {confidence:.1f}%")

        for i, (prob, pbar, pct_lbl) in enumerate(zip(probs, self.prob_bars, self.prob_labels)):
            val = float(prob * 100.0)
            pbar["value"] = val
            pct_lbl.config(text=f"{val:4.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI Application for Handwritten Digit Recognition")
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pth", help="Path to trained model checkpoint")
    args = parser.parse_args()

    root = tk.Tk()
    app = DigitRecognizerGUI(root, args.model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
