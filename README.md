# Handwritten Digit Recognition Using CNN

A Deep Convolutional Neural Network (CNN) implementation in PyTorch for classifying handwritten digits (0–9) trained on the benchmark MNIST dataset.

## System Architecture

The deep learning pipeline consists of a 4-layer Convolutional Neural Network with Batch Normalization, ReLU activation functions, Max Pooling, and Spatial Dropout for regularization.

```
Input (1 x 28 x 28)
  │
  ├── [Conv2D 3x3, 32] ── BatchNorm ── ReLU
  ├── [Conv2D 3x3, 32] ── BatchNorm ── ReLU
  ├── [MaxPool 2x2] ── Dropout (0.15)
  │     Output: (32 x 14 x 14)
  │
  ├── [Conv2D 3x3, 64] ── BatchNorm ── ReLU
  ├── [Conv2D 3x3, 64] ── BatchNorm ── ReLU
  ├── [MaxPool 2x2] ── Dropout (0.25)
  │     Output: (64 x 7 x 7)
  │
  ├── Flatten -> 3,136 features
  ├── Fully Connected (3136 -> 128) ── BatchNorm ── ReLU ── Dropout (0.30)
  └── Fully Connected (128 -> 10) -> Logits
```

## Directory Structure

```
.
├── src/
│   ├── __init__.py        # Export core modules
│   ├── model.py           # PyTorch CNN model definition (DigitCNN)
│   ├── dataset.py         # Data preprocessing & DataLoader factory
│   └── utils.py           # Seed control, checkpointing & plot utilities
├── train.py               # Model training script with LR scheduler
├── evaluate.py            # Model evaluation, confusion matrix & metrics
├── predict.py             # Inference pipeline for custom digit image files
├── gui_app.py             # Interactive Tkinter drawing canvas GUI
├── main.py                # Command-line interface orchestrator
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Quick Start

### 1. Installation

Clone or download this repository, then install requirements:

```bash
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

### 2. Train Model

Run training on the MNIST dataset (downloads dataset automatically):

```bash
python train.py --epochs 8 --batch-size 64 --lr 0.001
```

Training saves the optimal checkpoint to `artifacts/best_model.pth` and exports training history loss/accuracy curves to `artifacts/training_history.png`.

### 3. Evaluate Performance

Compute accuracy, precision, recall, F1-scores, confusion matrix, and misclassified digit grid:

```bash
python evaluate.py --model-path artifacts/best_model.pth
```

Outputs:
- `artifacts/confusion_matrix.png`
- `artifacts/misclassified_samples.png`

### 4. Single Image Prediction

Classify a single custom handwritten image file:

```bash
python predict.py --image path/to/digit.png
```

### 5. Interactive Drawing Application

Launch GUI canvas to draw digits with your mouse in real time:

```bash
python gui_app.py
```

## Benchmarks & Results

| Metric | Benchmark Score |
| :--- | :--- |
| **Test Accuracy** | ~99.2% |
| **Test Loss** | ~0.024 |
| **Params** | ~430,000 |
| **Epochs to Converge** | 6–8 Epochs |

## Technologies Used

- **Framework**: PyTorch 2.x, Torchvision
- **Data & Math**: NumPy, SciPy, Scikit-Learn
- **Visualization**: Matplotlib, Pillow
- **GUI**: Python Tkinter
