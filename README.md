# Neural Network from Scratch

A custom implementation of neural network components built from scratch using NumPy. This project demonstrates the fundamental concepts of deep learning by implementing layers, activation functions, loss functions, optimizers, and a complete training pipeline.

## Project Overview

This repository contains a complete neural network framework (`nnfs_custom`) that allows you to build and train neural networks without relying on high-level deep learning libraries. The implementation emphasizes educational value while maintaining practical usability.

## Project Structure

```
Neural-Network-from-Scratch/
├── nnfs_custom/                    # Custom neural network framework
│   ├── __init__.py                 # Module exports
│   ├── layers.py                   # Dense layers and dropout
│   ├── activations.py              # Activation functions
│   ├── losses.py                   # Loss functions
│   ├── optimizers.py               # Optimization algorithms
│   ├── metrics.py                  # Accuracy metrics
│   └── model.py                    # Model class and training logic
├── train_fashion_mnist.py          # Image classification example
├── train_spiral.py                 # Binary classification example
├── train_sine.py                   # Regression example
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Features

### Core Components

#### **Layers** (`nnfs_custom/layers.py`)
- **Dense Layer** (`layer_dense`): Fully connected layer with support for L1/L2 regularization
- **Dropout Layer** (`Layer_Dropout`): Regularization technique to prevent overfitting

#### **Activation Functions** (`nnfs_custom/activations.py`)
- **ReLU**: Rectified Linear Unit - commonly used in hidden layers
- **Softmax**: Multi-class classification activation
- **Sigmoid**: Binary classification activation
- **Linear**: For regression tasks

#### **Loss Functions** (`nnfs_custom/losses.py`)
- **Categorical Cross-Entropy**: Multi-class classification
- **Binary Cross-Entropy**: Binary classification
- **Mean Squared Error (MSE)**: Regression
- **Mean Absolute Error (MAE)**: Regression
- **Softmax + Categorical Cross-Entropy**: Combined loss for efficiency

#### **Optimizers** (`nnfs_custom/optimizers.py`)
- **SGD**: Stochastic Gradient Descent
- **AdaGrad**: Adaptive Gradient
- **RMSprop**: Root Mean Square Propagation
- **ADAM**: Adaptive Moment Estimation

#### **Metrics** (`nnfs_custom/metrics.py`)
- **Classification Accuracy**: For classification tasks
- **Regression Accuracy**: Mean absolute percentage error for regression

#### **Model** (`nnfs_custom/model.py`)
- **Model Class**: High-level API for building and training networks
- **Forward Propagation**: Compute network outputs
- **Backpropagation**: Compute gradients and update weights
- **Training Loop**: Full training with validation support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Himankyadavv/Neural-Network-from-Scratch.git
cd Neural-Network-from-Scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- `numpy`: Numerical computations
- `nnfs`: Neural Network from Scratch library (provides datasets)
- `matplotlib`: Visualization

## Usage Examples

### 1. Classification: Fashion MNIST

Train a neural network on the Fashion MNIST dataset (28×28 images of clothing items):

```bash
python train_fashion_mnist.py
```

**Network Architecture:**
- Input: 784 neurons (28×28 flattened images)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 128 neurons + ReLU
- Output: 10 neurons + Softmax (for 10 clothing classes)

**Features:**
- Automatic dataset download from TensorFlow/Keras datasets
- Training on 1000 samples with 100 validation samples
- ADAM optimizer with learning rate decay
- Categorical cross-entropy loss

### 2. Binary Classification: Spiral Dataset

Train on a spiral classification problem:

```bash
python train_spiral.py
```

**Network Architecture:**
- Input: 2 features
- Hidden Layer: 64 neurons + ReLU (with L2 regularization)
- Output: 1 neuron + Sigmoid

**Features:**
- Binary classification task
- L2 weight regularization to prevent overfitting
- Training and validation on spiral datasets (100 samples each)
- 1000 training epochs

### 3. Regression: Sine Wave

Train a network to learn the sine function:

```bash
python train_sine.py
```

**Network Architecture:**
- Input: 1 feature
- Hidden Layer: 64 neurons + ReLU
- Output: 1 neuron + Linear

**Features:**
- Regression task
- Mean Squared Error (MSE) loss
- 1000 training epochs with progress printing every 100 epochs

## How to Build a Model

```python
from nnfs_custom import *
import nnfs
from nnfs.datasets import sine_data

# Initialize NNFS
nnfs.init()

# Load or prepare data
X, y = sine_data()

# Create and build model
model = Model()
model.add(layer_dense(1, 64))           # Input to hidden
model.add(Activation_relu())             # Activation
model.add(layer_dense(64, 1))           # Hidden to output
model.add(Activation_linear())           # Output activation (for regression)

# Set loss, optimizer, and metrics
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_ADAM(),
    accuracy=Accuracy_Regression()
)

# Finalize model (connect layers)
model.finalize()

# Train the model
model.train(X, y, epochs=1000, print_every=100)
```

## Implementation Highlights

### Forward Propagation
Each layer implements a `forward()` method that computes the output given inputs.

### Backpropagation
Each layer implements a `backprop()` method that computes gradients with respect to its parameters and inputs.

### Regularization
- **L1 Regularization**: Encourages sparse weights
- **L2 Regularization**: Encourages small weights (weight decay)
- **Dropout**: Randomly deactivates neurons during training

### Training Pipeline
1. Forward pass through all layers
2. Calculate loss and metrics
3. Backpropagation to compute gradients
4. Optimizer updates weights and biases
5. Optional validation on separate data

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations and array operations |
| `nnfs` | Pre-built datasets (sine, spiral, MNIST) |
| `matplotlib` | Visualization (optional, used in some scripts) |

## Educational Value

This project is ideal for:
- Understanding how neural networks work from first principles
- Learning about different activation functions and their properties
- Exploring various loss functions and their applications
- Experimenting with different optimization algorithms
- Understanding forward and backward propagation
- Implementing regularization techniques

## Performance Notes

- The implementation prioritizes clarity and educational value over performance
- For production use, consider frameworks like TensorFlow, PyTorch, or Keras
- The current implementation trains on CPU only
- For large-scale datasets, consider using minibatch training

## Limitations

- No GPU acceleration
- Full batch training (no minibatch support in current version)
- Limited to CPU operations with NumPy
- No model saving/loading functionality

## Future Enhancements

Potential improvements could include:
- Minibatch training support
- Convolutional layers
- Recurrent layers (LSTM, GRU)
- Batch normalization
- Model serialization
- GPU support with CuPy or similar

## License

This project is open source and available for educational purposes. Thank You!

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Contributions that enhance clarity or add features are welcome!

## References

- Neural Networks and Deep Learning fundamentals
- NNFS (Neural Networks From Scratch) by Harrison Kinney
- Deep Learning by Goodfellow, Bengio, and Courville

---

**Happy Learning!** 🧠
