# Neural Networks From Scratch

A comprehensive, from-scratch implementation of neural network core components using only **NumPy**. This project demonstrates the mathematical foundations of deep learning through direct implementation, following *Neural Networks from Scratch*.

> **Purpose:** Understand the mathematics behind deep learning by implementing it directly, rather than relying on high-level frameworks like PyTorch or TensorFlow.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Learning Outcomes](#learning-outcomes)
- [References](#references)
- [Author](#author)

---

## ✨ Features

### Core Components Implemented

| Component | Details |
|-----------|---------|
| **Layers** | Dense (fully connected) layers |
| **Activation Functions** | ReLU, Softmax |
| **Loss Functions** | Categorical Cross-Entropy (with Softmax-CrossEntropy backward pass for numerical stability) |
| **Regularization** | L1 and L2 regularization |
| **Optimizers** | SGD, SGD with Momentum, AdaGrad, RMSProp, Adam |
| **Learning Rate** | Decay scheduling |
| **Training** | Spiral dataset classification using NNFS utilities |

### Key Implementations

- ✅ Forward propagation
- ✅ Backpropagation algorithm
- ✅ Gradient descent optimization
- ✅ Weight initialization strategies
- ✅ Training and evaluation pipeline
- ✅ Decision boundary visualization

---

## 📁 Project Structure

```
Neural-Network-from-Scratch/
├── README.md                                    # This file
└── neural_networks_from_scratch.ipynb          # Complete implementation notebook
```

All implementations are contained in a single, well-documented Jupyter notebook for ease of understanding and experimentation.

---

## 🚀 Getting Started

### Prerequisites

- **Python** ≥ 3.8
- **NumPy** — Numerical computations
- **Matplotlib** — Visualization
- **NNFS** — Dataset utilities

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Himankyadavv/Neural-Network-from-Scratch.git
   cd Neural-Network-from-Scratch
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib nnfs
   ```

### Usage

1. **Open the notebook:**
   ```bash
   jupyter notebook neural_networks_from_scratch.ipynb
   ```

2. **Run all cells** to execute the complete pipeline, from network initialization to training and evaluation.

---

## 🏗️ Architecture

### Network Design

The implementation follows a modular architecture:

- **Layers**: Dense layers with configurable input/output dimensions
- **Activation Functions**: Applied after linear transformations
- **Loss Calculation**: Combines predictions with target labels
- **Optimization**: Iterative weight updates using various optimizer algorithms
- **Regularization**: Prevents overfitting through L1/L2 penalties

### Forward Pass

Inputs → Dense Layer → Activation → Dense Layer → Softmax → Loss

### Backward Pass

Gradient computation flows backward through each layer, updating weights via the chosen optimizer.

---

## 📚 Learning Outcomes

This project develops understanding of:

- **Linear Algebra** — Matrix operations, transposes, and dimensions
- **Calculus** — Derivatives, chain rule, and gradient computation
- **Backpropagation** — Gradient flow through network layers
- **Weight Initialization** — Xavier/He initialization strategies
- **Optimization Algorithms** — SGD, Momentum, Adaptive methods (Adam, RMSProp)
- **Regularization** — L1/L2 penalties to prevent overfitting
- **Training Pipeline** — End-to-end neural network workflow
- **Numerical Stability** — Combined Softmax-CrossEntropy backward pass

---

## 📊 Results

<!-- Results section with decision boundary plots and accuracy metrics will be added upon completion. -->

---

## 📖 References

- **Kinsley, H. & Kukieła, D.** — *Neural Networks from Scratch* — In-depth guide to implementing neural networks from fundamentals
- **Stanford CS231n** — Convolutional Neural Networks for Visual Recognition
- **Stanford CS229** — Machine Learning

---

## 👤 Author

**Himank Yadav**

Computer Science Engineering Student exploring Deep Learning, Machine Learning, and AI Research.

---

## 📝 License

This project is for educational purposes.

---

## 💡 Tips for Understanding the Code

1. **Start with the forward pass** — Understand how data flows through the network
2. **Study the loss function** — See how predictions are evaluated
3. **Trace backpropagation** — Follow gradients backward through layers
4. **Experiment with hyperparameters** — Modify learning rates, optimizers, and regularization
5. **Visualize decision boundaries** — See how the network learns on the spiral dataset

---

**Happy Learning! 🎓**
