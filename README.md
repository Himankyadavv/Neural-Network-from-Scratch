# Neural Networks From Scratch

A from-scratch implementation of a neural network's core machinery — forward propagation, backpropagation, optimizers, and regularization — built using only **NumPy**, following *Neural Networks from Scratch* (NNFS) by Harrison Kinsley & Daniel Kukieła.

The goal is to understand the mathematics behind deep learning by implementing it directly, rather than relying on high-level frameworks like PyTorch or TensorFlow.

## Overview

This project implements every core component of a neural network by hand:

- Forward propagation
- Backpropagation
- Gradient descent
- Activation functions
- Loss functions
- Optimizers
- Regularization
- Training and evaluation on a synthetic dataset

## Implementation

- **Dense (fully connected) layers**
- **Activations:** ReLU, Softmax
- **Loss:** Categorical Cross-Entropy, with a combined Softmax + Cross-Entropy backward pass for numerical stability
- **Regularization:** L1 and L2
- **Optimizers:** SGD, SGD with Momentum, AdaGrad, RMSProp, Adam
- **Learning rate decay**
- **Dataset:** spiral dataset classification (NNFS dataset utilities)

## Project Structure

The entire implementation lives in a single Jupyter notebook:

```
neural-networks-from-scratch/
└── neural_networks_from_scratch.ipynb
```

## Getting Started

### Requirements
```
python >= 3.8
numpy
matplotlib
nnfs
```

### Installation
```bash
git clone https://github.com/<your-username>/neural-networks-from-scratch.git
cd neural-networks-from-scratch
pip install numpy matplotlib nnfs
```

### Usage
Open the notebook and run all cells:
```bash
jupyter notebook neural_networks_from_scratch.ipynb
```

## Results

<!-- Add a plot of the decision boundary on the spiral dataset, and final train/validation accuracy and loss, once available. -->

## Learning Goals

This project was built to develop a solid understanding of:

- Linear algebra for deep learning
- Calculus behind backpropagation
- Gradient flow and matrix derivatives
- Weight initialization
- Optimization algorithms
- Regularization techniques
- The end-to-end neural network training pipeline

## References

- Kinsley, H. & Kukieła, D. — *Neural Networks from Scratch*
- Stanford CS231n — Convolutional Neural Networks for Visual Recognition
- Stanford CS229 — Machine Learning

## Author

**Himank Yadav**
Computer Science Engineering Student — exploring Deep Learning, Machine Learning, and AI Research.

## License

This project is for educational purposes. Thank You