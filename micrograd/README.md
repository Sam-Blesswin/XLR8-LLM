# Micrograd: Manual Backpropagation and Tiny Neural Network Framework from Scratch

This notebook is an educational project that walks through **manual backpropagation**, the construction of a minimal **autograd engine**, and eventually a working **multi-layer perceptron (MLP)** trained via gradient descent. It mirrors the philosophy of tinygrad/micrograd projects but with clearer step-by-step annotations.

It is structured as a Jupyter Notebook that demonstrates:
1. Manual gradient computation using the chain rule.
2. Building a minimal autograd engine (`Value` class).
3. Training a multi-layer perceptron (MLP) from scratch using backpropagation.
4. Visualizing computation graphs.
5. Comparing results with PyTorch for validation.

---

## 📘 Overview

### Manual Backpropagation
The project begins with concrete examples of backpropagation by hand. You’ll compute gradients for a scalar function:
```
L = ((a * b) + c) * f
```
This is the basis for understanding how backpropagation works via the chain rule.

### Autograd Engine
The core of this project is a class called `Value`, which mimics PyTorch’s autograd functionality:

- Overloads operators (`+`, `*`, `**`, `tanh`, etc.)
- Tracks the computation graph dynamically.
- Implements `.backward()` to compute gradients with toposort and reverse-mode autodiff.
- Each node in the graph stores:
  - `data` (forward pass value)
  - `grad` (gradient accumulated during backward pass)
  - `_op`, `_prev` (graph tracking)

### Visualizing the Computation Graph
Using Graphviz, you can inspect how the operations form a computation graph:
```python
draw_dot(L)
```
This helps in debugging and visualizing how data and gradients flow through the graph.

---

## 🧠 Neural Network Construction

You build a neural network step-by-step:

- **Neuron** class: Represents a single neuron with weights and bias, and tanh activation.
- **Layer** class: Stack of neurons to form a layer.
- **MLP (Multi-layer Perceptron)**: A full network with multiple layers, fully connected.

```python
n = MLP(3, [4, 4, 1])
```
This creates a network with 3 input features, two hidden layers of 4 neurons each, and 1 output neuron.

---

## 🏋️ Training

Training involves:

- A forward pass using current weights.
- Calculating the loss (mean squared error).
- Resetting gradients.
- Calling `.backward()` on the loss.
- Updating weights using gradient descent.

---

## 📝 Credits

This work is inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.