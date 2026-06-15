# Autograd-from-scratch

# Neural Network From Scratch in C++

A minimal implementation of an automatic differentiation engine and a Multi-Layer Perceptron (MLP) built entirely in modern C++.

This project is a C++ reimplementation of the core ideas behind neural network frameworks such as PyTorch, focusing on understanding:

* Computational graphs
* Automatic differentiation (Autograd)
* Backpropagation
* Feedforward neural networks
* Gradient-based optimization

## Features

### Autograd Engine

The `Value` class supports:

* Scalar values
* Gradient tracking
* Computational graph construction
* Reverse-mode automatic differentiation

Supported operations:

* Addition (`+`)
* Multiplication (`*`)
* Power (`pow`)
* ReLU activation
* Division
* Subtraction
* Negation

### Neural Network Components

Implemented from scratch:

* `Module`
* `Neuron`
* `Layer`
* `MLP (Multi-Layer Perceptron)`

Features:

* Fully connected layers
* ReLU activations
* Parameter collection
* Gradient resetting
* Forward propagation
* Backward propagation

## Project Structure

```text
.
├── engine.h
├── engine.cpp
├── nn.h
├── nn.cpp
└── main.cpp
```

### Files

| File         | Description                       |
| ------------ | --------------------------------- |
| `engine.h`   | Autograd engine declarations      |
| `engine.cpp` | Autograd engine implementation    |
| `nn.h`       | Neural network class declarations |
| `nn.cpp`     | Neural network implementation     |
| `main.cpp`   | Example usage and training loop   |

## Example

```cpp
#include "nn.h"

int main()
{
    MLP model(3, {4, 4, 1});

    std::vector<std::shared_ptr<Value>> x = {
        std::make_shared<Value>(2.0),
        std::make_shared<Value>(3.0),
        std::make_shared<Value>(-1.0)
    };

    auto output = model.forward(x);

    output[0]->backward();

    return 0;
}
```

## Build

### Using g++

```bash
g++ -std=c++17 engine.cpp nn.cpp main.cpp -o neural_net
```

Run:

```bash
./neural_net
```

## Learning Goals

This project is intended for:

* Students learning Deep Learning fundamentals
* Understanding how PyTorch autograd works internally
* Exploring computational graphs
* Practicing Modern C++
* Implementing neural networks without external ML libraries

## Future Improvements

* Tanh activation
* Sigmoid activation
* Matrix/Tensor support
* Optimizers (SGD, Adam)
* Loss functions
* Batch training
* Serialization
* GPU acceleration

## References

* Karpathy's Micrograd
* Neural Networks: Zero to Hero
* Deep Learning by Ian Goodfellow
