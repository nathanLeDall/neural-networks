# Neural Networks From Scratch in Rust

A configurable **feed-forward neural network built from scratch in Rust**, without TensorFlow, PyTorch, or other machine-learning frameworks.

This project implements the core mathematics behind neural networks directly, including custom matrix operations, forward propagation, backpropagation, Xavier weight initialization, activation functions, and gradient-based training.

The included example trains a multilayer network to solve the **XOR problem**.

## Features

* Neural network implemented from scratch in Rust
* Configurable multilayer architecture
* Custom matrix implementation
* Feed-forward propagation
* Backpropagation
* Xavier weight initialization
* Trainable weights and biases
* Sigmoid activation function
* Dynamic learning-rate adjustment
* Random weight and bias initialization
* Interactive prediction after training
* No external machine-learning framework

## Technologies

* **Rust**
* **Cargo**
* **rand**
* **serde**
* **serde_json**

All neural-network and matrix calculations are implemented directly in Rust.

## Network Architecture

The network architecture can be configured by passing a vector containing the number of neurons in each layer.

The included XOR example uses:

```rust
Network::new(vec![2, 4, 3, 1], 0.05, SIGMOID);
```

Which creates the following network:

```text
2 Input Neurons
       │
       ▼
4 Hidden Neurons
       │
       ▼
3 Hidden Neurons
       │
       ▼
1 Output Neuron
```

Different architectures can be created simply by changing the layer configuration.

For example:

```rust
Network::new(vec![3, 8, 4, 2], 0.05, SIGMOID);
```

## How It Works

Training consists of three main stages:

### 1. Feed Forward

Input values are passed through each layer of the network.

For each layer:

```text
Output = Activation(Weights × Input + Bias)
```

The output of one layer becomes the input to the next layer.

### 2. Calculate Error

The network compares its prediction against the expected target value:

```text
Error = Target - Prediction
```

This error is then propagated backward through the network.

### 3. Backpropagation

The network calculates gradients using the derivative of the activation function and adjusts the weights and biases.

Conceptually:

```text
Gradient = Activation Derivative × Error × Learning Rate
```

The error is then propagated backward through each previous layer.

## Xavier Initialization

Weights are initialized using **Xavier initialization** rather than using unrestricted random values.

The initialization range is based on the number of neurons entering and leaving each layer.

This helps keep initial network values within a more reasonable range during training.

## Sigmoid Activation

The current implementation uses the sigmoid activation function:

```text
             1
σ(x) = ─────────────
        1 + e^(-x)
```

Its derivative is implemented as:

```text
σ'(x) = x(1 - x)
```

The activation logic is separated into its own module so the network architecture is not directly tied to the sigmoid implementation.

## Custom Matrix Operations

The neural network uses a custom `Matrix` implementation instead of a linear-algebra or machine-learning library.

Supported operations include:

* Matrix creation
* Zero matrices
* Random matrices
* Xavier-initialized matrices
* Matrix multiplication
* Element-wise multiplication
* Matrix addition
* Matrix subtraction
* Transposition
* Function mapping

These operations are used directly during forward propagation and backpropagation.

## XOR Example

The included example trains the network to learn the XOR truth table:

| Input 1 | Input 2 | Expected Output |
| ------: | ------: | --------------: |
|       0 |       0 |               0 |
|       0 |       1 |               1 |
|       1 |       0 |               1 |
|       1 |       1 |               0 |

The training data is defined as:

```rust
let inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];

let targets = vec![
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![0.0],
];
```

The network is then created and trained:

```rust
let mut network =
    Network::new(vec![2, 4, 3, 1], 0.05, SIGMOID);

network.train(
    inputs,
    targets,
    100000,
    0.01
);
```

This trains the network for **100,000 epochs**, gradually adjusting the learning rate from `0.05` toward `0.01`.

## Project Structure

```text
neural-networks/
│
├── src/
│   ├── main.rs
│   ├── network.rs
│   ├── matrix.rs
│   └── activations.rs
│
├── Cargo.toml
├── Cargo.lock
├── some_research.txt
└── README.md
```

### `main.rs`

Contains the XOR training example, creates the neural network, trains it, displays predictions, and accepts custom input values.

### `network.rs`

Contains the core neural-network implementation, including:

* Network initialization
* Feed-forward propagation
* Backpropagation
* Training
* Weight updates
* Bias updates
* Learning-rate adjustment

### `matrix.rs`

Contains the custom matrix implementation used by the neural network.

### `activations.rs`

Defines activation functions and their derivatives.

The current implementation includes sigmoid activation.

## Installation

Make sure you have Rust and Cargo installed.

Clone the repository:

```bash
git clone https://github.com/nathanLeDall/neural-networks.git
cd neural-networks
```

Build the project:

```bash
cargo build
```

## Running

Run the neural network with:

```bash
cargo run
```

The program will:

1. Initialize the neural network
2. Train it on the XOR dataset
3. Print predictions for all four XOR inputs
4. Accept two custom input values
5. Run those values through the trained network

## Example Usage

After training, predictions are generated using:

```rust
network.feed_forward(vec![0.0, 0.0]);
network.feed_forward(vec![0.0, 1.0]);
network.feed_forward(vec![1.0, 0.0]);
network.feed_forward(vec![1.0, 1.0]);
```

You can also provide your own two input values through the terminal.

## Concepts Demonstrated

This project demonstrates several foundational machine-learning and computer-science concepts:

* Artificial neural networks
* Feed-forward neural networks
* Backpropagation
* Gradient-based learning
* Activation functions
* Xavier initialization
* Matrix multiplication
* Linear algebra
* Weight and bias optimization
* Learning-rate scheduling
* Multilayer network architectures
* Rust ownership and data structures

## Purpose

The goal of this project was to understand neural networks at a lower level by implementing the underlying algorithms directly rather than relying on a machine-learning framework.

Building the matrix operations, forward propagation, weight initialization, and backpropagation manually provides a clearer understanding of what frameworks such as TensorFlow and PyTorch normally handle automatically.

## Author

**Nathan Le Dall**

GitHub: [@nathanLeDall](https://github.com/nathanLeDall)
