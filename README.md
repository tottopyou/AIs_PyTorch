# AI Solutions with PyTorch

Welcome to the AI Solutions repository based on PyTorch!

## Overview

This repository contains various AI solutions and projects implemented using the PyTorch framework. You'll find a range of projects here, such as linear regression, neural network classification, and more.

## Getting Started

To get started with PyTorch, you'll need to have Python installed on your system. You can install PyTorch and its dependencies via pip:

```bash
pip install torch torchvision
```

## Table of Contents

- [Linear Regression](#linear-regression)
- [Multi Classification](#multi-classification)
- [Computer Vision](#computer-vision)

---

# Linear Regression

This project demonstrates the implementation of a simple linear regression model using the PyTorch framework.

## Overview

Linear regression is a fundamental machine learning technique used for predicting the relationship between two variables. In this project, we aim to predict a target variable (Y) based on a single input feature (X). We'll train a linear regression model to learn the best-fitting line that represents the relationship between X and Y.

## Data Generation

We generate synthetic data for training and testing purposes. The input feature (X) is generated using a range of values from 0 to 1 with a step size of 0.02. The target variable (Y) is computed using the equation: Y = 0.7 * X + 0.3, with added noise.

## Model Architecture

We use a simple linear regression model implemented in PyTorch. The model consists of a single linear layer.

```python
class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
```
## Here is how it looks in the graph 

![screen](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/f97091b6-9d25-4db7-96e0-e52dcab4bdda)

## And how it evolve during epochs

![linear_regression](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/e82bafca-e12f-4603-aaa0-6dc977bf0a4a)

---

# Multi Classification

This project demonstrates the implementation of a neural network for multi-class classification using PyTorch.

## Overview

Multi-class classification is a task where we aim to classify data points into multiple classes based on their features. In this project, we generate synthetic data blobs with multiple classes and train a neural network to classify them.

## Data Generation

We generate synthetic blob data using the `make_blobs` function from scikit-learn. The data consists of 1000 samples with 2 features and 4 clusters (classes). Each sample is assigned a label indicating its cluster.

## Model Architecture

We define a neural network model for multi-class classification. The model consists of three fully connected layers with ReLU activation functions.

```python
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=20):
        super().__init__()
        self.linear_later_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self,x):
        return self.linear_later_stack(x)

model_4 = BlobModel(input_features=2, output_features=4)
```

## Here is how it looks in the graph 

![image](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/02a19418-b0ea-47ef-9218-3f1ba1956183)

## And how it evolve during epochs

![Untitled-2](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/ede51347-31b8-46b4-ac2e-8ca3cc290ea1)

---

# Computer Vision


