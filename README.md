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
- [Neural Network Classification](#neural-network-classification)

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

## Neural Network Classification

Description of your neural network classification project goes here.
