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
- [Creating Human Face](#creating-human-face)
- [Text To Image](#text-to-image)

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

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for classifying FashionMNIST images using PyTorch.

## Overview

Computer Vision is used to classify FashionMNIST images into 10 categories. FashionMNIST is a dataset of Zalando's article images—consisting of 60,000 training examples and 10,000 test examples—intended for benchmarking machine learning models.

## Data Preparation

We use the FashionMNIST dataset provided by torchvision, which consists of grayscale images of clothing items. The dataset is split into a training set and a test set.

## Model Architecture

We define a CNN model for image classification. The model consists of two convolutional blocks followed by a fully connected classifier.

```python
class ComputerVision(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

model = ComputerVision(
    input_shape=1,
    hidden_units=20,
    output_shape=len(class_names))
```
## Training

We train the CNN model using the training data. We use the Cross Entropy Loss function and Stochastic Gradient Descent (SGD) optimizer.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

epochs = 10
for epoch in tqdm(range(epochs)):
    train_step(data_loader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)
    test_step(data_loader=test_dataloader, model=model, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
```

## Prediction of AI after training and testing

![img2](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/be3eaf05-f8c0-40b8-a140-49e268475949)

## AI prediction table for each type

![img](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/a2e6a138-f409-4c30-9faa-f2dcf9c30ab2)

---

# Creating Human Face

This project implements a Generative Adversarial Network (GAN) to generate realistic human faces using the CelebA dataset.

## Overview

The Generative Adversarial Network (GAN) is a deep learning architecture consisting of two neural networks, a Generator and a Discriminator, that compete against each other in a game-theoretic framework. The Generator learns to produce realistic images, while the Discriminator learns to distinguish between real and generated images.

## Data Preparation

We use the CelebA dataset, which contains over 200,000 celebrity images with annotations. We preprocess the images to resize them to 64x64 pixels and normalize the pixel values.

```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

celeba_dataset = CelebA(root='path_to_celeba_dataset',
                        split='train',
                        transform=transform,
                        download=True)
```

## Model Architecture

### Generator

The Generator takes random noise as input and generates fake images that resemble the real images from the dataset.

```python
class Generator(nn.Module):
    def __init__(self, input_shape, hidden_layer_gen, layers):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_shape, hidden_layer_gen*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_layer_gen*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen*8, hidden_layer_gen*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_gen*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen*4, hidden_layer_gen*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_gen*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen*2, hidden_layer_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_gen),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_layer_gen, layers, 4, 2, 1, bias=False),
            nn.Tanh()
        )
```
### Discriminator

The Discriminator takes an image as input and predicts whether it is real (from the dataset) or fake (generated by the Generator).

```python
class Discriminator(nn.Module):
    def __init__(self, layers, hidden_layer_dis):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(layers, hidden_layer_dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis, hidden_layer_dis*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_dis*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis*2, hidden_layer_dis*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_dis*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis*4, hidden_layer_dis*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_layer_dis*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_layer_dis*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
```

## Results

During training, both real and fake images are periodically saved to visualize the progress of the GAN.

| Epoch 1 | Epoch 5 | Epoch 10 |
| ------- | ------- | -------- |
| ![fake_samples_epoch_000](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/1d4ba4a8-dec9-4b9b-845d-6244f568de59) | ![fake_samples_epoch_004](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/87f81d08-9b2b-42c4-a103-13317fbd3b60) | ![fake_samples_epoch_009](https://github.com/tottopyou/AIs_PyTorch/assets/110258834/c4afd0b7-adfc-450e-ac9b-852b11a00254) |

---
# Text To Image


