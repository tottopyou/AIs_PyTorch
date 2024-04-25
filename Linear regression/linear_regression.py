import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange( start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

splited = int(0.8 * len(X))
train_X, train_Y = X[:splited],Y[:splited]
test_X, test_Y = X[splited:], Y[splited:]

predictions = None

plt.figure(figsize=(10, 7))

plt.scatter(train_X, train_Y, c="b", s=4, label="Training data")

plt.scatter(test_X, test_Y, c="g", s=4, label="Testing data")

if predictions is not None:

    plt.scatter(test_X, predictions, c="r", s=4, label="Predictions")

plt.legend(prop={"size": 14})
plt.show()


class LinearRegressionV2 (nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_layer = nn.Linear(in_features=1,out_features=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionV2()
model_1, model_1.state_dict()

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_1.parameters(),
                            lr = 0.01)

torch.manual_seed(42)

epochs = 116

for epoch in range(epochs):
    model_1.train()

    y_pred = model_1(train_X)

    loss = loss_fn(y_pred, train_Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(test_X)

        test_loss = loss_fn(test_pred, test_Y)

        predictions = test_pred

        plt.clf()

        plt.scatter(train_X, train_Y, c="b", s=4, label="Training data")

        plt.scatter(test_X, test_Y, c="g", s=4, label="Testing data")

        if predictions is not None:

            plt.scatter(test_X, test_pred.detach(), c="r", s=4, label=f"Predictions (Epoch {epoch})")

            if epoch == 0 :
                plt.pause(10)

        plt.legend(prop={"size": 14})
        plt.pause(0.1)

plt.show()

print(model_1.state_dict())

