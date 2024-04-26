import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

num_classes = 4
num_features = 2
random_seed = 42

x_blob, y_blob = make_blobs(n_samples=1000, n_features=num_features, centers=num_classes, cluster_std=1.5,
                            random_state=random_seed)

X_blob = torch.from_numpy(x_blob).type(torch.float32)
Y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, Y_blob_train, Y_blob_test = train_test_split(X_blob,Y_blob,test_size=0.2, random_state=random_seed)

plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0],X_blob[:,1], c=Y_blob, cmap=plt.cm.RdYlBu)
plt.show()

class BlobModel (nn.Module):
    def __init__(self, input_features, output_features,hidden_units=20):
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

model_4 = BlobModel(input_features=2,output_features=4)

print(model_4)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.05)

torch.manual_seed(random_seed )

epoches = 100

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

for epoch in range(epoches):

    model_4.train()

    Y_logits = model_4(X_blob_train)
    Y_pred = torch.softmax(Y_logits, dim=1).argmax(dim=1) 
    
    loss = loss_fn(Y_logits,Y_blob_train)
    acc = accuracy_fn(y_true= Y_blob_train,
                      y_pred=Y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1) 

        test_loss = loss_fn(test_logits, Y_blob_test)
        test_acc = accuracy_fn(y_true= Y_blob_test,
                        y_pred=test_pred)
        
        ax[0].clear()
        ax[1].clear()
        ax[0].set_title(f"Train | Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%")
        plot_decision_boundary(model_4, X_blob_train, Y_blob_train, epoch)
        ax[1].set_title(f"Test | Epoch: {epoch} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
        plot_decision_boundary(model_4, X_blob_test, Y_blob_test, epoch)
        plt.pause(0.1)
        if epoch == 0:
            plt.pause(15)

plt.show()