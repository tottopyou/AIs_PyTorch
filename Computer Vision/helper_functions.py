import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:

            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(),
                                      dim=0) 
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)