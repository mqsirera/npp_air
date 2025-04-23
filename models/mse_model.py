import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEModel(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.model = autoencoder
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        recon, _ = self.model(x)
        return recon

    def compute_loss(self, preds, pins, targets):
        total_loss = 0
        for pred, p, t in zip(preds, pins, targets):
            pred_vals = pred.squeeze()[p[:, 0], p[:, 1]]
            total_loss += self.loss_fn(pred_vals, t)
        return total_loss / len(preds)
