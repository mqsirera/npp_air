import torch
import torch.nn as nn
import numpy as np

class RBFKernel(nn.Module):
    def __init__(self, length_scale=1.0, learnable=False):
        super().__init__()
        if learnable:
            self.log_length_scale = nn.Parameter(torch.tensor(np.log(length_scale), dtype=torch.float32))
        else:
            self.register_buffer("log_length_scale", torch.tensor(np.log(length_scale), dtype=torch.float32))

    def forward(self, X, X_prime, kernel_param=None):
        log_length_scale = kernel_param if kernel_param is not None else self.log_length_scale
        length_scale = torch.exp(log_length_scale)
        dist_sq = torch.cdist(X, X_prime, p=2)**2
        return torch.exp(-dist_sq / (2.0 * length_scale**2))


class NPPLoss(nn.Module):
    def __init__(self, kernel: RBFKernel, noise=1e-4):
        super().__init__()
        self.kernel = kernel
        self.noise = noise

    def forward(self, preds, pins, targets, kernel_params=None):
        total_loss = 0
        for i, (pred, p, y) in enumerate(zip(preds, pins, targets)):
            y_hat = pred.squeeze()[p[:, 0], p[:, 1]]
            diff = y - y_hat
            K = self.kernel(p.float(), p.float(), kernel_params[i] if kernel_params is not None else None)
            K = K + self.noise * torch.eye(len(p), device=p.device)
            K_inv = torch.linalg.pinv(K)
            loss = diff @ K_inv @ diff + torch.logdet(K)
            total_loss += loss
        return total_loss / len(preds)


class NPPModel(nn.Module):
    def __init__(self, autoencoder, kernel_mode="fixed", noise=1e-4):
        super().__init__()
        self.model = autoencoder
        self.kernel_mode = kernel_mode
        self.kernel = RBFKernel(length_scale=1.0, learnable=(kernel_mode == "learned"))
        self.loss_fn = NPPLoss(kernel=self.kernel, noise=noise)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, preds, pins, targets):
        recon, kernel_params = preds
        return self.loss_fn(recon, pins, targets, kernel_params if self.kernel_mode == "predicted" else None)