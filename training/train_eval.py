import torch
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader=None, test_loader=None, name="Model", device='cuda', lr=1e-4, epochs=20):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    train_losses = []
    val_mse_scores = []

    best_model_state = None
    best_val_mse = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress:
            x = batch['image'].to(device)
            pins = [p.to(device) for p in batch['pins']]
            targets = [t.to(device) for t in batch['outputs']]

            output = model(x)
            loss = model.compute_loss(output, pins, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[{name}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            y_true, y_pred = [], []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch['image'].to(device)
                    pins = [p.to(device) for p in batch['pins']]
                    targets = [t.to(device) for t in batch['outputs']]

                    output = model(x)
                    if isinstance(output, tuple):
                        output = output[0]

                    for pred, p, y in zip(output, pins, targets):
                        y_hat = pred.squeeze()[p[:, 0], p[:, 1]]
                        y_true.append(y.cpu().numpy())
                        y_pred.append(y_hat.cpu().numpy())

            y_true_all = np.concatenate(y_true)
            y_pred_all = np.concatenate(y_pred)
            val_mse = mean_squared_error(y_true_all, y_pred_all)
            val_r2 = r2_score(y_true_all, y_pred_all)
            val_mse_scores.append(val_mse)
            print(f"[{name}]  └── Val MSE: {val_mse:.4f} | R²: {val_r2:.4f}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_state = copy.deepcopy(model.state_dict())

        model.train()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[{name}] Restored best model (Val MSE: {best_val_mse:.4f})")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{name} Training Loss")
    plt.grid(True); plt.legend()

    if val_loader is not None:
        plt.subplot(1, 2, 2)
        plt.plot(val_mse_scores, label="Val MSE")
        plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title(f"{name} Validation MSE")
        plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()

    if test_loader:
        evaluate_model(model, test_loader, name, device)

    return model

def evaluate_model(model, test_loader, name="Model", device='cuda'):
    model.to(device)
    model.eval()
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['image'].to(device)
            pins = [p.to(device) for p in batch['pins']]
            targets = [t.to(device) for t in batch['outputs']]

            output = model(x)
            if isinstance(output, tuple):
                output = output[0]

            for pred, p, y in zip(output, pins, targets):
                y_hat = pred.squeeze()[p[:, 0], p[:, 1]]
                all_y_true.append(y.cpu().numpy())
                all_y_pred.append(y_hat.cpu().numpy())

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    mse = mean_squared_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)

    print(f"[{name}] Test MSE: {mse:.4f} | R²: {r2:.4f}")
    return mse, r2