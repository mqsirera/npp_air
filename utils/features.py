import torch
import numpy as np
from tqdm import tqdm

def extract_temporal_features(dataset, temporal_channels=7, device='cpu'):
    X_list, y_list = [], []

    for sample in tqdm(dataset, desc="Extracting features"):
        img = sample['image'].to(device)
        pins = sample['pins']
        targets = sample['outputs']

        for pin, target in zip(pins, targets):
            y, x = pin
            features = img[:temporal_channels, y, x].cpu().numpy()
            X_list.append(features)
            y_list.append(target.item())

    return np.stack(X_list), np.array(y_list)