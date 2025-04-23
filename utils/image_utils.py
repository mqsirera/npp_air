from PIL import Image
import numpy as np
import torch

def load_base_images(image_paths, grid_h, grid_w):
    base_layers = []
    for path in image_paths:
        img = Image.open(path).convert('L')
        img = img.resize((grid_w, grid_h))
        img_arr = np.array(img) / 255.0
        base_layers.append(torch.tensor(img_arr).float())
    return torch.stack(base_layers, dim=0) if base_layers else torch.empty(0)