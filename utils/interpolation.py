import numpy as np
import torch
from scipy.interpolate import griddata

def spatial_interpolate(sensor_map, values, grid_h, grid_w, method='linear', alpha=0.2):
    points = np.array(list(sensor_map.values()))
    vals = np.array(values)
    mean_val = np.mean(vals)

    grid_x, grid_y = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    interpolated = griddata(points, vals, grid, method=method, fill_value=mean_val)
    blended = (1 - alpha) * mean_val + alpha * interpolated

    return torch.tensor(blended.reshape(grid_h, grid_w)).float()