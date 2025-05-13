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

def spatial_interpolate_with_nans(sensor_map, values, grid_h, grid_w, method='linear', alpha=0.2):
    """
    Spatial interpolation that properly handles NaN values in the data.
    
    Args:
        sensor_map: Dictionary mapping sensor indices to (x, y) positions
        values: Tensor of sensor values, may contain NaNs
        grid_h, grid_w: Height and width of output grid
        method: Interpolation method ('linear', 'cubic', 'nearest')
        alpha: Blending factor for mean (0 = only mean, 1 = only interpolated)
        
    Returns:
        Interpolated tensor of shape (grid_h, grid_w)
    """
    # Convert to numpy arrays
    points = np.array(list(sensor_map.values()))
    vals = values.numpy() if isinstance(values, torch.Tensor) else np.array(values)
    
    # Find valid (non-NaN) sensor readings
    valid_mask = ~np.isnan(vals)
    
    # If no valid values, return grid filled with zeros
    if not np.any(valid_mask):
        return torch.zeros((grid_h, grid_w), dtype=torch.float32)
    
    # Extract valid points and values
    valid_points = points[valid_mask]
    valid_vals = vals[valid_mask]
    
    # Calculate mean of valid values
    mean_val = np.mean(valid_vals)
    
    # Create grid for interpolation
    grid_x, grid_y = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    
    # Perform interpolation with only valid values
    interpolated = griddata(valid_points, valid_vals, grid, method=method, fill_value=mean_val)
    
    # Blend with mean value
    blended = (1 - alpha) * mean_val + alpha * interpolated
    
    return torch.tensor(blended.reshape(grid_h, grid_w)).float()
