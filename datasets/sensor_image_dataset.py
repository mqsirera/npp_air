import torch
from torch.utils.data import Dataset
import numpy as np
from utils.interpolation import spatial_interpolate_with_nans

class MultichannelSensorImageDataset(Dataset):
    def __init__(self, pm25, temps, rh, elevation_arr, pixel_map, static_tensor,
                 lookback=5, normalize=False, stats=None):
        self.pm25 = pm25
        self.temps = temps
        self.rh = rh
        self.elevation = elevation_arr
        self.pixel_map = pixel_map
        self.static_tensor = static_tensor
        self.lookback = lookback
        self.normalize = normalize
        self.stats = stats
        self.valid_range = range(lookback, len(pm25))
        self.grid_height = static_tensor.shape[1] if static_tensor.numel() > 0 else 64
        self.grid_width = static_tensor.shape[2] if static_tensor.numel() > 0 else 64
    
    def __len__(self):
        return len(self.valid_range)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_range[idx]
        
        # Get current PM2.5 values and identify valid (non-NaN) sensors
        pm_values = self.pm25.iloc[actual_idx].values
        valid_mask = ~np.isnan(pm_values)
        
        # If no valid sensors, still create the image but return empty pins and outputs
        if not np.any(valid_mask):
            image_tensor = self.create_multichannel_image(actual_idx)
            if self.normalize:
                image_tensor = self._normalize_image(image_tensor)
            return {
                'image': image_tensor,
                'pins': torch.zeros((0, 2), dtype=torch.long),
                'outputs': torch.zeros(0, dtype=torch.float32)
            }
        
        # Create the image tensor with all channels
        image_tensor = self.create_multichannel_image(actual_idx)
        if self.normalize:
            image_tensor = self._normalize_image(image_tensor)
        
        # Filter pixel map and values to only include valid sensors
        valid_indices = np.where(valid_mask)[0]
        valid_pins = []
        valid_outputs = []
        
        for i in valid_indices:
            if i in self.pixel_map:
                valid_pins.append(self.pixel_map[i])
                valid_outputs.append(pm_values[i])
        
        # Convert to tensors
        pins = torch.tensor(valid_pins, dtype=torch.long)
        outputs = torch.tensor(valid_outputs, dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'pins': pins,
            'outputs': outputs
        }
    
    def create_multichannel_image(self, index):
        num_channels = self.lookback + 3 + self.static_tensor.shape[0]
        image = torch.zeros((num_channels, self.grid_height, self.grid_width))
        
        # Historical PM2.5 channels
        for l in range(1, self.lookback + 1):
            t_idx = index - l
            if t_idx < 0:
                continue
            pm_values = self.pm25.iloc[t_idx].values
            # Use the modified interpolation function that handles NaNs
            image[l-1] = spatial_interpolate_with_nans(
                self.pixel_map, 
                torch.log(torch.tensor(pm_values + 1e-3)), 
                self.grid_height, 
                self.grid_width
            )
        
        # Temperature channel
        temp_values = self.temps.iloc[index].values
        image[self.lookback] = spatial_interpolate_with_nans(
            self.pixel_map, 
            temp_values, 
            self.grid_height, 
            self.grid_width
        )
        
        # Relative humidity channel
        rh_values = self.rh.iloc[index].values
        image[self.lookback + 1] = spatial_interpolate_with_nans(
            self.pixel_map, 
            rh_values, 
            self.grid_height, 
            self.grid_width
        )
        
        # Elevation channel - elevation is static, so no NaNs expected
        image[self.lookback + 2] = spatial_interpolate_with_nans(
            self.pixel_map, 
            self.elevation, 
            self.grid_height, 
            self.grid_width
        )
        
        # Static layers (map/satellite images)
        if self.static_tensor.numel() > 0:
            image[self.lookback + 3:] = self.static_tensor
        
        return image
    
    def _normalize_image(self, image):
        if self.stats is None:
            # Handle potential NaNs in the normalization
            mean = torch.nanmean(image.view(image.shape[0], -1), dim=1)
            std = torch.nanstd(image.view(image.shape[0], -1), dim=1)
            
            # Replace NaNs with zeros in case all values in a channel are NaN
            mean = torch.nan_to_num(mean, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0)  # Use 1.0 to avoid division by zero
        else:
            mean, std = self.stats
        
        # Reshape for broadcasting
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
        
        # Add small epsilon to std to avoid division by zero
        normalized = (image - mean) / (std + 1e-6)
        
        # Replace any remaining NaNs with zeros
        return torch.nan_to_num(normalized, nan=0.0)
    
    def compute_channel_stats(self, max_samples=500):
        sums = None
        sq_sums = None
        counts = None
        
        for i, sample in enumerate(self):
            img = sample['image']
            
            if sums is None:
                sums = torch.zeros(img.shape[0])
                sq_sums = torch.zeros(img.shape[0])
                counts = torch.zeros(img.shape[0])
            
            # Handle NaNs by only computing stats on valid values
            for c in range(img.shape[0]):
                channel = img[c]
                valid_mask = ~torch.isnan(channel)
                if valid_mask.any():
                    sums[c] += torch.sum(channel[valid_mask])
                    sq_sums[c] += torch.sum(channel[valid_mask] ** 2)
                    counts[c] += valid_mask.sum()
            
            if i >= max_samples:
                break
        
        # Compute mean and std while handling potential zeros in counts
        mean = torch.zeros_like(sums)
        std = torch.ones_like(sums)
        
        valid_mask = counts > 0
        mean[valid_mask] = sums[valid_mask] / counts[valid_mask]
        variance = sq_sums[valid_mask] / counts[valid_mask] - mean[valid_mask] ** 2
        std[valid_mask] = torch.sqrt(torch.clamp(variance, min=1e-6))
        
        return mean, std
