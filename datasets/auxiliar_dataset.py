import os
import torch
from torch.utils.data import Dataset, random_split
from utils.image_utils import load_base_images
import numpy as np
import pandas as pd
import pickle

def load_dataset_from_folder(data_folder, PIXELS_PER_DEGREE=2000):
    """Load and process data from a specific folder"""
    # Load data
    temps = pd.read_csv(os.path.join(data_folder, "temp.csv"))
    rh = pd.read_csv(os.path.join(data_folder, "rh.csv"))
    pm25 = pd.read_csv(os.path.join(data_folder, "pm25.csv"))
    sens_df = pd.read_csv(os.path.join(data_folder, "sensor_data_with_elevation.csv"))
    
    with open(os.path.join(data_folder, "sens_code.pkl"), "rb") as f:
        sens_code = pickle.load(f)
    
    # Time alignment
    timestamps = pd.date_range(start="2024-05-24 00:00", periods=len(temps), freq="H")
    for df in [temps, rh, pm25]:
        df.index = timestamps
    
    # Normalize elevation
    elevation = (sens_df["Elevation"] - sens_df["Elevation"].min()) / (sens_df["Elevation"].max() - sens_df["Elevation"].min())
    
    # Map to pixel
    lat_min, lat_max = sens_df["Latitude"].min(), sens_df["Latitude"].max()
    lon_min, lon_max = sens_df["Longitude"].min(), sens_df["Longitude"].max()
    grid_height = int(np.ceil((lat_max - lat_min) * PIXELS_PER_DEGREE))
    grid_width = int(np.ceil((lon_max - lon_min) * PIXELS_PER_DEGREE))
    
    def map_to_pixel(lat, lon):
        x = int((lat_max - lat) * PIXELS_PER_DEGREE)
        y = int((lon - lon_min) * PIXELS_PER_DEGREE)
        x = min(max(x, 0), grid_height - 1)
        y = min(max(y, 0), grid_width - 1)
        return x, y
    
    sensor_pixel_map = {i: map_to_pixel(lat, lon) for i, (lat, lon) in enumerate(zip(sens_df["Latitude"], sens_df["Longitude"]))}
    
    # Load base layers
    base_paths = [
        os.path.join(data_folder, "image_map.png"),
        os.path.join(data_folder, "image_sat.png")
    ]
    base_images_tensor = load_base_images(base_paths, grid_height, grid_width)
    
    return {
        'pm25': pm25,
        'temps': temps,
        'rh': rh,
        'elevation': elevation.values,
        'sensor_pixel_map': sensor_pixel_map,
        'base_images_tensor': base_images_tensor,
        'grid_dimensions': (grid_height, grid_width),
        'sens_code': sens_code
    }

# Compute channel stats from combined datasets
def compute_combined_stats(dataset1, dataset2, max_samples=300):
    # Get samples from both datasets
    samples_per_dataset = max_samples // 2
    
    # Select random indices from each dataset
    indices1 = np.random.choice(range(len(dataset1)), min(samples_per_dataset, len(dataset1)), replace=False)
    indices2 = np.random.choice(range(len(dataset2)), min(samples_per_dataset, len(dataset2)), replace=False)
    
    # Collect channel data separately for each dataset
    channel_sums = None
    channel_squared_sums = None
    total_pixels = 0
    num_channels = None
    
    # Process dataset 1
    for idx in indices1:
        sample = dataset1[idx]
        img = sample['image']
        
        if num_channels is None:
            num_channels = img.shape[0]
            channel_sums = torch.zeros(num_channels, device=img.device)
            channel_squared_sums = torch.zeros(num_channels, device=img.device)
        
        # Sum across spatial dimensions
        channel_sums += img.sum(dim=(1, 2))
        channel_squared_sums += (img ** 2).sum(dim=(1, 2))
        total_pixels += img.shape[1] * img.shape[2]
    
    # Process dataset 2
    for idx in indices2:
        sample = dataset2[idx]
        img = sample['image']
        
        # Sum across spatial dimensions
        channel_sums += img.sum(dim=(1, 2))
        channel_squared_sums += (img ** 2).sum(dim=(1, 2))
        total_pixels += img.shape[1] * img.shape[2]
    
    # Compute mean and standard deviation
    mean = channel_sums / total_pixels
    variance = (channel_squared_sums / total_pixels) - (mean ** 2)
    std = torch.sqrt(variance + 1e-8)  # Small epsilon to avoid numerical issues
    
    return mean, std

# Split each dataset chronologically
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

# Create a ResizeDataset wrapper to standardize dimensions
class ResizedSensorDataset(Dataset):
    def __init__(self, base_dataset, target_size):
        """
        Wrapper dataset that resizes images to a standard size
        
        Args:
            base_dataset: The original dataset
            target_size: Tuple (height, width) for the target size
        """
        self.base_dataset = base_dataset
        self.target_size = target_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Get original dimensions
        _, h, w = sample['image'].shape
        
        # Create interpolation scale factors
        h_scale = self.target_size[0] / h
        w_scale = self.target_size[1] / w
        
        # Resize the image using interpolation
        resized_image = torch.nn.functional.interpolate(
            sample['image'].unsqueeze(0),  # Add batch dimension for interpolate
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Scale the pin coordinates to match the new dimensions
        scaled_pins = sample['pins'].clone()
        scaled_pins[:, 0] = scaled_pins[:, 0] * h_scale
        scaled_pins[:, 1] = scaled_pins[:, 1] * w_scale
        
        # Ensure pin coordinates are within bounds
        scaled_pins[:, 0] = torch.clamp(scaled_pins[:, 0], 0, self.target_size[0] - 1)
        scaled_pins[:, 1] = torch.clamp(scaled_pins[:, 1], 0, self.target_size[1] - 1)
        
        return {
            'image': resized_image,
            'pins': scaled_pins,
            'outputs': sample['outputs']
        }

def custom_collate_fn(batch):
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'pins': [x['pins'] for x in batch],  # list of variable-length tensors
        'outputs': [x['outputs'] for x in batch]
    }

def check_and_fix_dataset(dataset, num_samples=None, fix_nans=False, replace_value=0.0):
    """
    Check a dataset for NaN/Inf values and optionally fix them.
    
    Args:
        dataset: The dataset to check
        num_samples: Number of samples to check (None for all)
        fix_nans: Whether to fix NaN/Inf values or just report them
        replace_value: Value to use for replacing NaNs/Infs (if fix_nans=True)
        
    Returns:
        bool: True if the dataset is clean or has been fixed, False otherwise
    """
    if num_samples is None:
        indices = range(len(dataset))
    else:
        indices = np.random.choice(range(len(dataset)), min(num_samples, len(dataset)), replace=False)
    
    total_samples = len(indices)
    samples_with_issues = 0
    fixed_samples = 0
    
    print(f"Checking {total_samples} samples for NaN/Inf values...")
    
    for idx in indices:
        sample = dataset[idx]
        has_issues = False
        
        # Check each tensor in the sample
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                # Check for NaN/Inf values
                if torch.isnan(value).any() or torch.isinf(value).any():
                    has_issues = True
                    samples_with_issues += 1
                    
                    print(f"Sample {idx}, '{key}' tensor contains NaN/Inf values")
                    print(f"  Shape: {value.shape}, NaNs: {torch.isnan(value).sum().item()}, Infs: {torch.isinf(value).sum().item()}")
                    
                    # Fix the values if requested
                    if fix_nans:
                        # Replace NaN/Inf with the specified value
                        mask = torch.isnan(value) | torch.isinf(value)
                        if mask.any():
                            # Create a fixed copy
                            fixed_value = value.clone()
                            fixed_value[mask] = replace_value
                            
                            # Update the sample
                            sample[key] = fixed_value
                            fixed_samples += 1
                            
                            print(f"  Fixed by replacing {mask.sum().item()} values with {replace_value}")
    
    if samples_with_issues == 0:
        print("No NaN/Inf values found. Dataset is clean!")
        return True
    elif fix_nans:
        print(f"Fixed {fixed_samples} samples with NaN/Inf values.")
        return True
    else:
        print(f"Found {samples_with_issues} samples with NaN/Inf values. Use fix_nans=True to fix them.")
        return False

def validate_loaders(train_loader, val_loader, test_loader):
    """
    Check all data loaders for NaN/Inf values to prevent training failures.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        
    Returns:
        bool: True if all loaders are clean, False otherwise
    """
    print("Validating data loaders for NaN/Inf values...")
    
    # Helper function to check a batch
    def check_batch(batch, batch_idx):
        has_issues = False
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    has_issues = True
                    print(f"  Batch {batch_idx}, '{key}' tensor contains NaN/Inf values")
                    print(f"    Shape: {value.shape}, NaNs: {torch.isnan(value).sum().item()}, "
                          f"Infs: {torch.isinf(value).sum().item()}")
            elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                # Handle lists of tensors (like 'pins' and 'outputs')
                for i, tensor in enumerate(value):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        has_issues = True
                        print(f"  Batch {batch_idx}, '{key}' list item {i} contains NaN/Inf values")
                        print(f"    Shape: {tensor.shape}, NaNs: {torch.isnan(tensor).sum().item()}, "
                              f"Infs: {torch.isinf(tensor).sum().item()}")
        
        return has_issues
    
    # Check each loader
    all_clean = True
    
    print("Checking training loader...")
    for batch_idx, batch in enumerate(train_loader):
        if check_batch(batch, batch_idx):
            all_clean = False
        if batch_idx >= 2:  # Check first few batches only
            break
    
    print("Checking validation loader...")
    for batch_idx, batch in enumerate(val_loader):
        if check_batch(batch, batch_idx):
            all_clean = False
        if batch_idx >= 2:
            break
            
    print("Checking test loader...")
    for batch_idx, batch in enumerate(test_loader):
        if check_batch(batch, batch_idx):
            all_clean = False
        if batch_idx >= 2:
            break
    
    if all_clean:
        print("All data loaders are clean!")
    else:
        print("Found NaN/Inf values in data loaders!")
        
    return all_clean