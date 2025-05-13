import torch
from torch.utils.data import Dataset
import random
import numpy as np

class MultiCropSensorDataset(Dataset):
    def __init__(self, base_datasets, num_crops_per_dataset=4, crop_size=(32, 32), max_attempts=30):
        """
        Create a dataset of cropped samples from multiple source datasets.
        
        Args:
            base_datasets: List of datasets to sample from
            num_crops_per_dataset: Number of crops to extract from each sample in each dataset
            crop_size: Size of the crops (height, width)
            max_attempts: Maximum number of attempts to find valid crops
        """
        self.base_datasets = base_datasets if isinstance(base_datasets, list) else [base_datasets]
        self.num_crops = num_crops_per_dataset
        self.crop_size = crop_size
        self.max_attempts = max_attempts
        self.index_map = []  # (dataset_idx, sample_idx, crop_spec)
        
        # Process each dataset
        for dataset_idx, dataset in enumerate(self.base_datasets):
            for base_idx in range(len(dataset)):
                base_sample = dataset[base_idx]
                image = base_sample['image']
                pins = base_sample['pins']
                
                # Skip samples with no valid pins
                if len(pins) == 0:
                    continue
                    
                H, W = image.shape[1], image.shape[2]
                ch, cw = crop_size
                
                # Ensure the image is large enough to crop
                if H < ch or W < cw:
                    print(f"Warning: Image size {(H, W)} is smaller than crop size {crop_size} in dataset {dataset_idx}, sample {base_idx}")
                    continue
                
                valid_crops = []
                attempts = 0
                
                while len(valid_crops) < num_crops_per_dataset and attempts < max_attempts:
                    top = random.randint(0, H - ch)
                    left = random.randint(0, W - cw)
                    
                    # Check if any pins fall within this crop
                    mask = (pins[:, 0] >= top) & (pins[:, 0] < top + ch) & \
                           (pins[:, 1] >= left) & (pins[:, 1] < left + cw)
                    
                    if mask.sum() > 0:
                        valid_crops.append((top, left))
                    
                    attempts += 1
                
                # Add all valid crops to the index map
                for crop in valid_crops:
                    self.index_map.append((dataset_idx, base_idx, crop))
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        dataset_idx, base_idx, (top, left) = self.index_map[idx]
        
        # Get the base sample from the appropriate dataset
        base_sample = self.base_datasets[dataset_idx][base_idx]
        
        # Extract the crop
        ch, cw = self.crop_size
        cropped_image = base_sample['image'][:, top:top+ch, left:left+cw]
        
        # Get the pins and outputs
        pins = base_sample['pins']
        outputs = base_sample['outputs']
        
        # Filter pins to only include those in the crop and adjust coordinates
        mask = (pins[:, 0] >= top) & (pins[:, 0] < top + ch) & \
               (pins[:, 1] >= left) & (pins[:, 1] < left + cw)
        
        cropped_pins = pins[mask] - torch.tensor([top, left])
        cropped_outputs = outputs[mask]
        
        return {
            'image': cropped_image,
            'pins': cropped_pins,
            'outputs': cropped_outputs,
            'dataset_idx': dataset_idx  # Optionally include source dataset info
        }
