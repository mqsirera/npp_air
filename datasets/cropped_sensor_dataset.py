import torch
from torch.utils.data import Dataset
import numpy as np
import random

class CroppedSensorDataset(Dataset):
    def __init__(self, full_dataset, crop_size=(32, 32), crop_attempts=10):
        self.full_dataset = full_dataset
        self.crop_size = crop_size
        self.crop_attempts = crop_attempts
        self.valid_indices = list(range(len(full_dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        base_sample = self.full_dataset[self.valid_indices[idx]]
        full_image = base_sample['image']
        pins = base_sample['pins']
        outputs = base_sample['outputs']

        H, W = full_image.shape[1], full_image.shape[2]
        ch, cw = self.crop_size

        for _ in range(self.crop_attempts):
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)

            # Check how many sensors fall into the crop
            mask = (pins[:, 0] >= top) & (pins[:, 0] < top + ch) & (pins[:, 1] >= left) & (pins[:, 1] < left + cw)
            if mask.sum() > 0:
                cropped_image = full_image[:, top:top+ch, left:left+cw]
                cropped_pins = pins[mask] - torch.tensor([top, left])
                cropped_outputs = outputs[mask]
                return {
                    'image': cropped_image,
                    'pins': cropped_pins,
                    'outputs': cropped_outputs
                }

        # Fallback: return full image if no crop found
        return {
            'image': full_image,
            'pins': pins,
            'outputs': outputs
        }