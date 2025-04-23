import torch
from torch.utils.data import Dataset
import random

class MultiCropSensorDataset(Dataset):
    def __init__(self, base_dataset, num_crops=4, crop_size=(32, 32), max_attempts=30):
        self.base_dataset = base_dataset
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.max_attempts = max_attempts

        self.index_map = []  # (base_idx, crop_spec)

        for base_idx in range(len(base_dataset)):
            base_sample = base_dataset[base_idx]
            image = base_sample['image']
            pins = base_sample['pins']
            H, W = image.shape[1], image.shape[2]
            ch, cw = crop_size

            valid_crops = []
            attempts = 0

            while len(valid_crops) < num_crops and attempts < max_attempts:
                top = random.randint(0, H - ch)
                left = random.randint(0, W - cw)

                mask = (pins[:, 0] >= top) & (pins[:, 0] < top + ch) & (pins[:, 1] >= left) & (pins[:, 1] < left + cw)
                if mask.sum() > 0:
                    valid_crops.append((top, left))

                attempts += 1

            for crop in valid_crops:
                self.index_map.append((base_idx, crop))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        base_idx, (top, left) = self.index_map[idx]
        base_sample = self.base_dataset[base_idx]

        ch, cw = self.crop_size
        cropped_image = base_sample['image'][:, top:top+ch, left:left+cw]

        pins = base_sample['pins']
        outputs = base_sample['outputs']

        mask = (pins[:, 0] >= top) & (pins[:, 0] < top + ch) & (pins[:, 1] >= left) & (pins[:, 1] < left + cw)
        cropped_pins = pins[mask] - torch.tensor([top, left])
        cropped_outputs = outputs[mask]

        return {
            'image': cropped_image,
            'pins': cropped_pins,
            'outputs': cropped_outputs
        }