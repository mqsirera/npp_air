import torch
from torch.utils.data import Dataset
from utils.interpolation import spatial_interpolate

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
        image_tensor = self.create_multichannel_image(actual_idx)

        if self.normalize:
            image_tensor = self._normalize_image(image_tensor)

        pins = torch.tensor(list(self.pixel_map.values()), dtype=torch.long)
        outputs = torch.tensor(self.pm25.iloc[actual_idx].values, dtype=torch.float32)

        return {
            'image': image_tensor,
            'pins': pins,
            'outputs': outputs
        }

    def create_multichannel_image(self, index):
        num_channels = self.lookback + 3 + self.static_tensor.shape[0]
        image = torch.zeros((num_channels, self.grid_height, self.grid_width))

        for l in range(1, self.lookback + 1):
            t_idx = index - l
            if t_idx < 0:
                continue
            pm_values = self.pm25.iloc[t_idx].values
            image[l] = spatial_interpolate(self.pixel_map, torch.log(torch.tensor(pm_values + 1e-3)), self.grid_height, self.grid_width)

        temp_values = self.temps.iloc[index].values
        image[self.lookback] = spatial_interpolate(self.pixel_map, temp_values, self.grid_height, self.grid_width)

        rh_values = self.rh.iloc[index].values
        image[self.lookback + 1] = spatial_interpolate(self.pixel_map, rh_values, self.grid_height, self.grid_width)

        image[self.lookback + 2] = spatial_interpolate(self.pixel_map, self.elevation, self.grid_height, self.grid_width)

        if self.static_tensor.numel() > 0:
            image[self.lookback + 3:] = self.static_tensor

        return image

    def _normalize_image(self, image):
        if self.stats is None:
            mean = image.view(image.shape[0], -1).mean(dim=1, keepdim=True)
            std = image.view(image.shape[0], -1).std(dim=1, keepdim=True)
        else:
            mean, std = self.stats
            mean = mean[:, None, None]
            std = std[:, None, None]
        return (image - mean) / (std + 1e-6)

    def compute_channel_stats(self, max_samples=500):
        sums = None
        sq_sums = None
        count = 0

        for i, sample in enumerate(self):
            img = sample['image']
            if sums is None:
                sums = torch.zeros(img.shape[0])
                sq_sums = torch.zeros(img.shape[0])

            sums += img.view(img.shape[0], -1).mean(dim=1)
            sq_sums += img.view(img.shape[0], -1).pow(2).mean(dim=1)
            count += 1

            if i >= max_samples:
                break

        mean = sums / count
        std = (sq_sums / count - mean ** 2).sqrt()
        return mean, std