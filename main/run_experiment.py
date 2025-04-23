import os
import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, random_split
from utils.features import extract_temporal_features
from models.classical_models import train_and_compare_classical_models
import matplotlib.pyplot as plt

from utils.image_utils import load_base_images
from utils.interpolation import spatial_interpolate
from models.autoencoder import Autoencoder
from models.mse_model import MSEModel
from models.npp_model import NPPModel
from training.train_eval import train_model, evaluate_model
from datasets.cropped_sensor_dataset import CroppedSensorDataset
from datasets.multi_crop_sensor_dataset import MultiCropSensorDataset
from datasets.sensor_image_dataset import MultichannelSensorImageDataset

# Config
DATA_FOLDER = "./data/Pollution/Brookline"
LOOKBACK = 5
PIXELS_PER_DEGREE = 2000
CROP_SIZE = (64, 64)
EPOCHS = 20

# Load data
temps = pd.read_csv(os.path.join(DATA_FOLDER, "temp.csv"))
rh = pd.read_csv(os.path.join(DATA_FOLDER, "rh.csv"))
pm25 = pd.read_csv(os.path.join(DATA_FOLDER, "pm25.csv"))
sens_df = pd.read_csv(os.path.join(DATA_FOLDER, "sensor_data_with_elevation.csv"))
with open(os.path.join(DATA_FOLDER, "sens_code.pkl"), "rb") as f:
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
    os.path.join(DATA_FOLDER, "image_map.png"),
    os.path.join(DATA_FOLDER, "image_sat.png")
]
base_images_tensor = load_base_images(base_paths, grid_height, grid_width)

raw_dataset = MultichannelSensorImageDataset(pm25, temps, rh, elevation.values, sensor_pixel_map, base_images_tensor, lookback=LOOKBACK)
mean, std = raw_dataset.compute_channel_stats(max_samples=300)
full_dataset = MultichannelSensorImageDataset(pm25, temps, rh, elevation.values, sensor_pixel_map, base_images_tensor, lookback=LOOKBACK, normalize=True, stats=(mean, std))

# Split chronologically
total_len = len(full_dataset)
train_len = int(0.7 * total_len)
val_len = int(0.15 * total_len)
test_len = total_len - train_len - val_len
train_base, val_base, test_base = random_split(full_dataset, [train_len, val_len, test_len])

# Wrap with crop augmentation
train_dataset = MultiCropSensorDataset(train_base, num_crops=8, crop_size=CROP_SIZE)
val_dataset = MultiCropSensorDataset(val_base, num_crops=8, crop_size=CROP_SIZE)   # maybe fewer for val
test_dataset = MultiCropSensorDataset(test_base, num_crops=8, crop_size=CROP_SIZE)  # 1 crop for consistent testing
#train_dataset = CroppedSensorDataset(train_base, crop_size=CROP_SIZE)
#val_dataset = CroppedSensorDataset(val_base, crop_size=CROP_SIZE)
#test_dataset = CroppedSensorDataset(test_base, crop_size=CROP_SIZE)

def custom_collate_fn(batch):
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'pins': [x['pins'] for x in batch],  # list of variable-length tensors
        'outputs': [x['outputs'] for x in batch]
    }

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)

# Models
input_channels = LOOKBACK + 3 + base_images_tensor.shape[0]
input_shape = (CROP_SIZE[0], CROP_SIZE[1])

auto_mse = Autoencoder([32, 64], [32], input_shape, input_channels, param_size=0)
auto_npp = Autoencoder([32, 64], [32], input_shape, input_channels, param_size=1)

mse_model = MSEModel(auto_mse)
npp_model = NPPModel(auto_npp, kernel_mode="predicted")

# Train and evaluate
mse_model = train_model(mse_model, train_loader, val_loader, test_loader, name="MSE", device='cuda', epochs=EPOCHS)
npp_model = train_model(npp_model, train_loader, val_loader, test_loader, name="NPP", device='cuda', epochs=EPOCHS)

# Extract features
X_train, y_train = extract_temporal_features(train_dataset, temporal_channels=LOOKBACK+1)
X_test, y_test = extract_temporal_features(test_dataset, temporal_channels=LOOKBACK+1)

# Train classical regressors
classical_results = train_and_compare_classical_models(X_train, y_train, X_test, y_test)

mse_mse, mse_r2 = evaluate_model(mse_model, test_loader, name="MSE", device='cuda')
npp_mse, npp_r2 = evaluate_model(npp_model, test_loader, name="NPP", device='cuda')

# Gather results
model_names = ["MSE", "NPP"] + list(classical_results.keys())
mses = [mse_mse, npp_mse] + [classical_results[k]["mse"] for k in classical_results]
r2s = [mse_r2, npp_r2] + [classical_results[k]["r2"] for k in classical_results]

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(model_names, mses, color='skyblue')
plt.title("Test MSE Comparison")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(model_names, r2s, color='lightgreen')
plt.title("Test R² Comparison")
plt.ylabel("R² Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()