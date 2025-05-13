import os
import torch
import random
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.features import extract_temporal_features
from utils.image_utils import load_base_images
from utils.interpolation import spatial_interpolate
from models.autoencoder import Autoencoder
from models.mse_model import MSEModel
from models.npp_model import NPPModel
from models.classical_models import train_and_compare_classical_models
from training.train_eval import train_model, evaluate_model
from datasets.cropped_sensor_dataset import CroppedSensorDataset
from datasets.multi_crop_sensor_dataset import MultiCropSensorDataset
from datasets.sensor_image_dataset import MultichannelSensorImageDataset
from datasets.auxiliar_dataset import *

# Config
DATA_FOLDER_1 = "./data/Pollution/Brookline"
DATA_FOLDER_2 = "./data/Pollution/Chelsea"  # Second data folder
LOOKBACK = 5
PIXELS_PER_DEGREE = 3000
CROP_SIZE = (64, 64)
EPOCHS = 10

# Load data from both folders
print(f"Loading data from {DATA_FOLDER_1}...")
data1 = load_dataset_from_folder(DATA_FOLDER_1, PIXELS_PER_DEGREE)
print(f"Loading data from {DATA_FOLDER_2}...")
data2 = load_dataset_from_folder(DATA_FOLDER_2, PIXELS_PER_DEGREE)

# Print grid dimensions for debugging
print(f"Grid dimensions for dataset 1: {data1['grid_dimensions']}")
print(f"Grid dimensions for dataset 2: {data2['grid_dimensions']}")

# Create raw datasets from both sources
print("Creating raw datasets...")
raw_dataset1 = MultichannelSensorImageDataset(
    data1['pm25'], data1['temps'], data1['rh'], 
    data1['elevation'], data1['sensor_pixel_map'], 
    data1['base_images_tensor'], lookback=LOOKBACK
)

raw_dataset2 = MultichannelSensorImageDataset(
    data2['pm25'], data2['temps'], data2['rh'], 
    data2['elevation'], data2['sensor_pixel_map'], 
    data2['base_images_tensor'], lookback=LOOKBACK
)

# Compute combined stats for normalization
combined_mean, combined_std = compute_combined_stats(raw_dataset1, raw_dataset2, max_samples=300)

# Create normalized datasets using combined stats
print("Creating normalized datasets...")
full_dataset1 = MultichannelSensorImageDataset(
    data1['pm25'], data1['temps'], data1['rh'], 
    data1['elevation'], data1['sensor_pixel_map'], 
    data1['base_images_tensor'], lookback=LOOKBACK, 
    normalize=True, stats=(combined_mean, combined_std)
)

full_dataset2 = MultichannelSensorImageDataset(
    data2['pm25'], data2['temps'], data2['rh'], 
    data2['elevation'], data2['sensor_pixel_map'], 
    data2['base_images_tensor'], lookback=LOOKBACK, 
    normalize=True, stats=(combined_mean, combined_std)
)

# Split datasets chronologically
print("Splitting datasets...")
train_base1, val_base1, test_base1 = split_dataset(full_dataset1)
train_base2, val_base2, test_base2 = split_dataset(full_dataset2)

print(f"Dataset 1 - Train: {len(train_base1)}, Val: {len(val_base1)}, Test: {len(test_base1)}")
print(f"Dataset 2 - Train: {len(train_base2)}, Val: {len(val_base2)}, Test: {len(test_base2)}")

# Initialize multi-crop datasets directly from the base datasets (no resizing needed)
print("Creating multi-crop datasets...")

# Training set from both datasets
train_dataset = MultiCropSensorDataset(
    base_datasets=[train_base1, train_base2],
    num_crops_per_dataset=4,  # 4 crops per sample from each dataset
    crop_size=CROP_SIZE,
    max_attempts=30
)

# Validation set from both datasets
val_dataset = MultiCropSensorDataset(
    base_datasets=[val_base1, val_base2],
    num_crops_per_dataset=4,
    crop_size=CROP_SIZE,
    max_attempts=30
)

# Test set from both datasets
test_dataset = MultiCropSensorDataset(
    base_datasets=[test_base1, test_base2],
    num_crops_per_dataset=4,
    crop_size=CROP_SIZE,
    max_attempts=30
)

print(f"Combined Multi-crop - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)

# Models
input_channels = LOOKBACK + 3 + data1['base_images_tensor'].shape[0]
input_shape = CROP_SIZE
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
