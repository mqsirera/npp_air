import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from training.experiments_save import *
from datasets.cropped_sensor_dataset import CroppedSensorDataset
from datasets.multi_crop_sensor_dataset import MultiCropSensorDataset
from datasets.sensor_image_dataset import MultichannelSensorImageDataset
from datasets.auxiliar_dataset import *

# Config
DATA_FOLDER_1 = "./data/Pollution/Brookline"
DATA_FOLDER_2 = "./data/Pollution/Chelsea"  # Second data folder
LOOKBACK = 5
PIXELS_PER_DEGREE = 4000
CROP_SIZE = (32, 32)
EPOCHS = 20

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

# Check datasets correctness
check_and_fix_dataset(raw_dataset1, num_samples=50)
check_and_fix_dataset(raw_dataset2, num_samples=50)

check_and_fix_dataset(train_dataset, num_samples=100, fix_nans=False)
check_and_fix_dataset(val_dataset, num_samples=50, fix_nans=False)
check_and_fix_dataset(test_dataset, num_samples=50, fix_nans=False)

validate_loaders(train_loader, val_loader, test_loader)

# Train and evaluate
mse_model = train_model(mse_model, train_loader, val_loader, test_loader, name="MSE", device='cpu', epochs=EPOCHS)
npp_model = train_model(npp_model, train_loader, val_loader, test_loader, name="NPP", device='cpu', epochs=EPOCHS)

# Extract features
X_train, y_train = extract_temporal_features(train_dataset, temporal_channels=LOOKBACK+1)
X_test, y_test = extract_temporal_features(test_dataset, temporal_channels=LOOKBACK+1)

# Train classical regressors
classical_results = train_and_compare_classical_models(X_train, y_train, X_test, y_test)

mse_mse, mse_r2 = evaluate_model(mse_model, test_loader, name="MSE", device='cpu')
npp_mse, npp_r2 = evaluate_model(npp_model, test_loader, name="NPP", device='cpu')

# Save all results
print("\nSaving results...")
experiment_name = "baseline_experiment_lookback5"  # Change this for different runs
saved_paths = save_model_results(
    classical_results, mse_mse, mse_r2, npp_mse, npp_r2,
    experiment_name=experiment_name
)

# Gather results for analysis
model_names = ["MSE", "NPP"] + list(classical_results.keys())
mses = [mse_mse, npp_mse] + [classical_results[k]["mse"] for k in classical_results]
r2s = [mse_r2, npp_r2] + [classical_results[k]["r2"] for k in classical_results]

# Create comparison plots and save them
print("\nCreating and saving plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MSE comparison
bars1 = ax1.bar(model_names, mses, color='skyblue', alpha=0.7, edgecolor='navy')
ax1.set_title("Test MSE Comparison", fontsize=14, fontweight='bold')
ax1.set_ylabel("Mean Squared Error", fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mse in zip(bars1, mses):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{mse:.4f}', ha='center', va='bottom', fontsize=10)

# R² comparison
bars2 = ax2.bar(model_names, r2s, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
ax2.set_title("Test R² Comparison", fontsize=14, fontweight='bold')
ax2.set_ylabel("R² Score", fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, r2 in zip(bars2, r2s):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{r2:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save the plot
plot_path = os.path.join("model_results", f"{experiment_name}_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {plot_path}")
plt.show()

# Create a detailed performance table
print("\n" + "="*60)
print("DETAILED RESULTS SUMMARY")
print("="*60)

# Load the saved results to display
results_data = load_results(experiment_name)
if 'dataframe' in results_data:
    df = results_data['dataframe'].copy()
    df['Rank_MSE'] = df['MSE'].rank()
    df['Rank_R2'] = df['R2'].rank(ascending=False)
    
    print("\nModel Performance (sorted by MSE):")
    print("-" * 50)
    for idx, row in df.iterrows():
        print(f"{row['Model']:12} | MSE: {row['MSE']:8.4f} (#{int(row['Rank_MSE'])}) | "
              f"R²: {row['R2']:7.4f} (#{int(row['Rank_R2'])})")

# Find best performing models
best_mse_idx = np.argmin(mses)
best_r2_idx = np.argmax(r2s)

print(f"\n🏆 BEST PERFORMERS:")
print(f"   Lowest MSE:  {model_names[best_mse_idx]} ({mses[best_mse_idx]:.4f})")
print(f"   Highest R²:  {model_names[best_r2_idx]} ({r2s[best_r2_idx]:.4f})")

# Statistical analysis
print(f"\n📊 STATISTICS:")
print(f"   MSE Range:   {min(mses):.4f} - {max(mses):.4f}")
print(f"   R² Range:    {min(r2s):.4f} - {max(r2s):.4f}")
print(f"   MSE Std Dev: {np.std(mses):.4f}")
print(f"   R² Std Dev:  {np.std(r2s):.4f}")

# Create a performance heatmap
print("\nCreating performance heatmap...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Normalize metrics for better visualization
mse_norm = (np.array(mses) - min(mses)) / (max(mses) - min(mses))
r2_norm = (np.array(r2s) - min(r2s)) / (max(r2s) - min(r2s))

# Create heatmap data
heatmap_data = np.array([mse_norm, 1-r2_norm]).T  # Invert R² so lower is better
heatmap_labels = np.array([mses, r2s]).T

im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')

# Set ticks and labels
ax.set_xticks([0, 1])
ax.set_xticklabels(['MSE (↓)', 'R² (↑)'])
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names)

# Add text annotations
for i in range(len(model_names)):
    for j in range(2):
        if j == 0:  # MSE
            text = f'{heatmap_labels[i, j]:.4f}'
        else:  # R²
            text = f'{heatmap_labels[i, j]:.4f}'
        ax.text(j, i, text, ha="center", va="center", 
                color="white" if heatmap_data[i, j] > 0.5 else "black", fontweight='bold')

ax.set_title('Model Performance Heatmap\n(Darker = Worse Performance)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Performance (0=Best, 1=Worst)')

plt.tight_layout()

# Save heatmap
heatmap_path = os.path.join("model_results", f"{experiment_name}_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"Performance heatmap saved to: {heatmap_path}")
plt.show()

# Summary of saved files
print("\n" + "="*60)
print("FILES SAVED:")
print("="*60)
for key, path in saved_paths.items():
    print(f"  {key.replace('_', ' ').title()}: {path}")
print(f"  Comparison Plot: {plot_path}")
print(f"  Performance Heatmap: {heatmap_path}")

print(f"\n✅ Experiment '{experiment_name}' completed successfully!")
print(f"📁 All files saved in: model_results/")

# Optional: Quick comparison with previous experiments (if any exist)
print("\n" + "="*60)
print("EXPERIMENT HISTORY:")
print("="*60)

try:
    # List all experiment files
    results_dir = "model_results"
    if os.path.exists(results_dir):
        experiment_files = [f for f in os.listdir(results_dir) if f.endswith('_summary.json')]
        if len(experiment_files) > 1:
            print("Previous experiments found:")
            for file in sorted(experiment_files):
                exp_name = file.replace('_summary.json', '')
                print(f"  - {exp_name}")
            
            print(f"\nTo compare experiments, use:")
            print(f"comparison_df = compare_experiments(['{experiment_name}', 'other_experiment_name'])")
        else:
            print("This is your first saved experiment!")
    else:
        print("Results directory created for the first time.")
except Exception as e:
    print(f"Could not check experiment history: {e}")

print("\n🎉 Analysis complete! Check the saved files for detailed results.")
