# Neural Point Processes for Sparse Air Pollution Regression

This repository implements Neural Point Processes (NPPs) for pixel-wise regression with sparse supervision, and applies them to the task of estimating urban air pollution (PM2.5) from a small set of sensor readings.

Unlike typical image regression tasks with dense labels, here the goal is to predict pollution levels at every pixel in a 2D grid from:
- a small set of sparse ground-truth labels (sensor readings),
- image-like features constructed from static context (e.g., satellite imagery, elevation),
- dynamic features (e.g., interpolated temperature, humidity, PM2.5 history).

<p align="center">
  <img src="figures/pixelwise_regression.png" width="700"/>
</p>

---

## 🌍 Why Air Pollution?

Air pollution monitoring suffers from spatial sparsity: sensors are costly, often solar-powered, and can fail intermittently.  
This makes traditional grid-based training ineffective. We need models that generalize across space and can adapt to new sensor readings at test time.

Neural Point Processes (NPPs) are ideal for this:
- They model per-image Gaussian Processes with CNN-predicted means.
- They naturally enforce spatial smoothness using a learnable kernel.
- They allow test-time refinement using newly revealed sensor values.

---

## 🧠 Key Features

- End-to-end pipeline for training and evaluating NPPs on real-world PM2.5 sensor data.
- Construction of multichannel inputs from sensor time series and static map layers.
- Spatial interpolation with guaranteed alignment of sensor coordinates.
- Multi-crop data augmentation: each image generates valid random crops with at least one sensor.
- Support for multiple loss modes: MSE baseline and full NPP kernel loss.
- Easy comparison with classical ML models (Random Forest, SVR, Ridge, KNN).

---

## 🗂 Repository Structure

```
npp_experiment/
├── datasets/             # Data loaders for full-frame and cropped datasets
│   ├── sensor_image_dataset.py
│   ├── multi_crop_sensor_dataset.py
├── models/               # Autoencoder, MSE model, NPP model, GP kernel
│   ├── autoencoder.py
│   ├── mse_model.py
│   ├── npp_model.py
├── training/             # Training loop and evaluation metrics
│   └── train_eval.py
├── utils/                # Interpolation, visualization, feature extractors
│   ├── interpolation.py
│   ├── features.py
├── main/
│   └── run_experiment.py # Full experiment: training + evaluation + comparison
├── figures/              # Visuals for report/manuscript
└── data/                 # Not included in repo – must be added manually
```

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/npp_experiment.git
cd npp_experiment

# Create a conda environment
conda create -n npp python=3.9
conda activate npp

# Install requirements
pip install -r requirements.txt
```

---

## 📊 Data Preparation

Place your input data in the following structure:

```
data/Pollution/Brookline/
├── pm25.csv
├── temp.csv
├── rh.csv
├── sensor_data_with_elevation.csv
├── sens_code.pkl
├── image_map.png
├── image_sat.png
```

All time series must be hourly aligned. Static layers like satellite images or maps are resized and aligned to the sensor bounding box using `PIXELS_PER_DEGREE`.

---

## 🚀 Running the Experiment

Run everything from data processing to evaluation with:

```bash
python -m main.run_experiment
```

This will:
- Load and interpolate sensor data
- Build and normalize multi-channel images
- Generate crops with at least one labeled sensor
- Train MSE and NPP models
- Evaluate on temporally held-out test set
- Train classical baselines for comparison

Results (MSE, R²) are printed and plotted.

---

## 🔍 Baselines and Models

| Model         | Description                              |
|---------------|-------------------------------------------|
| MSE Autoencoder | CNN trained with standard MSE on sensor pixels |
| NPP (ours)    | CNN + GP kernel loss (static or predicted kernel) |
| Ridge         | Linear regression on temporal sensor features |
| SVR           | Support Vector Regression on same features |
| Random Forest | Tree ensemble using local temporal features |
| KNN           | Nearest neighbor in temporal feature space |

---

## 📈 Example Outputs

- Per-pixel predictions overlaid with true sensor values.
- Evaluation curves: MSE, R² by epoch.
- Model comparison bar plots.

---

## 🛠 Customization

| To change                      | Edit in                                  |
|-------------------------------|-------------------------------------------|
| Lookback window               | `LOOKBACK_HOURS` in `run_experiment.py`   |
| Kernel type (fixed, learned)  | `kernel_mode` in `NPPModel`               |
| Crop size / number            | `multi_crop_sensor_dataset.py`            |
| Sensor interpolation method   | `utils/interpolation.py`                  |
| Normalization                 | Dataset construction stats                |
| Loss function                 | `train_eval.py`, model class              |

---

## 📚 Citation

If you use this work, please cite the NPP paper:

```bibtex
@inproceedings{shi2025neural,
  title={Neural Point Processes for Pixel-wise Regression},
  author={Shi, C. and Özcan, G. and Sirera Perelló, M. and Li, Y. and Iftikhar, N. and Ioannidis, S.},
  booktitle={Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2025}
}
```

---

## 📄 License

MIT License.  
See `LICENSE` file for full terms.

---

## 🙌 Acknowledgements

This project builds upon the foundational Neural Point Process codebase and adapts it to a new domain: environmental monitoring.  
Special thanks to open air-quality initiatives for releasing high-resolution PM2.5 data and to the AISTATS 2025 authors for foundational theory.