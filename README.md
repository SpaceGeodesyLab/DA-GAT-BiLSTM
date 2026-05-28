# DA-GAT-BiLSTM

---

## For software usage issues

Please contact:
**zhkhuang@whu.edu.cn** (Zhengkai Huang, corresponding author)
and **13775871578@163.com** (Siyu Zhu, first author).

> Note: please verify that the contact email matches the corresponding author's current institutional address before release.

---

## Overview

Accurate prediction of Global Navigation Satellite System (GNSS) coordinate displacement time series is fundamental to crustal deformation monitoring, seismic hazard assessment, and geophysical modeling. GNSS displacement signals exhibit strong spatial correlations across station networks, yet most existing approaches treat each station and each displacement component independently, failing to exploit inter-station tectonic dependencies.

We present **DA-GAT-BiLSTM**, a Python deep learning framework that integrates inter-station spatial graph modeling with bidirectional temporal sequence learning in an end-to-end pipeline. The model constructs a KNN-based geodesic station graph and uses stacked GATv2 layers to learn adaptive inter-station attention weights, a Bidirectional LSTM for temporal dynamics, and a Direction-Aware prediction head to handle the distinct yet coupled behavior of East, North, and Up displacement components. A kinematically-motivated **Direction Loss** enforces temporal smoothness, drift correction, and cross-component covariance consistency beyond standard regression objectives. Bayesian hyperparameter optimization via Optuna-TPE handles model tuning.

The framework supports the complete workflow from raw NGL tenv3 data ingestion and quality-controlled preprocessing, through spatiotemporal feature engineering and model training, to prediction evaluation, ablation analysis, and attention weight visualization.

---

## Key Features

- **Data Import**: Supports NGL (Nevada Geodetic Laboratory) tenv3 `.txt` format — the standard daily GNSS displacement product aligned to ITRF2020, covering East, North, and Up components with formal uncertainties
- **Preprocessing**: Iterative MAD-based jump detection and correction, IQR-based outlier removal anchored to the training split, controlled gap filling with rolling median baseline, and chronological train / validation / test splitting
- **Feature Engineering**: Multi-scale rolling statistics (7 / 30 / 60 / 365-day windows), first- and second-order difference features, linear trend slopes, sinusoidal seasonal encodings, and spatial cross-station correlation features via KNN graph construction
- **Spatial Graph Modeling**: KNN geodesic graph with haversine-distance inverse-weighting; stacked GATv2 layers with layer normalization and residual connections; support for arbitrary station networks
- **Temporal Modeling**: Bidirectional LSTM followed by a temporal multi-head self-attention module (applied to the BiLSTM output) for long-range dependency capture within a configurable historical input window; multi-step daily forecast output. (This self-attention layer is part of the implemented architecture — see the Architecture note in the Train section regarding its status in the manuscript.)
- **Direction-Aware Prediction Head**: Component-specific encoding pathways for E, N, and U with gated cross-direction interaction, preserving the distinct signal characteristics of horizontal versus vertical motion
- **Physics-Constrained Direction Loss**: Combines Directional Huber Loss, velocity smoothness penalty, temporal second-order smoothness, drift correction, and E/N/U covariance constraint with dynamic per-component weight adaptation
- **Bayesian Hyperparameter Optimization**: Optuna TPE sampler with cosine annealing warm restart scheduler
- **Accuracy Evaluation**: RMSE, MAE, R², and NRMSE computed independently per component (E / N / U) on a fully held-out chronological test set
- **Experimental Benchmarks**: Built-in scripts for loss function comparison (MSE / MAE / Huber / Direction Loss), component-level ablation study (Full / No-GAT / No-BiLSTM / No-DA), and multi-model comparison (VARIMA / GCN-LSTM / GAT-Transformer / DA-GAT-BiLSTM)
- **Attention Visualization**: Geographic map of learned inter-station GAT attention weights overlaid on tectonic fault structures (Cartopy + Natural Earth shapefiles + GeoJSON faults)

---

## Units

Two different units are used at two different stages of the pipeline, so please read this section carefully before interpreting any metric.

**During preprocessing (internal, always mm).** When a tenv3 file is parsed, displacements and their formal uncertainties are immediately converted to **millimeters**: `E_mm = (col7 + col8) × 1000`, and similarly for N, U, and the sigma columns. All quality-control thresholds in `1gnss_data_preprocessing.py` are expressed in this internal mm convention (for example `JUMP_MIN_MM = 5.0`, `JUMP_RESID_MIN_MM = 3.0`). This is fixed and is **not** affected by `DISPLACEMENT_UNIT`.

**When the preprocessed CSVs are written (controlled by `DISPLACEMENT_UNIT`).** Just before saving `train.csv` / `val.csv` / `test.csv`, the displacement columns (and `*_raw`, `*_sig`) are multiplied by a unit factor: `1e-3` when `DISPLACEMENT_UNIT = "m"` and `1.0` when `DISPLACEMENT_UNIT = "mm"`. **The code default is `"m"`, so by default the preprocessed CSVs are written in meters.**

**Feature engineering, training, prediction, and saved arrays operate in the CSV unit — meters by default.** `2gnss_feature_engineering.py` and all four model scripts read the `east` / `north` / `up` columns directly from these CSVs with no further unit conversion, and the target columns are not normalized. The model therefore learns, predicts, and stores displacements in **meters** under the default configuration. This is why the prediction-result columns described in the Evaluate section carry the `_m` suffix and the `test_predictions.npz` arrays are in meters.

**Reported error metrics are converted to mm at print/log time.** RMSE and MAE are first computed in the working unit (meters) and then multiplied by `1000` purely for display, e.g. `3gnss_model_training.py` prints `MAE = metrics['MAE'] * 1000` "mm" and `6Model_comparison.py` applies the same `* 1000  # Convert to mm`. R² is dimensionless and unaffected. This `× 1000` is the reason the paper reports residuals and RMSE/MAE in mm.

> **Important — keep `DISPLACEMENT_UNIT = "m"`.** The `× 1000` reporting factor is hard-coded on the assumption that the working unit is meters. If you set `DISPLACEMENT_UNIT = "mm"`, the model will train correctly in mm, but the reported RMSE/MAE will be inflated by a factor of 1000 unless you also remove the `* 1000` conversions in the training, loss-comparison, ablation, and model-comparison scripts. With the default `"m"`, preprocessing is internally in mm, the model and saved predictions are in meters, and all printed RMSE/MAE values are in mm — consistent with the manuscript.

---

## Installation

### Requirements

- Python 3.9 or later
- PyTorch 2.0 or later (CUDA recommended)
- torch-geometric
- numpy, pandas, scipy, scikit-learn
- matplotlib, cartopy
- optuna
- tqdm, statsmodels

### Steps

1. Clone this repository:

```bash
git clone https://github.com/SpaceGeodesyLab/DA-GAT-BiLSTM.git
cd DA-GAT-BiLSTM
```

2. Create and activate a virtual environment (recommended).

On Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bat
python -m venv venv
venv\Scripts\activate
```

3. Install PyTorch with CUDA support:

```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. Install torch-geometric:

```bash
pip install torch-geometric
```

5. Install remaining dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib cartopy optuna tqdm statsmodels
```

6. Verify the installation (checks imports and CUDA, does not start training):

```bash
python -c "import torch, torch_geometric; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```

---

## Quick Start

### 1. Load Data

DA-GAT-BiLSTM accepts daily GNSS displacement time series in **NGL tenv3 `.txt` format** from the Nevada Geodetic Laboratory (https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14/). Place all station `.txt` files in a single input directory. Each file must be named after the station 4-character code (e.g., `ELKO.txt`). A sample station file is provided at `sample_data/ELKO.txt`.

**Input file format (space-separated, NGL tenv3 — one row per day):**

| Column | Field | Description |
|--------|-------|-------------|
| 0 | `site` | 4-character station code (e.g., `ELKO`) |
| 1 | `YYMMMDD` | Date string (e.g., `94JAN23`) |
| 2 | `yyyy.yyyy` | Decimal year — primary time axis |
| 3 | `__MJD` | Modified Julian Date |
| 7–8 | `e0 + east(m)` | East displacement = reference + residual (m) |
| 9–10 | `n0 + north(m)` | North displacement (m) |
| 11–12 | `u0 + up(m)` | Vertical displacement (m) |
| 14–16 | `sig_e / sig_n / sig_u` | Formal uncertainties per component (m) |
| 20 | `_latitude(deg)` | Station geodetic latitude |
| 21 | `_longitude(deg)` | Station geodetic longitude |
| 22 | `__height(m)` | Station ellipsoidal height |

> **Note:** At ingestion each component is read in millimeters as `E_mm = (col7 + col8) × 1000` (similarly for N and U), and preprocessing runs internally in mm. The displacement columns are then written to the preprocessed CSVs in the unit chosen by `DISPLACEMENT_UNIT` — **meters by default**. See the Units section above for the full mm/m convention.

---

### 2. Preprocess

Edit the `CONFIG` dictionary at the top of `1gnss_data_preprocessing.py` to set your paths, then run:

```bash
python 1gnss_data_preprocessing.py
```

Key configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_DIR` | — | Directory containing `.txt` station files |
| `OUTPUT_DIR` | — | Root output directory for preprocessed data |
| `TRAIN_END` | `"2015-01-01"` | End date of the training split (exclusive) |
| `VAL_END` | `"2020-01-01"` | End date of the validation split (test = remainder) |
| `DISPLACEMENT_UNIT` | `"m"` | Unit of the displacement columns written to the preprocessed CSVs: `"m"` or `"mm"`. Preprocessing is always internally in mm; this only rescales the saved columns. The model consumes these CSVs directly, so this is also the model's working unit. Keep `"m"` (default) — reported RMSE/MAE assume meters and apply a hard-coded `× 1000` to display mm (see Units) |
| `JUMP_WIN` | `60` | Sliding-window half-width for jump detection (days) |
| `JUMP_K` | `3.5` | MAD multiplier threshold for jump detection |
| `JUMP_MIN_MM` | `5.0` | Minimum jump magnitude to correct (mm) |
| `JUMP_MAX_ITER` | `8` | Maximum correction iterations per component |
| `IQR_K` | `1.5` | IQR multiplier for outlier removal (Tukey's rule) |
| `MAX_SHORT_GAP` | `30` | Maximum gap length filled by time interpolation (days) |
| `TARGET_MAX_MISS` | `2.0` | Target missing-data rate after gap filling (%) |
| `CENTER_ON_TRAIN_MEAN` | `True` | Center each component on its training-period mean |

The preprocessing pipeline applies the following steps in sequence for each component (E, N, U):

1. **Jump detection and correction** — iterative MAD-based sliding-window method, repeated up to `JUMP_MAX_ITER` times, with an optional secondary pass on the baseline residuals
2. **Outlier removal** — IQR thresholds computed on the training split only, then applied to all splits
3. **Gap filling** — time interpolation for gaps ≤ `MAX_SHORT_GAP` days; rolling-median baseline infill for remaining gaps until missing rate ≤ `TARGET_MAX_MISS`%
4. **Mean centering** — displacement centered on the training-period mean (prevents data leakage)

**Output structure** (date ranges follow the `TRAIN_END` / `VAL_END` settings above):

```
preprocessed_data/
├── ELKO/
│   ├── train.csv          # start of record – 2014-12-31
│   ├── val.csv            # 2015-01-01 – 2019-12-31
│   ├── test.csv           # 2020-01-01 – end of record
│   └── ELKO_full.csv      # Complete processed record
└── quality_report.csv     # Per-station QC summary (jumps, outliers, missing %)
```

---

### 3. Build Features

Run the feature engineering pipeline on the preprocessed data:

```bash
python 2gnss_feature_engineering.py
```

The `TemporalFeatureBuilder` generates rolling statistics over 7 / 30 / 60 / 365-day windows (mean, std, skewness, kurtosis), first- and second-order differences, rolling linear trend slopes, and sinusoidal seasonal encodings (annual and semi-annual harmonics). The `SpatialFeatureBuilder` constructs a KNN proximity graph using scikit-learn's `BallTree` with the haversine metric, and computes spatially weighted interpolation and inter-station correlation features.

A timestamped log file is written to the output directory at runtime.

---

### 4. Train

Launch training with Bayesian hyperparameter optimization:

```bash
python 3gnss_model_training.py
```

**DA-GAT-BiLSTM architecture summary:**

| Component | Role |
|-----------|------|
| KNN spatial graph (k = 10, haversine weights) | Inter-station connectivity prior |
| GATv2 layers with residual + LayerNorm | Adaptive spatial message passing |
| Bidirectional LSTM | Forward + backward temporal encoding |
| Temporal Multi-Head Self-Attention (on BiLSTM output) | Long-range temporal dependency (implemented; see note below) |
| Direction-Aware head (E / N / U pathways) | Component-specific prediction |
| 7-step daily output per component | Short-term forecast horizon |

Optuna searches over learning rate, GAT hidden size, LSTM hidden size, dropout, and Direction Loss weights. The best configuration is saved automatically.

> **Architecture note (temporal self-attention).** The implemented model (`DAGATBiLSTM` in `3gnss_model_training.py`) applies a temporal multi-head self-attention module (`MultiHeadAttention`) to the BiLSTM output before the Direction-Aware head, i.e. the forward path is **GATv2 → BiLSTM → temporal self-attention → Direction-Aware head**. This README describes the code as implemented. The manuscript's architecture figure and Section 3 currently present the core GATv2 + BiLSTM + Direction-Aware blocks and do not depict this temporal self-attention layer (the only "self-attention" mentioned in the paper refers to the GAT-Transformer *baseline*). For full repo–manuscript agreement, this temporal self-attention module should be added to the paper (architecture figure and model description); alternatively, if the module is to be excluded from the published model, it should be removed from `DAGATBiLSTM.forward` and from this README. Until that is reconciled, treat the code as the authoritative description of what was trained.

**Output:**

```
model_output/
├── model_best.pt          # Best model checkpoint (lowest validation loss)
├── best_params.json       # Optimal hyperparameters found by Optuna
├── training_log_*.txt     # Epoch-level training and validation metrics
└── predictions/           # Per-station test-set prediction CSVs
```

---

### 5. Evaluate

#### Visualize per-station prediction results and lag diagnostics:

```bash
python station_prediction_lag.py \
    --csv  "path/to/prediction_result.csv" \
    --station  "ELKO" \
    --outdir  "./results" \
    --max_lag  10
```

The prediction result CSV (generated by the training script) must contain the following columns:

| Column | Description |
|--------|-------------|
| `time` | Decimal year (time axis) |
| `east_obs_m` | Observed East displacement (m) |
| `north_obs_m` | Observed North displacement (m) |
| `up_obs_m` | Observed Up displacement (m) |
| `east_pred_m` | Predicted East displacement (m) |
| `north_pred_m` | Predicted North displacement (m) |
| `up_pred_m` | Predicted Up displacement (m) |

> If you are working with prediction files generated by an earlier version that used localized (non-English) column names, rename the headers to the names above before running the evaluation script.

**Output files:**

| File | Description |
|------|-------------|
| `fig_station_prediction.png` | Three-panel time series of observed vs. predicted displacement, annotated with Pearson r, RMSE, and MAE per component |
| `fig_lag_diagnostic.png` | Normalized cross-correlation functions for E / N / U within a ±`max_lag`-day window — detects systematic prediction time lag |
| `lag_table.csv` | Peak lag (days) and peak CCF value per component |

---

## Experimental Benchmarks

### Loss Function Comparison

Trains the same DA-GAT-BiLSTM architecture under four loss functions and compares test-set performance:

```bash
python 4Loss_function_comparison.py
```

Loss functions compared: MSE, MAE, Huber, and Direction Loss.

### Ablation Study

Quantifies the contribution of each architectural component by systematically removing one module at a time:

```bash
python 5Ablation_study.py
```

| Variant | Components |
|---------|-----------|
| Full Model | GAT + BiLSTM + Direction-Aware head |
| No-GAT | BiLSTM + Direction-Aware head only |
| No-BiLSTM | GAT + Direction-Aware head only |
| No-DA | GAT + BiLSTM only |

### Model Comparison

Benchmarks DA-GAT-BiLSTM against three baselines:

```bash
python 6Model_comparison.py
```

| Model | Type | Description |
|-------|------|-------------|
| VARIMA | Statistical | Vector autoregression — cross-station dependencies |
| GCN-LSTM | Deep Learning | Graph convolutional network + LSTM baseline |
| GAT-Transformer | Deep Learning | Graph attention + Transformer encoder baseline |
| **DA-GAT-BiLSTM** | **Deep Learning** | **Proposed — direction-aware spatial-temporal model** |

---

## Attention Weight Visualization

Visualize learned inter-station attention weights on a geographic map and analyze correlation with tectonic fault structures:

```bash
python visualize_attention_weights.py
```

Before running, set the following paths at the top of the script:

| Variable | Description |
|----------|-------------|
| `CKPT_PATH` | Path to the trained `model.pt` checkpoint |
| `STATIONS_CSV` | CSV file with station codes, latitudes, and longitudes |
| `LOADER_PKL` | Pickled DataLoader object saved during training |
| `FAULT_GEOJSON` | GeoJSON file of fault traces (e.g., USGS San Andreas Fault) |
| `NE_DIR` | Local directory containing Natural Earth shapefiles |
| `TOP_PERCENTILE` | Percentage of top attention edges to display (default: 20) |

**Output files:**

| File | Description |
|------|-------------|
| `fig_attention_san_andreas.png` | Geographic map with top-percentile attention edges and fault traces |
| `attention_stats.csv` | Per-edge mean attention weight, inter-station distance, and fault proximity |
| `attention_vs_fault.csv` | Correlation between attention weight and distance to nearest fault trace |

---

## Output Directory Structure

```
project_root/
├── sample_data/
│   └── ELKO.txt                   # Sample NGL tenv3 station file
├── preprocessed_data/
│   ├── ELKO/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── ELKO_full.csv
│   └── quality_report.csv
├── features/
│   └── ELKO/
│       ├── train_features.csv
│       ├── val_features.csv
│       └── test_features.csv
├── model_output/
│   ├── model_best.pt
│   ├── best_params.json
│   ├── training_log_*.txt
│   └── predictions/
│       └── ELKO_predictions.csv
└── results/
    ├── fig_station_prediction.png
    ├── fig_lag_diagnostic.png
    └── lag_table.csv
```

---

## Citation

If you use this code or the DA-GAT-BiLSTM framework in your work, please cite the associated paper:

```bibtex
@article{zhu_dagatbilstm,
  title   = {DA-GAT-BiLSTM: Direction-Aware Graph Attention Bidirectional LSTM for GNSS Displacement Time Series Prediction},
  author  = {Zhu, Siyu and He, Xiaoxing and Huang, Zhengkai and others},
  journal = {TBD},
  year    = {TBD},
  note    = {Update with final volume, pages, and DOI upon acceptance.}
}
```

---

## License

This software is released for academic and research use. If you use this code or the DA-GAT-BiLSTM framework in your work, please cite the paper listed in the Citation section above.


## SpaceGeodesyLab of Jiangxi University of Science and Technology/江西理工大学时空智能与对地观测团队