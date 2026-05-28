# -*- coding: utf-8 -*-
"""
Loss Function Comparison Experiment - DirectionLoss vs MSE
"""
import sys
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import RobustScaler
import pickle
from typing import Dict
from collections import defaultdict
from sklearn.neighbors import BallTree
from tqdm import tqdm

print("=" * 80)
print("Loss Function Training Comparison")
print("=" * 80)


# ==================== Data Normalizer ====================
class GNSSDataNormalizer:
    """GNSS Data Normalizer"""

    def __init__(self, method='robust'):
        self.method = method
        self.scaler_X = None
        self.scaler_y = None

    def fit(self, X_train, y_train):
        """Fit normalization parameters"""
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

        B, N, W, F = X_train.shape
        X_flat = X_train.reshape(-1, F)
        self.scaler_X.fit(X_flat)

        B, N, H, D = y_train.shape
        y_flat = y_train.reshape(-1, D)
        self.scaler_y.fit(y_flat)

        return self

    def transform_X(self, X):
        """Normalize features"""
        shape = X.shape
        X_flat = X.reshape(-1, shape[-1])
        X_scaled = self.scaler_X.transform(X_flat)
        return X_scaled.reshape(shape).astype(np.float32)

    def transform_y(self, y):
        """Normalize targets"""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        y_scaled = self.scaler_y.transform(y_flat)
        return y_scaled.reshape(shape).astype(np.float32)

    def inverse_transform_y(self, y_scaled):
        """Inverse normalize targets"""
        shape = y_scaled.shape
        y_flat = y_scaled.reshape(-1, shape[-1])
        y_original = self.scaler_y.inverse_transform(y_flat)
        return y_original.reshape(shape)

    def save(self, filepath):
        """Save normalizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'method': self.method
            }, f)

    def load(self, filepath):
        """Load normalizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler_X = data['scaler_X']
            self.scaler_y = data['scaler_y']
            self.method = data['method']
        return self


# ==================== Dataset Class ====================
class GNSSGraphDataset(Dataset):
    def __init__(self, feature_dir, split='train', window_size=30, pred_horizon=7,
                 k_neighbors=10, distance_threshold=1000.0, normalizer=None):
        self.feature_dir = feature_dir
        self.split = split
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.normalizer = normalizer

        print(f"\nLoading {split} dataset...")
        self.stations, self.data_dict, self.station_coords = self._load_all_stations()

        sample_df = list(self.data_dict.values())[0]
        exclude_cols = ['station', 'decimal_year', 'mjd', 'east', 'north', 'up']
        self.feature_names = [c for c in sample_df.columns if c not in exclude_cols]

        print(f"Building spatial graph...")
        self.edge_index, self.edge_weight = self._build_spatial_graph()

        print(f"Generating samples...")
        self.samples = self._generate_samples()

        print(f"Data loading complete: {len(self.samples)} samples\n")

        # Preprocess and cache all data to numpy arrays
        coords_df = self.station_coords.set_index('station')
        coords_np = []
        for s in self.stations:
            row = coords_df.loc[s]
            coords_np.append([row['latitude'], row['longitude'], row['height']])
        self.coords_np = np.asarray(coords_np, dtype=np.float32)

        self.feat_np = {}
        self.tgt_np = {}
        self.length_np = {}
        for s in self.stations:
            df = self.data_dict[s]
            feat = df[self.feature_names].to_numpy(dtype=np.float32, copy=True)
            tgt = df[['east', 'north', 'up']].to_numpy(dtype=np.float32, copy=True)

            # Preprocess NaN
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

            self.feat_np[s] = feat
            self.tgt_np[s] = tgt
            self.length_np[s] = feat.shape[0]

        self.num_nodes = len(self.stations)
        self.num_features = len(self.feature_names)

    def _load_all_stations(self):
        stations = sorted([d for d in os.listdir(self.feature_dir)
                           if os.path.isdir(os.path.join(self.feature_dir, d))])

        data_dict = {}
        coords_list = []

        for station in stations:
            file_path = os.path.join(self.feature_dir, station, f'{self.split}_featured.csv')
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)
            if len(df) < self.window_size + self.pred_horizon:
                continue

            data_dict[station] = df

            if 'latitude' in df.columns and 'longitude' in df.columns:
                coords_list.append({
                    'station': station,
                    'latitude': df['latitude'].iloc[0],
                    'longitude': df['longitude'].iloc[0],
                    'height': df.get('height', pd.Series([0])).iloc[0]
                })

        station_coords = pd.DataFrame(coords_list)
        return sorted(data_dict.keys()), data_dict, station_coords

    def _build_spatial_graph(self):
        coords = self.station_coords[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        tree = BallTree(coords_rad, metric='haversine')

        distances, indices = tree.query(coords_rad, k=self.k_neighbors + 1)
        R = 6371.0

        edge_list = []
        edge_weights = []

        for i in range(len(self.stations)):
            for j_idx in range(1, len(indices[i])):
                j = indices[i][j_idx]
                dist_km = distances[i][j_idx] * R

                if dist_km <= self.distance_threshold:
                    edge_list.append([i, j])
                    edge_weights.append(1.0 / (dist_km + 1.0))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_weight = edge_weight / edge_weight.sum()

        return edge_index, edge_weight

    def _generate_samples(self):
        min_n = min(len(df) for df in self.data_dict.values())
        max_start = min_n - self.window_size - self.pred_horizon

        if max_start < 0:
            return []

        return [{'start_idx': i} for i in range(max_start + 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        W = self.window_size
        H = self.pred_horizon
        N = self.num_nodes
        feat_dim = self.num_features

        x_np = np.zeros((N, W, feat_dim), dtype=np.float32)
        y_np = np.zeros((N, H, 3), dtype=np.float32)

        start_i_global = sample['start_idx']

        for node_i, station in enumerate(self.stations):
            feat = self.feat_np[station]
            tgt = self.tgt_np[station]
            n = self.length_np[station]

            max_start = n - W - H
            start_idx = min(start_i_global, max_start) if max_start >= 0 else 0
            end_idx = start_idx + W
            target_end = end_idx + H

            if target_end <= n:
                x_np[node_i, :, :] = feat[start_idx:end_idx, :]
                y_np[node_i, :, :] = tgt[end_idx:target_end, :]

        # Apply normalization
        if self.normalizer is not None:
            x_np = self.normalizer.transform_X(x_np)
            y_np = self.normalizer.transform_y(y_np)

        x = torch.from_numpy(np.ascontiguousarray(x_np))
        y = torch.from_numpy(np.ascontiguousarray(y_np))
        node_coords = torch.from_numpy(self.coords_np.copy())

        return {
            'x': x,
            'y': y,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'node_coords': node_coords
        }


# ==================== Direction-Aware Module ====================
class DirectionAwareModule(nn.Module):
    def __init__(self, hidden_dim=256, east_dim=128, north_dim=128, up_dim=64):
        super().__init__()

        self.east_encoder = nn.Sequential(
            nn.Linear(hidden_dim, east_dim),
            nn.SiLU(),
            nn.LayerNorm(east_dim)
        )

        self.north_encoder = nn.Sequential(
            nn.Linear(hidden_dim, north_dim),
            nn.SiLU(),
            nn.LayerNorm(north_dim)
        )

        self.up_encoder = nn.Sequential(
            nn.Linear(hidden_dim, up_dim),
            nn.SiLU(),
            nn.LayerNorm(up_dim)
        )

        self.gate_e_n = nn.Linear(north_dim * 2, north_dim)
        self.gate_e_u = nn.Linear(up_dim * 2, up_dim)
        self.gate_n_e = nn.Linear(east_dim * 2, east_dim)
        self.gate_n_u = nn.Linear(up_dim * 2, up_dim)
        self.gate_u_e = nn.Linear(east_dim * 2, east_dim)
        self.gate_u_n = nn.Linear(north_dim * 2, north_dim)

        self.V_en = nn.Linear(north_dim, east_dim)
        self.V_eu = nn.Linear(up_dim, east_dim)
        self.V_ne = nn.Linear(east_dim, north_dim)
        self.V_nu = nn.Linear(up_dim, north_dim)
        self.V_ue = nn.Linear(east_dim, up_dim)
        self.V_un = nn.Linear(north_dim, up_dim)

    def forward(self, h, h_prev=None):
        h_e = self.east_encoder(h)
        h_n = self.north_encoder(h)
        h_u = self.up_encoder(h)

        if h_prev is None:
            return h_e, h_n, h_u

        h_e_prev = self.east_encoder(h_prev)
        h_n_prev = self.north_encoder(h_prev)
        h_u_prev = self.up_encoder(h_prev)

        g_en = torch.sigmoid(self.gate_e_n(torch.cat([h_n, h_n_prev], dim=-1)))
        g_eu = torch.sigmoid(self.gate_e_u(torch.cat([h_u, h_u_prev], dim=-1)))
        g_ne = torch.sigmoid(self.gate_n_e(torch.cat([h_e, h_e_prev], dim=-1)))
        g_nu = torch.sigmoid(self.gate_n_u(torch.cat([h_u, h_u_prev], dim=-1)))
        g_ue = torch.sigmoid(self.gate_u_e(torch.cat([h_e, h_e_prev], dim=-1)))
        g_un = torch.sigmoid(self.gate_u_n(torch.cat([h_n, h_n_prev], dim=-1)))

        h_e_enhanced = h_e + g_en * self.V_en(h_n) + g_eu * self.V_eu(h_u)
        h_n_enhanced = h_n + g_ne * self.V_ne(h_e) + g_nu * self.V_nu(h_u)
        h_u_enhanced = h_u + g_ue * self.V_ue(h_e) + g_un * self.V_un(h_n)

        return h_e_enhanced, h_n_enhanced, h_u_enhanced


# ==================== DA-GAT-BiLSTM Model ====================
class DAGATBiLSTM(nn.Module):
    def __init__(self, num_features, base_factor=64, gat_hidden=256, lstm_hidden=512,
                 num_gat_layers=3, num_heads=4, dropout=0.25, pred_horizon=7):
        super().__init__()

        self.gat_hidden = gat_hidden
        self.lstm_hidden = lstm_hidden
        self.pred_horizon = pred_horizon

        self.input_proj = nn.Linear(num_features, base_factor)
        self.input_norm = nn.LayerNorm(base_factor)

        # GAT layers
        self.gat_convs = nn.ModuleList([
            GATv2Conv(
                base_factor if i == 0 else gat_hidden,
                gat_hidden // num_heads,
                heads=num_heads,
                concat=True,
                edge_dim=1,
                dropout=dropout
            )
            for i in range(num_gat_layers)
        ])

        self.gat_norms = nn.ModuleList([nn.LayerNorm(gat_hidden) for _ in range(num_gat_layers)])

        self.lstm = nn.LSTM(gat_hidden, lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.direction_module = DirectionAwareModule(
            hidden_dim=lstm_hidden * 2,
            east_dim=128, north_dim=128, up_dim=64
        )

        self.east_head = nn.Linear(128, pred_horizon)
        self.north_head = nn.Linear(128, pred_horizon)
        self.up_head = nn.Linear(64, pred_horizon)

    def forward(self, x, edge_index, edge_weight):
        B, N, W, feat_dim = x.shape

        x_reshaped = x.view(B * N, W, feat_dim)
        x_proj = self.input_proj(x_reshaped)
        x_proj = self.input_norm(x_proj)
        x_graph = x_proj.view(B, N, W, -1)

        edge_attr = edge_weight.view(-1, 1)

        h_gat_all = []
        for t in range(W):
            x_t = x_graph[:, :, t, :]
            h = x_t.view(B * N, -1)

            for i, (conv, norm) in enumerate(zip(self.gat_convs, self.gat_norms)):
                h_new = conv(h, edge_index, edge_attr)
                h_new = norm(h_new)
                h = F.silu(h_new) + (h if h.shape[-1] == h_new.shape[-1] else 0)

            h_gat_all.append(h.view(B, N, -1))

        h_gat_seq = torch.stack(h_gat_all, dim=2)
        h_gat_flat = h_gat_seq.view(B * N, W, self.gat_hidden)

        lstm_out, _ = self.lstm(h_gat_flat)
        h_lstm = lstm_out[:, -1, :]
        h_lstm_reshaped = h_lstm.view(B, N, self.lstm_hidden * 2)

        h_e, h_n, h_u = self.direction_module(h_lstm_reshaped)

        pred_e = self.east_head(h_e)
        pred_n = self.north_head(h_n)
        pred_u = self.up_head(h_u)

        return pred_e, pred_n, pred_u


# ==================== Loss Functions ====================
class MSELoss(nn.Module):
    def forward(self, pred_e, pred_n, pred_u, target_e, target_n, target_u):
        loss_e = F.mse_loss(pred_e, target_e)
        loss_n = F.mse_loss(pred_n, target_n)
        loss_u = F.mse_loss(pred_u, target_u)
        return (loss_e + loss_n + loss_u) / 3.0


class MAELoss(nn.Module):
    def forward(self, pred_e, pred_n, pred_u, target_e, target_n, target_u):
        loss_e = F.l1_loss(pred_e, target_e)
        loss_n = F.l1_loss(pred_n, target_n)
        loss_u = F.l1_loss(pred_u, target_u)
        return (loss_e + loss_n + loss_u) / 3.0


class HuberLossWrapper(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred_e, pred_n, pred_u, target_e, target_n, target_u):
        loss_e = F.huber_loss(pred_e, target_e, delta=self.delta)
        loss_n = F.huber_loss(pred_n, target_n, delta=self.delta)
        loss_u = F.huber_loss(pred_u, target_u, delta=self.delta)
        return (loss_e + loss_n + loss_u) / 3.0


class DirectionLoss(nn.Module):
    """Direction-Aware Loss Function"""

    def __init__(self, beta=0.1, gamma=0.05, delta_smooth=0.01, device='cuda'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta_smooth = delta_smooth
        self.delta = 1.0
        self.device = device

        # Dynamic weight mechanism
        self.z_east = nn.Parameter(torch.tensor(0.0))
        self.z_north = nn.Parameter(torch.tensor(0.0))
        self.z_up = nn.Parameter(torch.tensor(0.0))

        self.eta = {'east': 1.2, 'north': 1.1, 'up': 1.0}

        self.register_buffer('mae_east', torch.tensor(1.0))
        self.register_buffer('mae_north', torch.tensor(1.0))
        self.register_buffer('mae_up', torch.tensor(1.0))

    def dynamic_weights(self):
        """Dynamic weight calculation"""
        z_tensor = torch.stack([self.z_east, self.z_north, self.z_up])
        softmax_weights = F.softmax(z_tensor, dim=0)

        epsilon_avg = (self.mae_east + self.mae_north + self.mae_up) / 3.0

        w_east = softmax_weights[0] * (1.0 / (self.mae_east + epsilon_avg + 1e-6)) * self.eta['east']
        w_north = softmax_weights[1] * (1.0 / (self.mae_north + epsilon_avg + 1e-6)) * self.eta['north']
        w_up = softmax_weights[2] * (1.0 / (self.mae_up + epsilon_avg + 1e-6)) * self.eta['up']

        total = w_east + w_north + w_up
        return {
            'east': w_east / total,
            'north': w_north / total,
            'up': w_up / total
        }

    def update_mae(self, pred_e, pred_n, pred_u, target_e, target_n, target_u, momentum=0.9):
        """Update MAE"""
        with torch.no_grad():
            mae_e = torch.abs(pred_e - target_e).mean()
            mae_n = torch.abs(pred_n - target_n).mean()
            mae_u = torch.abs(pred_u - target_u).mean()

            self.mae_east = momentum * self.mae_east + (1 - momentum) * mae_e
            self.mae_north = momentum * self.mae_north + (1 - momentum) * mae_n
            self.mae_up = momentum * self.mae_up + (1 - momentum) * mae_u

    def forward(self, pred_e, pred_n, pred_u, target_e, target_n, target_u):
        # Main loss - using Huber for improved robustness
        def huber_loss(pred, target):
            error = pred - target
            abs_error = torch.abs(error)
            delta_tensor = torch.tensor(self.delta, dtype=pred.dtype, device=pred.device)
            return torch.where(
                abs_error <= delta_tensor,
                0.5 * error ** 2,
                delta_tensor * (abs_error - 0.5 * delta_tensor)
            ).mean()

        huber_e = huber_loss(pred_e, target_e)
        huber_n = huber_loss(pred_n, target_n)
        huber_u = huber_loss(pred_u, target_u)

        # Dynamic weights
        weights = self.dynamic_weights()
        huber_total = weights['east'] * huber_e + weights['north'] * huber_n + weights['up'] * huber_u

        # Drift penalty
        e_east = pred_e - target_e
        e_north = pred_n - target_n
        e_up = pred_u - target_u

        drift_e = torch.abs(e_east.mean(dim=2))
        drift_n = torch.abs(e_north.mean(dim=2))
        drift_u = torch.abs(e_up.mean(dim=2))
        drift = (drift_e + drift_n + drift_u).mean() / 3.0

        # Covariance constraint
        cov_en = torch.abs((e_east * e_north).mean())
        cov_eu = torch.abs((e_east * e_up).mean())
        cov_nu = torch.abs((e_north * e_up).mean())
        cov = cov_en + cov_eu + cov_nu

        # Smoothness regularization
        def second_diff(x):
            if x.size(2) < 3:
                return torch.tensor(0.0, device=x.device)
            return x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]

        smooth_e = (second_diff(pred_e) ** 2).mean()
        smooth_n = (second_diff(pred_n) ** 2).mean()
        smooth_u = (second_diff(pred_u) ** 2).mean()
        smooth = (smooth_e + smooth_n + smooth_u) / 3.0

        total_loss = huber_total + self.beta * drift + self.gamma * cov + self.delta_smooth * smooth

        # Update MAE
        self.update_mae(pred_e, pred_n, pred_u, target_e, target_n, target_u)

        return total_loss


# ==================== Evaluation Metrics ====================
def compute_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    mask = ~(np.isnan(pred) | np.isnan(target))
    pred = pred[mask]
    target = target[mask]

    if len(pred) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    mae = np.abs(pred - target).mean()
    rmse = np.sqrt(((pred - target) ** 2).mean())

    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ==================== Training and Evaluation ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, normalizer, epochs=200, use_amp=True, patience=30, min_delta=1e-5):
    """Training function - Mixed precision training + Learning rate scheduling"""
    model.to(device)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    history = {'train_loss': [], 'val_loss': []}

    print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Training]',
                    leave=False, dynamic_ncols=True, ascii=True)

        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            edge_index = batch['edge_index'][0].to(device)
            edge_weight = batch['edge_weight'][0].to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)
                    target_e = y[:, :, :, 0]
                    target_n = y[:, :, :, 1]
                    target_u = y[:, :, :, 2]
                    loss = criterion(pred_e, pred_n, pred_u, target_e, target_n, target_u)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)
                target_e = y[:, :, :, 0]
                target_n = y[:, :, :, 1]
                target_u = y[:, :, :, 2]
                loss = criterion(pred_e, pred_n, pred_u, target_e, target_n, target_u)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Update learning rate (OneCycleLR updates per batch)
            if scheduler is not None:
                scheduler.step()

        pbar.close()

        # Validation phase
        model.eval()
        val_losses = []

        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]',
                        leave=False, dynamic_ncols=True, ascii=True)

        with torch.no_grad():
            for batch in pbar_val:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                edge_index = batch['edge_index'][0].to(device)
                edge_weight = batch['edge_weight'][0].to(device)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)
                        target_e = y[:, :, :, 0]
                        target_n = y[:, :, :, 1]
                        target_u = y[:, :, :, 2]
                        loss = criterion(pred_e, pred_n, pred_u, target_e, target_n, target_u)
                else:
                    pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)
                    target_e = y[:, :, :, 0]
                    target_n = y[:, :, :, 1]
                    target_u = y[:, :, :, 2]
                    loss = criterion(pred_e, pred_n, pred_u, target_e, target_n, target_u)

                val_losses.append(loss.item())
                pbar_val.set_postfix({'val_loss': f'{loss.item():.6f}'})

        pbar_val.close()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping logic
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }
            improvement_marker = "New best"
        else:
            patience_counter += 1
            improvement_marker = f"({patience_counter}/{patience})"

        if (epoch + 1) % 10 == 0 or patience_counter > 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | LR: {current_lr:.6f} | {improvement_marker}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! Validation loss did not improve for {patience} epochs")
            print(f"Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state['model_state_dict'])
                print("Restored best model parameters")
            break

    if patience_counter < patience:
        print(f"\nTraining complete! Total {epoch + 1} epochs")
        print(f"Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")

    return history, best_val_loss


def evaluate_model(model, test_loader, device, normalizer, return_raw=False):
    """Evaluate model and optionally return raw prediction data"""
    model.eval()

    all_pred_e, all_pred_n, all_pred_u = [], [], []
    all_target_e, all_target_n, all_target_u = [], [], []

    pbar = tqdm(test_loader, desc='Test Set Evaluation', leave=False, dynamic_ncols=True, ascii=True)

    with torch.no_grad():
        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            edge_index = batch['edge_index'][0].to(device)
            edge_weight = batch['edge_weight'][0].to(device)

            pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)

            # Move to CPU
            pred_e_cpu = pred_e.cpu().numpy()
            pred_n_cpu = pred_n.cpu().numpy()
            pred_u_cpu = pred_u.cpu().numpy()

            target_e_cpu = y[:, :, :, 0].cpu().numpy()
            target_n_cpu = y[:, :, :, 1].cpu().numpy()
            target_u_cpu = y[:, :, :, 2].cpu().numpy()

            # Inverse normalization
            if normalizer is not None:
                B, N, H = pred_e_cpu.shape
                pred_concat = np.stack([pred_e_cpu, pred_n_cpu, pred_u_cpu], axis=-1)
                target_concat = np.stack([target_e_cpu, target_n_cpu, target_u_cpu], axis=-1)

                pred_concat = normalizer.inverse_transform_y(pred_concat)
                target_concat = normalizer.inverse_transform_y(target_concat)

                pred_e_cpu = pred_concat[:, :, :, 0]
                pred_n_cpu = pred_concat[:, :, :, 1]
                pred_u_cpu = pred_concat[:, :, :, 2]

                target_e_cpu = target_concat[:, :, :, 0]
                target_n_cpu = target_concat[:, :, :, 1]
                target_u_cpu = target_concat[:, :, :, 2]

            all_pred_e.append(pred_e_cpu)
            all_pred_n.append(pred_n_cpu)
            all_pred_u.append(pred_u_cpu)

            all_target_e.append(target_e_cpu)
            all_target_n.append(target_n_cpu)
            all_target_u.append(target_u_cpu)

    pbar.close()

    pred_e = np.concatenate(all_pred_e, axis=0)
    pred_n = np.concatenate(all_pred_n, axis=0)
    pred_u = np.concatenate(all_pred_u, axis=0)

    target_e = np.concatenate(all_target_e, axis=0)
    target_n = np.concatenate(all_target_n, axis=0)
    target_u = np.concatenate(all_target_u, axis=0)

    metrics_east = compute_metrics(pred_e, target_e)
    metrics_north = compute_metrics(pred_n, target_n)
    metrics_up = compute_metrics(pred_u, target_u)

    metrics = {
        'East': metrics_east,
        'North': metrics_north,
        'Up': metrics_up
    }

    if return_raw:
        raw_data = {
            'predictions': {
                'East': pred_e,
                'North': pred_n,
                'Up': pred_u
            },
            'targets': {
                'East': target_e,
                'North': target_n,
                'Up': target_u
            }
        }
        return metrics, raw_data
    else:
        return metrics


# ==================== Main Experiment ====================
def main():
    print("\nConfiguring experiment parameters...")

    feature_dir = r"C:\PycharmProjects\pythonProject\GNSStimeseriesprediction\features"
    output_dir = r"C:\PycharmProjects\pythonProject\GNSStimeseriesprediction\loss_comparison_results_fixed"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 16
    num_workers = 4
    epochs = 200
    patience = 30

    print("\n" + "=" * 80)
    print("Step 1: Load dataset and fit normalizer")
    print("=" * 80)

    # Load small amount of data to fit normalizer
    train_dataset_temp = GNSSGraphDataset(feature_dir, split='train', window_size=30,
                                          pred_horizon=7, normalizer=None)

    print("\nFitting data normalizer...")
    sample_X = []
    sample_y = []
    for i in range(min(100, len(train_dataset_temp))):
        sample = train_dataset_temp[i]
        sample_X.append(sample['x'].numpy())
        sample_y.append(sample['y'].numpy())

    sample_X = np.array(sample_X)
    sample_y = np.array(sample_y)

    normalizer = GNSSDataNormalizer(method='robust')
    normalizer.fit(sample_X, sample_y)
    normalizer.save(os.path.join(output_dir, 'normalizer.pkl'))

    # Reload dataset (with normalization)
    print("\nReloading dataset (applying normalization)...")
    train_dataset = GNSSGraphDataset(feature_dir, split='train', window_size=30,
                                     pred_horizon=7, normalizer=normalizer)
    val_dataset = GNSSGraphDataset(feature_dir, split='val', window_size=30,
                                   pred_horizon=7, normalizer=normalizer)
    test_dataset = GNSSGraphDataset(feature_dir, split='test', window_size=30,
                                    pred_horizon=7, normalizer=normalizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    num_features = train_dataset.num_features
    print(f"Feature dimensions: {num_features}")

    loss_functions = {
        'MSE': MSELoss,
        'MAE': MAELoss,
        'Huber': HuberLossWrapper,
        'DirectionLoss': DirectionLoss
    }

    results = {}

    print("\n" + "=" * 80)
    print("Step 2: Start loss function comparison experiment")
    print("=" * 80)

    for loss_name, LossClass in loss_functions.items():
        print(f"\n{'=' * 80}")
        print(f"Using loss function: {loss_name}")
        print(f"{'=' * 80}")

        model = DAGATBiLSTM(num_features=num_features)

        if loss_name == 'DirectionLoss':
            criterion = LossClass(device=device)  # DirectionLoss requires device parameter
        else:
            criterion = LossClass()

        # Using OneCycleLR scheduler
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.003,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

        print(f"Start training (Loss function: {loss_name})...")
        history, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, normalizer, epochs=epochs, use_amp=True if device.type == 'cuda' else False,
            patience=patience, min_delta=1e-5
        )

        print(f"\nEvaluating (Loss function: {loss_name})...")
        test_metrics, raw_data = evaluate_model(model, test_loader, device, normalizer, return_raw=True)

        results[loss_name] = {
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'history': history,
            'raw_predictions': raw_data
        }

        print(f"\n{loss_name} Test Set Results:")
        print(
            f"  East - MAE: {test_metrics['East']['MAE']:.6f}, RMSE: {test_metrics['East']['RMSE']:.6f}, R2: {test_metrics['East']['R2']:.4f}")
        print(
            f"  North - MAE: {test_metrics['North']['MAE']:.6f}, RMSE: {test_metrics['North']['RMSE']:.6f}, R2: {test_metrics['North']['R2']:.4f}")
        print(
            f"  Up - MAE: {test_metrics['Up']['MAE']:.6f}, RMSE: {test_metrics['Up']['RMSE']:.6f}, R2: {test_metrics['Up']['R2']:.4f}")

    print("\n" + "=" * 80)
    print("Step 3: Save results")
    print("=" * 80)

    with open(os.path.join(output_dir, 'loss_comparison_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    summary_file = os.path.join(output_dir, 'loss_comparison_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Loss Function Comparison Experiment Summary\n")
        f.write("=" * 80 + "\n\n")

        for loss_name in loss_functions.keys():
            f.write(f"\n{loss_name}:\n")
            f.write(f"  Validation Loss: {results[loss_name]['best_val_loss']:.6f}\n")
            metrics = results[loss_name]['test_metrics']
            f.write(
                f"  East - MAE: {metrics['East']['MAE']:.6f}, RMSE: {metrics['East']['RMSE']:.6f}, R2: {metrics['East']['R2']:.4f}\n")
            f.write(
                f"  North - MAE: {metrics['North']['MAE']:.6f}, RMSE: {metrics['North']['RMSE']:.6f}, R2: {metrics['North']['R2']:.4f}\n")
            f.write(
                f"  Up - MAE: {metrics['Up']['MAE']:.6f}, RMSE: {metrics['Up']['RMSE']:.6f}, R2: {metrics['Up']['R2']:.4f}\n")

    print(f"Results saved to: {output_dir}")

    # Create and save visualization data
    print("\n" + "=" * 80)
    print("Step 4: Prepare visualization data")
    print("=" * 80)

    first_station = test_dataset.stations[0]
    station_df = test_dataset.data_dict[first_station]
    num_samples = len(test_dataset.samples)
    num_nodes = test_dataset.num_nodes
    pred_horizon = test_dataset.pred_horizon

    print(f"  Test set samples: {num_samples}")
    print(f"  Number of stations: {num_nodes}")
    print(f"  Prediction horizon: {pred_horizon}")

    visualization_data = {
        'meta': {
            'num_samples': num_samples,
            'num_nodes': num_nodes,
            'pred_horizon': pred_horizon,
            'stations': test_dataset.stations,
            'description': 'Averaged predictions across all nodes and samples'
        }
    }

    for direction in ['East', 'North', 'Up']:
        true_vals = results['MSE']['raw_predictions']['targets'][direction]
        true_avg = true_vals.mean(axis=(0, 1))
        visualization_data[direction] = {'true': true_avg}

        for loss_name in loss_functions.keys():
            pred_vals = results[loss_name]['raw_predictions']['predictions'][direction]
            pred_avg = pred_vals.mean(axis=(0, 1))
            visualization_data[direction][loss_name] = pred_avg

    if 'decimal_year' in station_df.columns:
        visualization_data['time'] = station_df['decimal_year'].values[-pred_horizon:]
    else:
        visualization_data['time'] = np.arange(pred_horizon)

    vis_data_file = os.path.join(output_dir, 'visualization_data.pkl')
    with open(vis_data_file, 'wb') as f:
        pickle.dump(visualization_data, f)

    print(f"  Visualization data saved to: {vis_data_file}")
    print(f"  Time series length: {len(visualization_data['time'])}")
    print(f"  Included loss functions: {list(loss_functions.keys())}")

    print("\nLoss function comparison experiment complete!")

    # Compare DirectionLoss and MSE
    print("\n" + "=" * 80)
    print("DirectionLoss vs MSE Performance Comparison:")
    print("=" * 80)

    mse_metrics = results['MSE']['test_metrics']
    dir_metrics = results['DirectionLoss']['test_metrics']

    for direction in ['East', 'North', 'Up']:
        print(f"\n{direction} direction:")
        for metric in ['MAE', 'RMSE', 'R2']:
            mse_val = mse_metrics[direction][metric]
            dir_val = dir_metrics[direction][metric]

            if metric == 'R2':
                improvement = ((dir_val - mse_val) / abs(mse_val)) * 100
                print(f"  {metric}: MSE={mse_val:.4f}, DirectionLoss={dir_val:.4f} (Improvement: {improvement:+.2f}%)")
            else:
                improvement = ((mse_val - dir_val) / mse_val) * 100
                print(f"  {metric}: MSE={mse_val:.6f}, DirectionLoss={dir_val:.6f} (Improvement: {improvement:+.2f}%)")


if __name__ == '__main__':
    main()