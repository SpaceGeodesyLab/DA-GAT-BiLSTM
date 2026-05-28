# -*- coding: utf-8 -*-
"""
DA-GAT-BiLSTM Model Ablation Study
"""
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
import time
import pickle
from sklearn.neighbors import BallTree

print("=" * 80)
print("Ablation Study - Model Component Importance Analysis")
print("=" * 80)


# ==================== Dataset Class ====================
class GNSSGraphDataset(Dataset):
    def __init__(self, feature_dir, split='train', window_size=30, pred_horizon=7,
                 k_neighbors=10, distance_threshold=1000.0, cache_data=True):
        self.feature_dir = feature_dir
        self.split = split
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.cache_data = cache_data

        print(f"\nLoading {split} dataset...")
        self.stations, self.data_dict, self.station_coords = self._load_all_stations()

        sample_df = list(self.data_dict.values())[0]
        exclude_cols = ['station', 'decimal_year', 'mjd', 'east', 'north', 'up']
        self.feature_names = [c for c in sample_df.columns if c not in exclude_cols]

        print(f"Building spatial graph...")
        self.edge_index, self.edge_weight = self._build_spatial_graph()

        print(f"Preprocessing data...")
        self._preprocess_data()

        print(f"Generating sample indices...")
        self.samples = self._generate_samples()

        print(f"Data loading complete: {len(self.samples)} samples\n")

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

    def _preprocess_data(self):
        """Preprocess and cache all data to numpy arrays"""
        coords_df = self.station_coords.set_index('station')
        coords_np = []
        for s in self.stations:
            row = coords_df.loc[s]
            coords_np.append([row['latitude'], row['longitude'], row['height']])
        self.coords_np = np.asarray(coords_np, dtype=np.float32)

        # Preprocess all station data
        self.feat_np = {}
        self.tgt_np = {}
        self.length_np = {}

        for s in self.stations:
            df = self.data_dict[s]
            feat = df[self.feature_names].to_numpy(dtype=np.float32, copy=True)
            tgt = df[['east', 'north', 'up']].to_numpy(dtype=np.float32, copy=True)

            # Handle NaN values
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

            self.feat_np[s] = feat
            self.tgt_np[s] = tgt
            self.length_np[s] = feat.shape[0]

        self.num_nodes = len(self.stations)
        self.num_features = len(self.feature_names)

        # If caching enabled, pre-generate all samples
        if self.cache_data:
            print("Preloading all samples into memory...")
            self.cached_samples = {}

    def _generate_samples(self):
        min_n = min(len(df) for df in self.data_dict.values())
        max_start = min_n - self.window_size - self.pred_horizon

        if max_start < 0:
            return []

        return [{'start_idx': i} for i in range(max_start + 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Check cache
        if self.cache_data and idx in self.cached_samples:
            return self.cached_samples[idx]

        sample = self.samples[idx]
        W = self.window_size
        H = self.pred_horizon
        N = self.num_nodes
        F = self.num_features

        # Pre-allocate arrays
        x_np = np.zeros((N, W, F), dtype=np.float32)
        y_np = np.zeros((N, H, 3), dtype=np.float32)

        start_i_global = sample['start_idx']

        # Batch fill data
        for node_i, station in enumerate(self.stations):
            feat = self.feat_np[station]
            tgt = self.tgt_np[station]
            n = self.length_np[station]

            max_start = n - W - H
            start_idx = min(start_i_global, max_start) if max_start >= 0 else 0
            end_idx = start_idx + W
            target_end = end_idx + H

            if target_end <= n:
                x_np[node_i] = feat[start_idx:end_idx]
                y_np[node_i] = tgt[end_idx:target_end]

        # Convert to tensor (avoid copying)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        node_coords = torch.from_numpy(self.coords_np)

        result = {
            'x': x,
            'y': y,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'node_coords': node_coords
        }

        # Cache result
        if self.cache_data:
            self.cached_samples[idx] = result

        return result


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


# ==================== Model Variants ====================
class FullModel(nn.Module):
    """Full Model: GAT + BiLSTM + DA (Batched GAT Processing)"""

    def __init__(self, num_features, base_factor=64, gat_hidden=256, lstm_hidden=512,
                 num_gat_layers=3, num_heads=4, dropout=0.25, pred_horizon=7):
        super().__init__()

        self.gat_hidden = gat_hidden
        self.lstm_hidden = lstm_hidden
        self.pred_horizon = pred_horizon

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, base_factor),
            nn.SiLU(),
            nn.LayerNorm(base_factor),
            nn.Dropout(dropout)
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()

        gat_dims = [base_factor] + [gat_hidden] * num_gat_layers

        for i in range(num_gat_layers):
            self.gat_layers.append(GATv2Conv(
                in_channels=gat_dims[i],
                out_channels=gat_dims[i + 1] // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=1
            ))
            self.gat_norms.append(nn.LayerNorm(gat_dims[i + 1]))
            self.gat_dropouts.append(nn.Dropout(dropout))

        self.gat_residual_proj = nn.Linear(base_factor, gat_hidden)

        self.bilstm = nn.LSTM(
            input_size=gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.bilstm_norm = nn.LayerNorm(lstm_hidden * 2)

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(lstm_hidden * 2)

        self.da_module = DirectionAwareModule(
            hidden_dim=lstm_hidden * 2,
            east_dim=128,
            north_dim=128,
            up_dim=64
        )

        self.east_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.north_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.up_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, pred_horizon)
        )

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, num_nodes, window_size, num_features = x.shape

        # Reshape to (batch*nodes*time, features)
        x = x.reshape(batch_size * num_nodes * window_size, num_features)
        x = self.input_proj(x)  # (batch*nodes*time, base_factor)

        # Prepare edge attributes
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1) if edge_weight.dim() == 1 else edge_weight
        else:
            edge_attr = None

        # Batched GAT processing
        # Expand edge_index to handle all timesteps
        batch_edge_index = []
        batch_edge_attr = []

        for t in range(window_size):
            offset = t * num_nodes
            batch_edge_index.append(edge_index + offset)
            if edge_attr is not None:
                batch_edge_attr.append(edge_attr)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        if edge_attr is not None:
            batch_edge_attr = torch.cat(batch_edge_attr, dim=0)
        else:
            batch_edge_attr = None

        # Process all timesteps at once
        h = x
        h_residual = self.gat_residual_proj(h)

        for gat, norm, dropout in zip(self.gat_layers, self.gat_norms, self.gat_dropouts):
            h = gat(h, batch_edge_index, edge_attr=batch_edge_attr)
            h = norm(h)
            h = dropout(F.silu(h))

        h = h + h_residual

        # Reshape back to (batch*nodes, time, hidden)
        h_spatial = h.reshape(batch_size * num_nodes, window_size, self.gat_hidden)

        # BiLSTM processing
        h_temporal, _ = self.bilstm(h_spatial)
        h_temporal = self.bilstm_norm(h_temporal)

        # Attention mechanism
        h_attn, _ = self.attention(h_temporal, h_temporal, h_temporal)
        h_attn = self.attention_norm(h_attn + h_temporal)

        # Take last timestep
        h_final = h_attn[:, -1, :]

        # Direction-aware module
        h_east, h_north, h_up = self.da_module(h_final)

        # Prediction heads
        y_east = self.east_head(h_east).reshape(batch_size, num_nodes, self.pred_horizon)
        y_north = self.north_head(h_north).reshape(batch_size, num_nodes, self.pred_horizon)
        y_up = self.up_head(h_up).reshape(batch_size, num_nodes, self.pred_horizon)

        return y_east, y_north, y_up


class NoGATModel(nn.Module):
    """No GAT Module: BiLSTM + DA"""

    def __init__(self, num_features, base_factor=64, lstm_hidden=512,
                 num_heads=4, dropout=0.25, pred_horizon=7):
        super().__init__()

        self.lstm_hidden = lstm_hidden
        self.pred_horizon = pred_horizon

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, base_factor),
            nn.SiLU(),
            nn.LayerNorm(base_factor),
            nn.Dropout(dropout)
        )

        self.bilstm = nn.LSTM(
            input_size=base_factor,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.bilstm_norm = nn.LayerNorm(lstm_hidden * 2)

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(lstm_hidden * 2)

        self.da_module = DirectionAwareModule(
            hidden_dim=lstm_hidden * 2,
            east_dim=128,
            north_dim=128,
            up_dim=64
        )

        self.east_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.north_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.up_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, pred_horizon)
        )

    def forward(self, x, edge_index=None, edge_weight=None):
        batch_size, num_nodes, window_size, num_features = x.shape

        x = x.reshape(batch_size * num_nodes, window_size, num_features)
        x = self.input_proj(x)

        h_temporal, _ = self.bilstm(x)
        h_temporal = self.bilstm_norm(h_temporal)

        h_attn, _ = self.attention(h_temporal, h_temporal, h_temporal)
        h_attn = self.attention_norm(h_attn + h_temporal)

        h_final = h_attn[:, -1, :]

        h_east, h_north, h_up = self.da_module(h_final)

        y_east = self.east_head(h_east).reshape(batch_size, num_nodes, self.pred_horizon)
        y_north = self.north_head(h_north).reshape(batch_size, num_nodes, self.pred_horizon)
        y_up = self.up_head(h_up).reshape(batch_size, num_nodes, self.pred_horizon)

        return y_east, y_north, y_up


class NoBiLSTMModel(nn.Module):
    """No BiLSTM Module: GAT + DA (Batched GAT)"""

    def __init__(self, num_features, base_factor=64, gat_hidden=256,
                 num_gat_layers=3, num_heads=4, dropout=0.25, pred_horizon=7):
        super().__init__()

        self.gat_hidden = gat_hidden
        self.pred_horizon = pred_horizon

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, base_factor),
            nn.SiLU(),
            nn.LayerNorm(base_factor),
            nn.Dropout(dropout)
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()

        gat_dims = [base_factor] + [gat_hidden] * num_gat_layers

        for i in range(num_gat_layers):
            self.gat_layers.append(GATv2Conv(
                in_channels=gat_dims[i],
                out_channels=gat_dims[i + 1] // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=1
            ))
            self.gat_norms.append(nn.LayerNorm(gat_dims[i + 1]))
            self.gat_dropouts.append(nn.Dropout(dropout))

        self.gat_residual_proj = nn.Linear(base_factor, gat_hidden)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.da_module = DirectionAwareModule(
            hidden_dim=gat_hidden,
            east_dim=128,
            north_dim=128,
            up_dim=64
        )

        self.east_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.north_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.up_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, pred_horizon)
        )

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, num_nodes, window_size, num_features = x.shape

        # Batched GAT processing
        x = x.reshape(batch_size * num_nodes * window_size, num_features)
        x = self.input_proj(x)

        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1) if edge_weight.dim() == 1 else edge_weight
        else:
            edge_attr = None

        # Batched edge_index
        batch_edge_index = []
        batch_edge_attr = []

        for t in range(window_size):
            offset = t * num_nodes
            batch_edge_index.append(edge_index + offset)
            if edge_attr is not None:
                batch_edge_attr.append(edge_attr)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        if edge_attr is not None:
            batch_edge_attr = torch.cat(batch_edge_attr, dim=0)
        else:
            batch_edge_attr = None

        h = x
        h_residual = self.gat_residual_proj(h)

        for gat, norm, dropout in zip(self.gat_layers, self.gat_norms, self.gat_dropouts):
            h = gat(h, batch_edge_index, edge_attr=batch_edge_attr)
            h = norm(h)
            h = dropout(F.silu(h))

        h = h + h_residual

        h_spatial = h.reshape(batch_size * num_nodes, window_size, self.gat_hidden)

        # Temporal pooling
        h_final = self.temporal_pool(h_spatial.transpose(1, 2)).squeeze(-1)

        h_east, h_north, h_up = self.da_module(h_final)

        y_east = self.east_head(h_east).reshape(batch_size, num_nodes, self.pred_horizon)
        y_north = self.north_head(h_north).reshape(batch_size, num_nodes, self.pred_horizon)
        y_up = self.up_head(h_up).reshape(batch_size, num_nodes, self.pred_horizon)

        return y_east, y_north, y_up


class NoDAModel(nn.Module):
    """No DA Module: GAT + BiLSTM (Batched GAT)"""

    def __init__(self, num_features, base_factor=64, gat_hidden=256, lstm_hidden=512,
                 num_gat_layers=3, num_heads=4, dropout=0.25, pred_horizon=7):
        super().__init__()

        self.gat_hidden = gat_hidden
        self.lstm_hidden = lstm_hidden
        self.pred_horizon = pred_horizon

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, base_factor),
            nn.SiLU(),
            nn.LayerNorm(base_factor),
            nn.Dropout(dropout)
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()

        gat_dims = [base_factor] + [gat_hidden] * num_gat_layers

        for i in range(num_gat_layers):
            self.gat_layers.append(GATv2Conv(
                in_channels=gat_dims[i],
                out_channels=gat_dims[i + 1] // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=1
            ))
            self.gat_norms.append(nn.LayerNorm(gat_dims[i + 1]))
            self.gat_dropouts.append(nn.Dropout(dropout))

        self.gat_residual_proj = nn.Linear(base_factor, gat_hidden)

        self.bilstm = nn.LSTM(
            input_size=gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.bilstm_norm = nn.LayerNorm(lstm_hidden * 2)

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(lstm_hidden * 2)

        self.east_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.north_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, pred_horizon)
        )

        self.up_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, pred_horizon)
        )

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, num_nodes, window_size, num_features = x.shape

        # Batched GAT
        x = x.reshape(batch_size * num_nodes * window_size, num_features)
        x = self.input_proj(x)

        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1) if edge_weight.dim() == 1 else edge_weight
        else:
            edge_attr = None

        batch_edge_index = []
        batch_edge_attr = []

        for t in range(window_size):
            offset = t * num_nodes
            batch_edge_index.append(edge_index + offset)
            if edge_attr is not None:
                batch_edge_attr.append(edge_attr)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        if edge_attr is not None:
            batch_edge_attr = torch.cat(batch_edge_attr, dim=0)
        else:
            batch_edge_attr = None

        h = x
        h_residual = self.gat_residual_proj(h)

        for gat, norm, dropout in zip(self.gat_layers, self.gat_norms, self.gat_dropouts):
            h = gat(h, batch_edge_index, edge_attr=batch_edge_attr)
            h = norm(h)
            h = dropout(F.silu(h))

        h = h + h_residual

        h_spatial = h.reshape(batch_size * num_nodes, window_size, self.gat_hidden)

        h_temporal, _ = self.bilstm(h_spatial)
        h_temporal = self.bilstm_norm(h_temporal)

        h_attn, _ = self.attention(h_temporal, h_temporal, h_temporal)
        h_attn = self.attention_norm(h_attn + h_temporal)

        h_final = h_attn[:, -1, :]

        y_east = self.east_head(h_final).reshape(batch_size, num_nodes, self.pred_horizon)
        y_north = self.north_head(h_final).reshape(batch_size, num_nodes, self.pred_horizon)
        y_up = self.up_head(h_final).reshape(batch_size, num_nodes, self.pred_horizon)

        return y_east, y_north, y_up


# ==================== Loss Function ====================
class DirectionLoss(nn.Module):
    def __init__(self, delta=1.0, beta=0.1, gamma=0.05, delta_smooth=0.01):
        super().__init__()
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.delta_smooth = delta_smooth

    def forward(self, pred_e, pred_n, pred_u, target_e, target_n, target_u):
        # Huber loss
        huber_e = F.huber_loss(pred_e, target_e, delta=self.delta)
        huber_n = F.huber_loss(pred_n, target_n, delta=self.delta)
        huber_u = F.huber_loss(pred_u, target_u, delta=self.delta)
        huber_total = (huber_e + huber_n + huber_u) / 3.0

        # Temporal drift penalty
        def temporal_diff(x):
            return x[:, :, 1:] - x[:, :, :-1]

        drift_e = (temporal_diff(pred_e) - temporal_diff(target_e)) ** 2
        drift_n = (temporal_diff(pred_n) - temporal_diff(target_n)) ** 2
        drift_u = (temporal_diff(pred_u) - temporal_diff(target_u)) ** 2
        drift = (drift_e.mean() + drift_n.mean() + drift_u.mean()) / 3.0

        # Covariance constraint
        e_east = (pred_e - target_e).flatten()
        e_north = (pred_n - target_n).flatten()
        e_up = (pred_u - target_u).flatten()

        e_east = e_east - e_east.mean()
        e_north = e_north - e_north.mean()
        e_up = e_up - e_up.mean()

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
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_losses = []

        # Training phase
        for batch in train_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            edge_index = batch['edge_index'][0].to(device, non_blocking=True)
            edge_weight = batch['edge_weight'][0].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)

            target_e = y[:, :, :, 0]
            target_n = y[:, :, :, 1]
            target_u = y[:, :, :, 2]

            loss = criterion(pred_e, pred_n, pred_u, target_e, target_n, target_u)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                edge_index = batch['edge_index'][0].to(device, non_blocking=True)
                edge_weight = batch['edge_weight'][0].to(device, non_blocking=True)

                pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)

                target_e = y[:, :, :, 0]
                target_n = y[:, :, :, 1]
                target_u = y[:, :, :, 2]

                loss = criterion(pred_e, pred_n, pred_u, target_e, target_n, target_u)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return history, best_val_loss


def evaluate_model(model, test_loader, device):
    model.eval()

    all_pred_e, all_pred_n, all_pred_u = [], [], []
    all_target_e, all_target_n, all_target_u = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            edge_index = batch['edge_index'][0].to(device, non_blocking=True)
            edge_weight = batch['edge_weight'][0].to(device, non_blocking=True)

            pred_e, pred_n, pred_u = model(x, edge_index, edge_weight)

            all_pred_e.append(pred_e.cpu().numpy())
            all_pred_n.append(pred_n.cpu().numpy())
            all_pred_u.append(pred_u.cpu().numpy())

            all_target_e.append(y[:, :, :, 0].cpu().numpy())
            all_target_n.append(y[:, :, :, 1].cpu().numpy())
            all_target_u.append(y[:, :, :, 2].cpu().numpy())

    pred_e = np.concatenate(all_pred_e)
    pred_n = np.concatenate(all_pred_n)
    pred_u = np.concatenate(all_pred_u)

    target_e = np.concatenate(all_target_e)
    target_n = np.concatenate(all_target_n)
    target_u = np.concatenate(all_target_u)

    metrics_east = compute_metrics(pred_e, target_e)
    metrics_north = compute_metrics(pred_n, target_n)
    metrics_up = compute_metrics(pred_u, target_u)

    return {
        'East': metrics_east,
        'North': metrics_north,
        'Up': metrics_up
    }


# ==================== Main Experiment ====================
def main():
    print("\nConfiguring experiment parameters...")

    feature_dir = r"C:\PycharmProjects\pythonProject\GNSStimeseriesprediction\features"
    output_dir = r"C:\PycharmProjects\pythonProject\GNSStimeseriesprediction\ablation_results"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 80)
    print("Step 1: Load datasets")
    print("=" * 80)

    # Enable data caching for acceleration
    train_dataset = GNSSGraphDataset(feature_dir, split='train', window_size=30,
                                     pred_horizon=7, cache_data=False)
    val_dataset = GNSSGraphDataset(feature_dir, split='val', window_size=30,
                                   pred_horizon=7, cache_data=True)
    test_dataset = GNSSGraphDataset(feature_dir, split='test', window_size=30,
                                    pred_horizon=7, cache_data=True)

    # Increase num_workers and enable pin_memory to accelerate data loading
    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    num_features = train_dataset.num_features
    print(f"Feature dimensions: {num_features}")

    models_config = {
        'Full_Model': FullModel,
        'No_GAT': NoGATModel,
        'No_BiLSTM': NoBiLSTMModel,
        'No_DA': NoDAModel
    }

    results = {}

    print("\n" + "=" * 80)
    print("Step 2: Start ablation study")
    print("=" * 80)

    for model_name, ModelClass in models_config.items():
        print(f"\n{'=' * 80}")
        print(f"Training model: {model_name}")
        print(f"{'=' * 80}")

        model = ModelClass(num_features=num_features)
        criterion = DirectionLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

        print(f"Start training {model_name}...")
        start_time = time.time()

        history, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, epochs=50
        )

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

        print(f"\nEvaluating {model_name}...")
        test_metrics = evaluate_model(model, test_loader, device)

        results[model_name] = {
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'history': history,
            'training_time': training_time
        }

        print(f"\n{model_name} Test Set Results:")
        print(
            f"  East - MAE: {test_metrics['East']['MAE']:.6f}, RMSE: {test_metrics['East']['RMSE']:.6f}, R2: {test_metrics['East']['R2']:.4f}")
        print(
            f"  North - MAE: {test_metrics['North']['MAE']:.6f}, RMSE: {test_metrics['North']['RMSE']:.6f}, R2: {test_metrics['North']['R2']:.4f}")
        print(
            f"  Up - MAE: {test_metrics['Up']['MAE']:.6f}, RMSE: {test_metrics['Up']['RMSE']:.6f}, R2: {test_metrics['Up']['R2']:.4f}")

    print("\n" + "=" * 80)
    print("Step 3: Save results")
    print("=" * 80)

    with open(os.path.join(output_dir, 'ablation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    summary_file = os.path.join(output_dir, 'ablation_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Ablation Study Results Summary\n")
        f.write("=" * 80 + "\n\n")

        for model_name in models_config.keys():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Validation Loss: {results[model_name]['best_val_loss']:.6f}\n")
            f.write(f"  Training Time: {results[model_name]['training_time']:.2f} seconds\n")
            metrics = results[model_name]['test_metrics']
            f.write(
                f"  East - MAE: {metrics['East']['MAE']:.6f}, RMSE: {metrics['East']['RMSE']:.6f}, R2: {metrics['East']['R2']:.4f}\n")
            f.write(
                f"  North - MAE: {metrics['North']['MAE']:.6f}, RMSE: {metrics['North']['RMSE']:.6f}, R2: {metrics['North']['R2']:.4f}\n")
            f.write(
                f"  Up - MAE: {metrics['Up']['MAE']:.6f}, RMSE: {metrics['Up']['RMSE']:.6f}, R2: {metrics['Up']['R2']:.4f}\n")

    print(f"Results saved to: {output_dir}")
    print("\nAblation study complete!")


if __name__ == '__main__':
    main()