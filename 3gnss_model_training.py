# -*- coding: utf-8 -*-
import sys
import time

print("=" * 80)
print("GNSS Time Series Prediction Model - Initialization")
print("=" * 80)

print("\n[Step 1/10] Environment Check...")
try:
    import numpy as np

    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")

    import warnings

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')
except Exception as e:
    print(f"NumPy import failed: {e}")
    sys.exit(1)

print("\n[Step 2/10] Importing Core Libraries...")

print("pandas imported successfully")

import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch imported successfully")

from torch.utils.data import DataLoader

print("torch.utils.data imported successfully")

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv

    print("torch_geometric imported successfully")
except Exception as e:
    print(f"torch_geometric import failed: {e}")
    print("\n Attempting compatibility fix...")
    import subprocess

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "torch-geometric"],
                       capture_output=True, check=True)
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import GATv2Conv

        print("torch_geometric imported")
    except:
        print("Please install: pip install torch-geometric")
        sys.exit(1)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc
import logging
import json


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        import torch
        from collections import defaultdict

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()

        if isinstance(obj, defaultdict):
            return dict(obj)

        if isinstance(obj, set):
            return list(obj)

        return super().default(obj)


from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')

print("All dependency libraries imported successfully\n")


class ProgressBar:
    """Dynamic progress bar"""

    def __init__(self, total, desc="Progress", width=40):
        self.total = total
        self.current = 0
        self.desc = desc
        self.width = width
        self.start_time = time.time()

    def update(self, n=1):
        """Update progress"""
        self.current += n
        self._display()

    def set_postfix(self, **kwargs):
        """Set postfix information"""
        self.postfix = kwargs
        self._display()

    def _display(self):
        """Display progress bar"""
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * percent)
        bar = '#' * filled + '-' * (self.width - filled)

        elapsed = time.time() - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0

        def format_time(seconds):
            if seconds < 60:
                return f"{int(seconds)}s"
            elif seconds < 3600:
                return f"{int(seconds / 60)}m {int(seconds % 60)}s"
            else:
                return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"

        msg = f"\r{self.desc}: |{bar}| {self.current}/{self.total} [{percent * 100:.1f}%] "
        msg += f"[{format_time(elapsed)}<{format_time(eta)}, {speed:.1f}it/s]"

        if hasattr(self, 'postfix') and self.postfix:
            postfix_str = ' '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in self.postfix.items()])
            msg += f" {postfix_str}"

        print(msg, end='', flush=True)

    def close(self):
        print()


def setup_training_logger(output_dir: str = "./model_output") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_log_{timestamp}.txt")

    logger = logging.getLogger("GNSSTraining")
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except:
            pass

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import logging


class GNSSGraphDataset(Dataset):
    def __init__(
            self,
            feature_dir: str,
            split: str = 'train',
            window_size: int = 60,
            pred_horizon: int = 7,
            k_neighbors: int = 10,
            distance_threshold: float = 1000.0,
            selected_features_file: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            normalize: bool = True,
            shared_scaler: Optional[RobustScaler] = None
    ):
        self.feature_dir = feature_dir
        self.split = split
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.logger = logger or logging.getLogger("GNSSTraining")
        self.normalize = normalize

        self.stations, self.data_dict, self.station_coords = self._load_all_stations()

        if self.normalize:
            if shared_scaler is not None:
                self.scaler = shared_scaler
            else:
                self.scaler = RobustScaler()

        if selected_features_file and os.path.exists(selected_features_file):
            with open(selected_features_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            sample_df = list(self.data_dict.values())[0]
            exclude_cols = ['station', 'decimal_year', 'mjd', 'east', 'north', 'up']
            self.feature_names = [c for c in sample_df.columns if c not in exclude_cols]

        if self.normalize:
            self._normalize_data()
        print(f"Building spatial graph...")
        self.edge_index, self.edge_weight = self._build_spatial_graph()

        print(f"Generating samples...")
        self.samples = self._generate_samples()

        print(f"Data loading complete: {len(self.samples)} samples")

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

            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

            self.feat_np[s] = feat
            self.tgt_np[s] = tgt
            self.length_np[s] = feat.shape[0]

        self.num_nodes = len(self.stations)
        self.num_features = len(self.feature_names)

    def _normalize_data(self):
        if self.split == 'train':
            all_features = []
            for station in self.stations:
                df = self.data_dict[station]
                features = df[self.feature_names].values
                all_features.append(features)

            all_features_concat = np.vstack(all_features)
            self.scaler.fit(all_features_concat)

        for station in self.stations:
            df = self.data_dict[station]
            features = df[self.feature_names].values
            normalized_features = self.scaler.transform(features)
            df[self.feature_names] = normalized_features
            self.data_dict[station] = df

    def _load_all_stations(self) -> Tuple[List[str], Dict, pd.DataFrame]:
        """Load all station data"""
        stations = sorted([d for d in os.listdir(self.feature_dir)
                           if os.path.isdir(os.path.join(self.feature_dir, d))])

        data_dict = {}
        coords_list = []

        for station in stations:
            csv_path = os.path.join(self.feature_dir, station, f'{self.split}_featured.csv')
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if len(df) < self.window_size + self.pred_horizon:
                continue

            data_dict[station] = df

            if all(c in df.columns for c in ['latitude', 'longitude', 'height']):
                coords_list.append({
                    'station': station,
                    'latitude': df['latitude'].iloc[0],
                    'longitude': df['longitude'].iloc[0],
                    'height': df['height'].iloc[0]
                })

        coords_df = pd.DataFrame(coords_list)
        stations = sorted(data_dict.keys())

        print(f"Loaded {len(stations)} stations")
        self.logger.info(f"Loaded stations: {len(stations)}")

        return stations, data_dict, coords_df

    def _build_spatial_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build spatial graph using KNN"""
        from sklearn.neighbors import NearestNeighbors

        coords_df = self.station_coords.set_index('station')
        coords_list = []
        for s in self.stations:
            row = coords_df.loc[s]
            coords_list.append([row['latitude'], row['longitude']])
        coords = np.array(coords_list)

        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(coords)),
                                algorithm='ball_tree', metric='haversine')
        nbrs.fit(np.radians(coords))
        distances, indices = nbrs.kneighbors(np.radians(coords))

        distances = distances * 6371.0

        edges = []
        weights = []

        for i in range(len(coords)):
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                if dist < self.distance_threshold:
                    edges.append([i, j])
                    weights.append(1.0 / (dist + 1e-6))

        if len(edges) == 0:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    edges.append([i, j])
                    edges.append([j, i])
                    weights.extend([1.0, 1.0])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        return edge_index, edge_weight

    def _generate_samples(self) -> List[Dict]:
        """Generate sliding window samples"""
        samples = []

        for station in self.stations:
            df_len = len(self.data_dict[station])

            for t in range(df_len - self.window_size - self.pred_horizon + 1):
                samples.append({
                    'station': station,
                    'start_idx': t,
                    'end_idx': t + self.window_size,
                    'target_start': t + self.window_size,
                    'target_end': t + self.window_size + self.pred_horizon
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Optimized __getitem__: use pre-converted numpy arrays
        Returns:
            x: [num_nodes, window_size, num_features]
            y: [num_nodes, pred_horizon, 3]
            edge_index, edge_weight, coords
        """
        sample = self.samples[idx]
        station = sample['station']

        station_idx = self.stations.index(station)

        x_all = np.zeros((self.num_nodes, self.window_size, self.num_features), dtype=np.float32)
        y_all = np.zeros((self.num_nodes, self.pred_horizon, 3), dtype=np.float32)

        for node_idx, s in enumerate(self.stations):
            feat_s = self.feat_np[s]
            tgt_s = self.tgt_np[s]

            if s == station:
                start = sample['start_idx']
                end = sample['end_idx']
                tgt_start = sample['target_start']
                tgt_end = sample['target_end']
            else:
                start = sample['start_idx']
                end = sample['end_idx']
                tgt_start = sample['target_start']
                tgt_end = sample['target_end']

                if end > self.length_np[s]:
                    end = self.length_np[s]
                    start = max(0, end - self.window_size)
                if tgt_end > self.length_np[s]:
                    tgt_end = self.length_np[s]
                    tgt_start = max(0, tgt_end - self.pred_horizon)

            x_window = feat_s[start:end]
            y_window = tgt_s[tgt_start:tgt_end]

            if x_window.shape[0] < self.window_size:
                pad_x = np.zeros((self.window_size - x_window.shape[0], self.num_features), dtype=np.float32)
                x_window = np.vstack([pad_x, x_window])

            if y_window.shape[0] < self.pred_horizon:
                pad_y = np.zeros((self.pred_horizon - y_window.shape[0], 3), dtype=np.float32)
                y_window = np.vstack([pad_y, y_window])

            x_all[node_idx] = x_window
            y_all[node_idx] = y_window

        return {
            'x': torch.from_numpy(x_all),
            'y': torch.from_numpy(y_all),
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'coords': torch.from_numpy(self.coords_np),
            'station_idx': station_idx
        }


def collate_graph_batch(batch):
    """Custom collate function for graph batches"""
    x = torch.stack([item['x'] for item in batch])
    y = torch.stack([item['y'] for item in batch])

    edge_index = batch[0]['edge_index']
    edge_weight = batch[0]['edge_weight']
    coords = batch[0]['coords']

    return {
        'x': x,
        'y': y,
        'edge_index': (edge_index, edge_weight),
        'edge_weight': (edge_weight,),
        'coords': coords
    }


class DirectionAwareGATLayer(nn.Module):
    """Direction-Aware Graph Attention Layer with adaptive feature refinement"""

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.directions = ['east', 'north', 'up']

        # 计算每个方向的特征维度
        self.features_per_dir = out_features // len(self.directions)

        self.gat = GATv2Conv(
            in_channels=in_features,
            out_channels=self.head_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
            edge_dim=1  # 添加边特征维度
        )

        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

        self.direction_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Sigmoid()
        )

        # 修复：使用 features_per_dir 而不是完整的输出维度
        self.feature_refinement_gates = nn.ModuleDict()
        for i, dir_i in enumerate(self.directions):
            for j, dir_j in enumerate(self.directions):
                if i != j:
                    gate_name = f"refine_{dir_i}_{dir_j}"
                    self.feature_refinement_gates[gate_name] = nn.Sequential(
                        nn.Linear(2 * self.features_per_dir, 128),  # ← 修复：使用 features_per_dir
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )

        self.direction_transforms = nn.ModuleDict()
        for dir_name in self.directions:
            self.direction_transforms[dir_name] = nn.Sequential(
                nn.Linear(self.features_per_dir, self.features_per_dir),  # ← 修复：使用 features_per_dir
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, coords):
        num_nodes = x.size(0)

        src, tgt = edge_index
        rel_pos = coords[tgt] - coords[src]

        dir_weights = self.direction_mlp(rel_pos)

        # 将 edge_weight 从 [num_edges] 重塑为 [num_edges, 1]
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
        out = self.gat(x, edge_index, edge_attr=edge_attr)

        if dir_weights.size(0) == out.size(0):
            dir_weights_expanded = dir_weights.unsqueeze(-1).expand(-1, -1, self.head_dim)
            dir_weights_expanded = dir_weights_expanded.reshape(dir_weights.size(0), -1)
            out = out * dir_weights_expanded

        residual = self.residual_proj(x)
        if residual.size(-1) == out.size(-1):
            out = self.layer_norm(out + residual)
        else:
            out = self.layer_norm(out)

        refined_out = out.clone()
        features_per_dir = out.size(1) // len(self.directions)

        for i, dir_i in enumerate(self.directions):
            start_idx = i * features_per_dir
            end_idx = (i + 1) * features_per_dir
            h_current = out[:, start_idx:end_idx]
            aggregated_info = []

            for j, dir_j in enumerate(self.directions):
                if i != j:
                    j_start = j * features_per_dir
                    j_end = (j + 1) * features_per_dir
                    h_other = out[:, j_start:j_end]

                    gate_name = f"refine_{dir_i}_{dir_j}"
                    gate_input = torch.cat([h_current, h_other], dim=1)
                    gate_value = self.feature_refinement_gates[gate_name](gate_input)
                    transformed = self.direction_transforms[dir_j](h_other)
                    weighted = gate_value * transformed
                    aggregated_info.append(weighted)

            if aggregated_info:
                aggregated = torch.sum(torch.stack(aggregated_info), dim=0)
                refined_out[:, start_idx:end_idx] = h_current + aggregated
            else:
                refined_out[:, start_idx:end_idx] = h_current

        out = self.dropout(refined_out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for temporal attention"""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.output_proj(context)
        output = self.layer_norm(output + x)

        return output


class DAGATBiLSTM(nn.Module):
    """Direction-Aware GAT + BiLSTM Model with adaptive state enhancement"""

    def __init__(
            self,
            num_features,
            base_factor=64,
            gat_hidden=256,
            lstm_hidden=512,
            num_gat_layers=3,
            num_heads=4,
            dropout=0.2,
            pred_horizon=7
    ):
        super().__init__()

        self.num_features = num_features
        self.pred_horizon = pred_horizon

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, base_factor),
            nn.LayerNorm(base_factor),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gat_layers = nn.ModuleList()
        current_dim = base_factor
        for i in range(num_gat_layers):
            next_dim = gat_hidden
            self.gat_layers.append(
                DirectionAwareGATLayer(current_dim, next_dim, num_heads, dropout)
            )
            current_dim = next_dim

        self.bilstm = nn.LSTM(
            input_size=gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.temporal_attention = MultiHeadAttention(
            hidden_size=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout
        )

        # Direction-specific encoders
        self.east_encoder = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.SiLU(),
            nn.LayerNorm(128)
        )

        self.north_encoder = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.SiLU(),
            nn.LayerNorm(128)
        )

        self.up_encoder = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.SiLU(),
            nn.LayerNorm(64)
        )

        # Cross-directional gates
        self.gate_e_n = nn.Linear(128 * 2, 128)
        self.gate_e_u = nn.Linear(128 + 64, 128)
        self.gate_n_e = nn.Linear(128 * 2, 128)
        self.gate_n_u = nn.Linear(128 + 64, 128)
        self.gate_u_e = nn.Linear(64 + 128, 64)
        self.gate_u_n = nn.Linear(64 + 128, 64)

        # Cross-directional transformation matrices
        self.V_en = nn.Linear(128, 128)
        self.V_eu = nn.Linear(64, 128)
        self.V_ne = nn.Linear(128, 128)
        self.V_nu = nn.Linear(64, 128)
        self.V_ue = nn.Linear(128, 64)
        self.V_un = nn.Linear(128, 64)

        # Direction-specific prediction heads
        self.fc_east = nn.Linear(128, pred_horizon)
        self.fc_north = nn.Linear(128, pred_horizon)
        self.fc_up = nn.Linear(64, pred_horizon)

    def forward(self, x, edge_index, edge_weight, coords=None):
        batch_size, num_nodes, window_size, _ = x.shape

        temporal_features = []
        for t in range(window_size):
            x_t = x[:, :, t, :]
            batch_out = []
            for b in range(batch_size):
                x_b = x_t[b]
                h = self.input_proj(x_b)
                for gat_layer in self.gat_layers:
                    h = gat_layer(h, edge_index, edge_weight, coords)
                batch_out.append(h)
            h_t = torch.stack(batch_out, dim=0)
            temporal_features.append(h_t)

        temporal_features = torch.stack(temporal_features, dim=2)
        combined_batch = temporal_features.reshape(batch_size * num_nodes, window_size, -1)

        lstm_output, _ = self.bilstm(combined_batch)
        lstm_attended = self.temporal_attention(lstm_output)
        lstm_final = lstm_attended[:, -1, :]
        lstm_out = lstm_final.reshape(batch_size, num_nodes, -1)

        # Apply direction-specific encoders
        h_east_all = self.east_encoder(lstm_out)
        h_north_all = self.north_encoder(lstm_out)
        h_up_all = self.up_encoder(lstm_out)

        # Cross-directional information aggregation
        enhanced_east_list = []
        enhanced_north_list = []
        enhanced_up_list = []

        for node_idx in range(num_nodes):
            h_e = h_east_all[:, node_idx, :]
            h_n = h_north_all[:, node_idx, :]
            h_u = h_up_all[:, node_idx, :]

            # East direction: aggregate from North and Up
            g_en = torch.sigmoid(self.gate_e_n(torch.cat([h_e, h_n], dim=-1)))
            g_eu = torch.sigmoid(self.gate_e_u(torch.cat([h_e, h_u], dim=-1)))
            h_e_enhanced = h_e + g_en * self.V_en(h_n) + g_eu * self.V_eu(h_u)

            # North direction: aggregate from East and Up
            g_ne = torch.sigmoid(self.gate_n_e(torch.cat([h_n, h_e], dim=-1)))
            g_nu = torch.sigmoid(self.gate_n_u(torch.cat([h_n, h_u], dim=-1)))
            h_n_enhanced = h_n + g_ne * self.V_ne(h_e) + g_nu * self.V_nu(h_u)

            # Up direction: aggregate from East and North
            g_ue = torch.sigmoid(self.gate_u_e(torch.cat([h_u, h_e], dim=-1)))
            g_un = torch.sigmoid(self.gate_u_n(torch.cat([h_u, h_n], dim=-1)))
            h_u_enhanced = h_u + g_ue * self.V_ue(h_e) + g_un * self.V_un(h_n)

            enhanced_east_list.append(h_e_enhanced)
            enhanced_north_list.append(h_n_enhanced)
            enhanced_up_list.append(h_u_enhanced)

        enhanced_east = torch.stack(enhanced_east_list, dim=1)
        enhanced_north = torch.stack(enhanced_north_list, dim=1)
        enhanced_up = torch.stack(enhanced_up_list, dim=1)

        pred_east = self.fc_east(enhanced_east)
        pred_north = self.fc_north(enhanced_north)
        pred_up = self.fc_up(enhanced_up)
        return pred_east, pred_north, pred_up


class DirectionLoss(nn.Module):
    def __init__(self, device='cpu', beta_velocity=0.1, beta_drift=0.1,
                 beta_cov=0.05, beta_smooth=0.01,
                 drift_penalty_weight=0.1, covariance_weight=0.1,
                 huber_deltas=(1.0, 1.0, 1.0),
                 dynamic_weight_eps=1e-6):
        super().__init__()
        self.device = device
        self.beta_velocity = beta_velocity
        self.beta_drift = beta_drift
        self.beta_cov = beta_cov
        self.beta_smooth = beta_smooth
        self.huber_deltas = huber_deltas
        self.dynamic_weight_eps = dynamic_weight_eps

        self.direction_weights = nn.Parameter(torch.ones(3))
        self.historical_errors = None
        self.ema_alpha = 0.1

    def update_dynamic_weights(self, current_errors):
        if self.historical_errors is None:
            self.historical_errors = current_errors
        else:
            self.historical_errors = (self.ema_alpha * current_errors +
                                      (1 - self.ema_alpha) * self.historical_errors)

        with torch.no_grad():
            error_reciprocal = 1.0 / (self.historical_errors + self.dynamic_weight_eps)
            softmax_weights = F.softmax(self.direction_weights, dim=0)
            dynamic_weights = softmax_weights * error_reciprocal
            dynamic_weights = dynamic_weights / (dynamic_weights.sum() + 1e-8)

        return dynamic_weights

    def forward(self, pred_e, pred_n, pred_u, target_e, target_n, target_u):
        target_e = target_e.squeeze(-1)
        target_n = target_n.squeeze(-1)
        target_u = target_u.squeeze(-1)

        batch_size, num_nodes, pred_horizon = pred_e.shape

        with torch.no_grad():
            current_errors = torch.tensor([
                F.mse_loss(pred_e, target_e).item(),
                F.mse_loss(pred_n, target_n).item(),
                F.mse_loss(pred_u, target_u).item()
            ], device=self.device)

        dynamic_weights = self.update_dynamic_weights(current_errors)

        def directional_huber_loss(pred, target, delta):
            abs_diff = torch.abs(pred - target)
            loss = torch.where(abs_diff < delta,
                               0.5 * abs_diff ** 2,
                               delta * (abs_diff - 0.5 * delta))
            return loss.mean()

        huber_e = directional_huber_loss(pred_e, target_e, self.huber_deltas[0])
        huber_n = directional_huber_loss(pred_n, target_n, self.huber_deltas[1])
        huber_u = directional_huber_loss(pred_u, target_u, self.huber_deltas[2])

        weighted_huber = (dynamic_weights[0] * huber_e +
                          dynamic_weights[1] * huber_n +
                          dynamic_weights[2] * huber_u)

        def velocity_loss(pred, target):
            pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
            target_diff = target[:, :, 1:] - target[:, :, :-1]
            return F.mse_loss(pred_diff, target_diff)

        vel_loss = (velocity_loss(pred_e, target_e) +
                    velocity_loss(pred_n, target_n) +
                    velocity_loss(pred_u, target_u))

        def smoothness_loss(pred):
            second_diff = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
            return torch.mean(second_diff ** 2)

        smooth_loss = (smoothness_loss(pred_e) +
                       smoothness_loss(pred_n) +
                       smoothness_loss(pred_u))

        def drift_penalty_loss(pred):
            return torch.mean(torch.abs(pred[:, :, -1] - pred[:, :, 0]))

        drift_loss = self.beta_drift * (
                drift_penalty_loss(pred_e) +
                drift_penalty_loss(pred_n) +
                drift_penalty_loss(pred_u)
        )

        def covariance_constraint_loss(pred_e, pred_n, pred_u):
            batch_size, num_nodes, pred_horizon = pred_e.shape
            combined = torch.stack([pred_e, pred_n, pred_u], dim=-1)
            combined_flat = combined.reshape(batch_size, num_nodes * pred_horizon, 3)

            cov_losses = []
            for i in range(batch_size):
                sample_data = combined_flat[i]
                valid_mask = torch.isfinite(sample_data).all(dim=1)
                if valid_mask.sum() > 1:
                    valid_data = sample_data[valid_mask]
                    centered = valid_data - valid_data.mean(dim=0, keepdim=True)
                    cov_matrix = torch.matmul(centered.t(), centered) / (centered.shape[0] - 1)
                    off_diag_sum = (torch.abs(cov_matrix[0, 1]) +
                                    torch.abs(cov_matrix[0, 2]) +
                                    torch.abs(cov_matrix[1, 2]))
                    cov_losses.append(off_diag_sum)

            return torch.mean(torch.stack(cov_losses)) if cov_losses else torch.tensor(0.0, device=pred_e.device)


        covariance_loss = self.beta_cov * covariance_constraint_loss(pred_e, pred_n, pred_u)

        total_loss = (weighted_huber +
                      self.beta_velocity * vel_loss +
                      self.beta_smooth * smooth_loss +
                      drift_loss +
                      covariance_loss)

        return total_loss, {
            'huber': weighted_huber.item(),
            'velocity': vel_loss.item(),
            'smoothness': smooth_loss.item(),
            'drift': drift_loss.item(),
            'covariance': covariance_loss.item(),
            'dynamic_weights': dynamic_weights.detach().cpu().numpy()
        }


class GNSSMetrics:
    """Metrics for GNSS prediction evaluation"""

    @staticmethod
    def compute_all_metrics(pred_e, pred_n, pred_u, target_e, target_n, target_u):
        """
        Compute comprehensive metrics for all directions

        Args:
            pred_*: [num_samples, num_nodes, pred_horizon]
            target_*: [num_samples, num_nodes, pred_horizon]

        Returns:
            dict: Metrics for each direction
        """
        metrics = {}

        for name, pred, target in [('East', pred_e, target_e),
                                   ('North', pred_n, target_n),
                                   ('Up', pred_u, target_u)]:
            pred_flat = pred.reshape(-1)
            target_flat = target.reshape(-1)

            mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
            pred_clean = pred_flat[mask]
            target_clean = target_flat[mask]

            if len(pred_clean) == 0:
                metrics[name] = {
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MedAE': np.nan,
                    'MAPE': np.nan
                }
                continue

            mae = np.mean(np.abs(pred_clean - target_clean))
            rmse = np.sqrt(np.mean((pred_clean - target_clean) ** 2))

            ss_res = np.sum((target_clean - pred_clean) ** 2)
            ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            medae = np.median(np.abs(pred_clean - target_clean))

            mape_mask = np.abs(target_clean) > 1e-4
            if mape_mask.sum() > 0:
                mape = np.mean(np.abs((target_clean[mape_mask] - pred_clean[mape_mask]) /
                                      target_clean[mape_mask])) * 100
            else:
                mape = np.nan

            metrics[name] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'MedAE': float(medae),
                'MAPE': float(mape)
            }

        return metrics


class GNSSTrainer:
    """GNSS Model Trainer"""

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device='cpu',
            logger=None,
            output_dir='./model_output'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(device)
        self.device = device
        self.logger = logger or logging.getLogger("GNSSTraining")
        self.output_dir = output_dir

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        os.makedirs(output_dir, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        loss_components = defaultdict(float)

        pbar = ProgressBar(len(self.train_loader), desc="Training")

        for batch in self.train_loader:
            x = batch['x'].to(self.device, non_blocking=True)
            y = batch['y'].to(self.device, non_blocking=True)
            edge_index = batch['edge_index'][0].to(self.device, non_blocking=True)
            edge_weight = batch['edge_weight'][0].to(self.device, non_blocking=True)
            coords = batch['coords'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            pred_e, pred_n, pred_u = self.model(x, edge_index, edge_weight, coords)

            loss, components = self.criterion(
                pred_e, pred_n, pred_u,
                y[:, :, :, 0], y[:, :, :, 1], y[:, :, :, 2]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            for k, v in components.items():
                if k != 'dynamic_weights':
                    loss_components[k] += v

            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

        pbar.close()

        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return avg_loss, avg_components

    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_pred_e, all_pred_n, all_pred_u = [], [], []
        all_target_e, all_target_n, all_target_u = [], [], []

        with torch.no_grad():
            pbar = ProgressBar(len(self.val_loader), desc="Validating")
            for batch in self.val_loader:
                x = batch['x'].to(self.device, non_blocking=True)
                y = batch['y'].to(self.device, non_blocking=True)
                edge_index = batch['edge_index'][0].to(self.device, non_blocking=True)
                edge_weight = batch['edge_weight'][0].to(self.device, non_blocking=True)
                coords = batch['coords'].to(self.device, non_blocking=True)

                pred_e, pred_n, pred_u = self.model(x, edge_index, edge_weight, coords)

                loss, _ = self.criterion(
                    pred_e, pred_n, pred_u,
                    y[:, :, :, 0], y[:, :, :, 1], y[:, :, :, 2]
                )

                total_loss += loss.item()
                num_batches += 1

                all_pred_e.append(pred_e.cpu().numpy())
                all_pred_n.append(pred_n.cpu().numpy())
                all_pred_u.append(pred_u.cpu().numpy())
                all_target_e.append(y[:, :, :, 0].cpu().numpy())
                all_target_n.append(y[:, :, :, 1].cpu().numpy())
                all_target_u.append(y[:, :, :, 2].cpu().numpy())

                pbar.update(1)
            pbar.close()

        avg_loss = total_loss / num_batches

        pred_e = np.concatenate(all_pred_e, axis=0)
        pred_n = np.concatenate(all_pred_n, axis=0)
        pred_u = np.concatenate(all_pred_u, axis=0)
        target_e = np.concatenate(all_target_e, axis=0)
        target_n = np.concatenate(all_target_n, axis=0)
        target_u = np.concatenate(all_target_u, axis=0)

        metrics = GNSSMetrics.compute_all_metrics(pred_e, pred_n, pred_u, target_e, target_n, target_u)

        return avg_loss, metrics

    def fit(self, num_epochs=200, patience=30, save_best=True):
        """Training loop"""
        history = defaultdict(list)

        print(f"\nTraining for {num_epochs} epochs with patience={patience}")
        self.logger.info(f"Starting training: epochs={num_epochs}, patience={patience}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss, train_components = self.train_epoch()

            val_loss, val_metrics = self.validate()

            self.scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            for k, v in train_components.items():
                history[f'train_{k}'].append(v)

            for direction in ['East', 'North', 'Up']:
                for metric_name, metric_value in val_metrics[direction].items():
                    history[f'val_{direction}_{metric_name}'].append(metric_value)

            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")

            for direction in ['East', 'North', 'Up']:
                metrics = val_metrics[direction]
                print(f"{direction}: MAE={metrics['MAE'] * 1000:.2f}mm, RMSE={metrics['RMSE'] * 1000:.2f}mm")

            self.logger.info(
                f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                if save_best:
                    self.save_checkpoint(os.path.join(self.output_dir, 'best_model.pth'))
                    print(f"  ✓ New best model saved (val_loss={val_loss:.6f})")
                    self.logger.info(f"Best model updated at epoch {epoch + 1}")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return history

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']


def main_training(
        feature_dir: str,
        output_dir: str,
        use_bayesian_opt: bool = True,
        logger: Optional[logging.Logger] = None
):
    """Main training function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    logger = logger or setup_training_logger(output_dir)

    config = None

    if use_bayesian_opt:
        print(f"\n[Step 3/10] Bayesian Hyperparameter Optimization")

        def objective(trial):
            cfg = {
                'base_factor': trial.suggest_categorical('base_factor', [32, 64, 128]),
                'gat_hidden': trial.suggest_categorical('gat_hidden', [128, 256, 512]),
                'lstm_hidden': trial.suggest_categorical('lstm_hidden', [256, 512, 1024]),
                'num_gat_layers': trial.suggest_int('num_gat_layers', 2, 4),
                'num_heads': trial.suggest_categorical('num_heads', [4, 8]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                'beta': trial.suggest_float('beta', 0.05, 0.2),
                'gamma': trial.suggest_float('gamma', 0.01, 0.1),
                'delta_smooth': trial.suggest_float('delta_smooth', 0.005, 0.02)
            }

            selected_features_file = os.path.join(feature_dir, 'selected_features.txt')

            train_dataset = GNSSGraphDataset(
                feature_dir=feature_dir, split='train', window_size=60, pred_horizon=7,
                k_neighbors=10, selected_features_file=selected_features_file, logger=logger
            )

            val_dataset = GNSSGraphDataset(
                feature_dir=feature_dir, split='val', window_size=60, pred_horizon=7,
                k_neighbors=10, selected_features_file=selected_features_file, logger=logger,
                shared_scaler=train_dataset.scaler
            )

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2,
                                      collate_fn=collate_graph_batch)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2,
                                    collate_fn=collate_graph_batch)

            model = DAGATBiLSTM(
                num_features=len(train_dataset.feature_names),
                base_factor=cfg['base_factor'],
                gat_hidden=cfg['gat_hidden'],
                lstm_hidden=cfg['lstm_hidden'],
                num_gat_layers=cfg['num_gat_layers'],
                num_heads=cfg['num_heads'],
                dropout=cfg['dropout'],
                pred_horizon=7
            )

            criterion = DirectionLoss(
                device=str(device),
                beta_velocity=cfg['beta'],
                beta_smooth=cfg['delta_smooth'],
                beta_drift=0.1,
                beta_cov=cfg['gamma']
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

            trainer = GNSSTrainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                device=str(device), logger=logger, output_dir=output_dir
            )

            history = trainer.fit(num_epochs=50, patience=10, save_best=False)

            val_loss = min(history['val_loss'])

            del model, trainer, train_loader, val_loader, train_dataset, val_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return val_loss

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        study.optimize(
            objective,
            n_trials=30,
            timeout=None
        )

        print(f"\nOptimization complete! Best hyperparameters:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    if config is None:
        config = {
            'base_factor': 64,
            'gat_hidden': 256,
            'lstm_hidden': 512,
            'num_gat_layers': 3,
            'num_heads': 4,
            'dropout': 0.25,
            'learning_rate': 0.002,
            'beta': 0.1,
            'gamma': 0.05,
            'delta_smooth': 0.01
        }
        print(f"\n[Step 4/10] Using Default Configuration")
    else:
        config.setdefault('beta', 0.1)
        config.setdefault('gamma', 0.05)
        config.setdefault('delta_smooth', 0.01)
        print(f"\n[Step 4/10] Using Configuration Parameters")

    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\n[Step 5/10] Loading Datasets")

    selected_features_file = os.path.join(feature_dir, 'selected_features.txt')

    train_dataset = GNSSGraphDataset(
        feature_dir=feature_dir, split='train', window_size=60, pred_horizon=7,
        k_neighbors=10, selected_features_file=selected_features_file, logger=logger
    )

    val_dataset = GNSSGraphDataset(
        feature_dir=feature_dir, split='val', window_size=60, pred_horizon=7,
        k_neighbors=10, selected_features_file=selected_features_file, logger=logger,
        shared_scaler=train_dataset.scaler
    )

    test_dataset = GNSSGraphDataset(
        feature_dir=feature_dir, split='test', window_size=60, pred_horizon=7,
        k_neighbors=10, selected_features_file=selected_features_file, logger=logger,
        shared_scaler=train_dataset.scaler
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_graph_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_graph_batch,
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)

    print(f"\nDataset loading complete:")
    print(f"  Training set: {len(train_dataset)} samples")
    print(f"  Validation set: {len(val_dataset)} samples")
    print(f"  Test set: {len(test_dataset)} samples")

    print(f"\n[Step 6/10] Creating Model")

    model = DAGATBiLSTM(
        num_features=len(train_dataset.feature_names),
        base_factor=config['base_factor'],
        gat_hidden=config['gat_hidden'],
        lstm_hidden=config['lstm_hidden'],
        num_gat_layers=config.get('num_gat_layers', 3),
        num_heads=config.get('num_heads', 4),
        dropout=config['dropout'],
        pred_horizon=7
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created")
    print(f"  Total parameters: {total_params:,}")
    logger.info(f"Model parameters: {total_params:,}")

    print(f"\n[Step 7/10] Configuring Optimizer and Loss")

    criterion = DirectionLoss(
        device=str(device),
        beta_velocity=config.get('beta', 0.1),
        beta_smooth=config.get('delta_smooth', 0.01),
        beta_drift=0.1,
        beta_cov=config.get('gamma', 0.05)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    print(f"Optimizer: AdamW (lr={config['learning_rate']}, wd=1e-5)")
    print(f"Learning rate schedule: CosineAnnealingWarmRestarts")
    print(f"Loss function: DirectionLoss (physics-constrained)")

    print(f"\n[Step 8/10] Starting Training")

    trainer = GNSSTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=str(device), logger=logger, output_dir=output_dir
    )

    history = trainer.fit(num_epochs=200, patience=30, save_best=True)

    print(f"\n[Step 9/10] Test Set Evaluation")
    trainer.load_checkpoint(os.path.join(output_dir, 'best_model.pth'))
    model.eval()

    all_pred_e, all_pred_n, all_pred_u = [], [], []
    all_target_e, all_target_n, all_target_u = [], [], []

    with torch.no_grad():
        pbar = ProgressBar(len(test_loader), desc="Testing")
        for batch in test_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            edge_index = batch['edge_index'][0].to(device, non_blocking=True)
            edge_weight = batch['edge_weight'][0].to(device, non_blocking=True)
            coords = batch['coords'].to(device, non_blocking=True)

            pred_e, pred_n, pred_u = model(x, edge_index, edge_weight, coords)

            all_pred_e.append(pred_e.cpu().numpy())
            all_pred_n.append(pred_n.cpu().numpy())
            all_pred_u.append(pred_u.cpu().numpy())
            all_target_e.append(y[:, :, :, 0].cpu().numpy())
            all_target_n.append(y[:, :, :, 1].cpu().numpy())
            all_target_u.append(y[:, :, :, 2].cpu().numpy())

            pbar.update(1)
        pbar.close()

    pred_e = np.concatenate(all_pred_e, axis=0)
    pred_n = np.concatenate(all_pred_n, axis=0)
    pred_u = np.concatenate(all_pred_u, axis=0)
    target_e = np.concatenate(all_target_e, axis=0)
    target_n = np.concatenate(all_target_n, axis=0)
    target_u = np.concatenate(all_target_u, axis=0)

    test_metrics = GNSSMetrics.compute_all_metrics(pred_e, pred_n, pred_u, target_e, target_n, target_u)

    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)

    for direction in ['East', 'North', 'Up']:
        print(f"\n{direction} Direction:")
        metrics = test_metrics[direction]
        print(f"  MAE: {metrics['MAE'] * 1000:.2f} mm")
        print(f"  RMSE: {metrics['RMSE'] * 1000:.2f} mm")
        print(f"  R2: {metrics['R2']:.4f}")
        logger.info(
            f"{direction}: MAE={metrics['MAE'] * 1000:.2f}mm, "
            f"RMSE={metrics['RMSE'] * 1000:.2f}mm, R2={metrics['R2']:.4f}"
        )

    print(f"\n[Step 10/10] Saving Results")

    results = {
        'config': config,
        'test_metrics': test_metrics,
        'history': dict(history)
    }

    final_json_path = os.path.join(output_dir, 'final_results.json')
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=JSONEncoder)

    np.savez(
        os.path.join(output_dir, 'test_predictions.npz'),
        pred_east=pred_e, pred_north=pred_n, pred_up=pred_u,
        target_east=target_e, target_north=target_n, target_up=target_u
    )

    print(f"Results saved to: {output_dir}")
    print(f"  - best_model.pth (best model)")
    print(f"  - final_results.json (evaluation metrics)")
    print(f"  - test_predictions.npz (prediction results)")
    print(f"  - training_log_*.txt (training log)")

    if use_bayesian_opt:
        print(f"  - optuna_optimization_results.json (optimization history)")
        print(f"  - optimization_history.html (optimization curve)")
        print(f"  - param_importances.html (parameter importance)")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    return model, history, test_metrics


if __name__ == "__main__":
    print("\n[Step 3/10] Configuring Paths")

    FEATURE_DIR = r"FEATURE_DIR"
    OUTPUT_DIR = r"OUTPUT_DIR"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"  Feature directory: {FEATURE_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")

    if not os.path.exists(FEATURE_DIR):
        print(f"\nError: Feature directory does not exist: {FEATURE_DIR}")
        print("Please run feature engineering code first to generate feature data")
        sys.exit(1)

    logger = setup_training_logger(OUTPUT_DIR)

    model, history, test_metrics = main_training(
        feature_dir=FEATURE_DIR,
        output_dir=OUTPUT_DIR,
        use_bayesian_opt=True,
        logger=logger
    )

    print("\nAll tasks complete!")