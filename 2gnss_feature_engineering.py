# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import os
from datetime import datetime
from tqdm import tqdm
import logging

from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def setup_feature_logging(output_dir: str = "./features") -> logging.Logger:
    """Set up feature engineering logger"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"feature_engineering_log_{timestamp}.txt")

    logger = logging.getLogger("FeatureEngineering")
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class TemporalFeatureBuilder:
    """Time series feature builder"""

    def __init__(self,
                 window_sizes_days: List[int] = [7, 30, 60, 365],
                 adaptive_windows: bool = False,
                 add_coverage_features: bool = False,
                 logger: Optional[logging.Logger] = None):
        self.window_sizes_days = window_sizes_days
        self.adaptive_windows = adaptive_windows
        self.add_coverage_features = add_coverage_features
        self.logger = logger or logging.getLogger("FeatureEngineering")
        self._mjd0 = pd.Timestamp("1858-11-17")

    def build_temporal_features(self,
                                df: pd.DataFrame,
                                displacement_cols: List[str] = ['east', 'north', 'up'],
                                time_col: str = 'decimal_year',
                                history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build time series features"""
        df_feat = df.copy()

        if time_col not in df_feat.columns:
            raise ValueError(f"Missing time column: {time_col}")

        df_feat[time_col] = pd.to_numeric(df_feat[time_col], errors="coerce")
        df_feat = df_feat.sort_values(time_col).reset_index(drop=True)

        df_feat = self._prepare_datetime_index(df_feat, time_col)
        df_feat = self._add_time_features_minimal(df_feat, time_col)
        df_feat = self._add_rolling_statistics_robust(
            df_feat, displacement_cols, history_df=history_df
        )
        df_feat = self._add_difference_features_robust(df_feat, displacement_cols, time_col)
        df_feat = self._add_trend_features_time_based(df_feat, displacement_cols, time_col)
        df_feat = self._add_periodic_features_minimal(df_feat, time_col)

        df_feat = df_feat.reset_index(drop=True)
        return df_feat

    def _prepare_datetime_index(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Prepare datetime index"""
        dt = None

        if 'mjd' in df.columns:
            mjd = pd.to_numeric(df['mjd'], errors="coerce")
            if mjd.notna().any():
                dt = self._mjd0 + pd.to_timedelta(mjd, unit="D")

        if dt is None and 'YYMMMDD' in df.columns:
            ymd = df['YYMMMDD'].astype(str).str.upper()
            dt_try = pd.to_datetime(ymd, format="%y%b%d", errors="coerce")
            if dt_try.notna().any():
                dt = dt_try

        if dt is None:
            dy = pd.to_numeric(df[time_col], errors="coerce").astype("float64")
            year = np.floor(dy).astype("int32")
            frac = dy - np.floor(dy)
            base = pd.to_datetime(year.astype(str) + "-01-01", errors="coerce")
            dt = base + pd.to_timedelta(frac * 365.25, unit="D")

        df = df.copy()
        df["dt"] = dt
        if df["dt"].isna().any():
            df["dt"] = df["dt"].ffill().bfill()
            if df["dt"].isna().any():
                start = pd.Timestamp("2000-01-01")
                df["dt"] = start + pd.to_timedelta(np.arange(len(df)), unit="D")

        df = df.sort_values("dt").reset_index(drop=True)
        df = df.set_index("dt", drop=False)
        return df

    def _add_time_features_minimal(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Add basic time features"""
        df = df.copy()

        if 'dt' in df.columns and pd.api.types.is_datetime64_any_dtype(df['dt']):
            t0 = df['dt'].min()
            elapsed_days = (df['dt'] - t0).dt.total_seconds() / 86400.0
            df['time_elapsed_years'] = elapsed_days / 365.25
            df['year'] = df['dt'].dt.year.astype(float)
        else:
            time_start = pd.to_numeric(df[time_col], errors="coerce").min()
            df['time_elapsed_years'] = pd.to_numeric(df[time_col], errors="coerce") - time_start
            df['year'] = pd.to_numeric(df[time_col], errors="coerce").apply(lambda x: int(x) if pd.notna(x) else np.nan)

        return df

    def _add_rolling_statistics_robust(self, df: pd.DataFrame,
                                       displacement_cols: List[str],
                                       history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add rolling window statistics features, preserving history for computation"""
        disp_cols = [c for c in displacement_cols if c in df.columns]
        if not disp_cols:
            return df

        # Add temporary marker column to track df rows
        if history_df is not None:
            if not isinstance(history_df.index, pd.DatetimeIndex):
                history_df = self._prepare_datetime_index(history_df.copy(), 'decimal_year')

            # Mark current dataset rows
            df_marked = df.copy()
            df_marked['__temp_is_current__'] = True
            history_df['__temp_is_current__'] = False

            combined = pd.concat([history_df, df_marked], ignore_index=False)
            combined = combined.sort_index()
        else:
            combined = df.copy()
            combined['__temp_is_current__'] = True

        # Convert data to numeric format
        df_disp = combined[disp_cols].apply(pd.to_numeric, errors="coerce")

        new_frames = []

        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                for window_days in self.window_sizes_days:
                    min_periods = max(5, int(window_days / 20))  # Increase minimum periods
                    roll = df_disp.rolling(f"{window_days}D", min_periods=min_periods)

                    # Calculate mean
                    mean = roll.mean()
                    mean.columns = [f"{c}_roll_mean_{window_days}d" for c in disp_cols]

                    # Calculate standard deviation
                    std = roll.std(ddof=1)
                    std.columns = [f"{c}_roll_std_{window_days}d" for c in disp_cols]
                    # Replace possible invalid values with NaN
                    std = std.replace([np.inf, -np.inf], np.nan)

                    # Calculate quantiles
                    q25 = roll.quantile(0.25)
                    q25.columns = [f"{c}_roll_q25_{window_days}d" for c in disp_cols]

                    q75 = roll.quantile(0.75)
                    q75.columns = [f"{c}_roll_q75_{window_days}d" for c in disp_cols]

                    # Calculate interquartile range
                    iqr = (q75.values - q25.values)
                    iqr = pd.DataFrame(iqr, index=combined.index,
                                       columns=[f"{c}_roll_iqr_{window_days}d" for c in disp_cols])

                    new_frames.extend([mean, std, q25, q75, iqr])

        # Use marker column to extract current dataset rows
        current_mask = combined['__temp_is_current__']

        result_frames = []
        for frame in new_frames:
            # Extract rows marked as True and reset index
            result_frames.append(frame[current_mask].reset_index(drop=True))

        # Extract original df rows (without marker column) and reset index
        df_current = combined[current_mask].drop(columns=['__temp_is_current__']).reset_index(drop=True)

        # Merge dataset with rolling features
        out = pd.concat([df_current] + result_frames, axis=1)

        # Restore original index
        out.index = df.index

        return out

    def _add_difference_features_robust(
            self,
            df: pd.DataFrame,
            displacement_cols: List[str],
            time_col: str
    ) -> pd.DataFrame:
        """Add difference features (velocity, acceleration, etc.)"""
        df = df.copy()

        if 'dt' in df.columns and pd.api.types.is_datetime64_any_dtype(df['dt']):
            dt_years = df['dt'].diff().dt.total_seconds() / (86400.0 * 365.25)
        else:
            dt_years = pd.to_numeric(df[time_col], errors="coerce").diff()

        dt_years = dt_years.replace(0, np.nan)

        for col in displacement_cols:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce")

            diff1 = s.diff(1)
            df[f"{col}_diff1"] = diff1

            df[f"{col}_velocity"] = diff1 / dt_years

            df[f"{col}_diff2"] = s.diff(1).diff(1)

            vel = df[f"{col}_velocity"]
            df[f"{col}_acceleration"] = vel.diff(1) / dt_years

            scale_name = f"{col}_roll_std_30d"

            if scale_name in df.columns:
                rolling_scale = pd.to_numeric(df[scale_name], errors="coerce")

                fallback = float(np.nanstd(s.to_numpy(dtype="float64")))
                if (not np.isfinite(fallback)) or fallback <= 0:
                    fallback = 1.0
                rolling_scale = rolling_scale.replace(0, np.nan).fillna(fallback)
                rolling_scale = rolling_scale.clip(lower=1e-6)

                rel = diff1 / rolling_scale
            else:
                prev = s.shift(1).abs().clip(lower=1e-3)
                rel = diff1 / prev
            rel = rel.replace([np.inf, -np.inf], np.nan).clip(lower=-20, upper=20)
            df[f"{col}_relative_change"] = rel

        return df

    @staticmethod
    def _rolling_slope(times: np.ndarray, values: np.ndarray, window_years: float, min_points: int = 3) -> np.ndarray:
        """Calculate rolling window slope"""
        t = np.asarray(times, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        n = len(t)

        out = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return out

        valid = np.isfinite(y) & np.isfinite(t)
        ones = valid.astype(np.float64)
        t_valid = np.where(valid, t, 0.0)
        y_valid = np.where(valid, y, 0.0)
        tt_valid = np.where(valid, t * t, 0.0)
        ty_valid = np.where(valid, t * y, 0.0)

        p1 = np.zeros(n + 1, dtype=np.float64)
        pt = np.zeros(n + 1, dtype=np.float64)
        py = np.zeros(n + 1, dtype=np.float64)
        ptt = np.zeros(n + 1, dtype=np.float64)
        pty = np.zeros(n + 1, dtype=np.float64)

        p1[1:] = np.cumsum(ones)
        pt[1:] = np.cumsum(t_valid)
        py[1:] = np.cumsum(y_valid)
        ptt[1:] = np.cumsum(tt_valid)
        pty[1:] = np.cumsum(ty_valid)

        for i in range(n):
            start_time = t[i] - window_years
            start = np.searchsorted(t, start_time, side="right")
            end = i + 1

            if start >= end:
                continue

            cnt = p1[end] - p1[start]
            if cnt < min_points:
                continue

            sum_t = pt[end] - pt[start]
            sum_y = py[end] - py[start]
            sum_tt = ptt[end] - ptt[start]
            sum_ty = pty[end] - pty[start]

            denom = cnt * sum_tt - sum_t * sum_t
            if denom == 0.0:
                continue

            out[i] = (cnt * sum_ty - sum_t * sum_y) / denom

        return out

    def _add_trend_features_time_based(self, df: pd.DataFrame,
                                       displacement_cols: List[str],
                                       time_col: str) -> pd.DataFrame:
        """Add trend features"""
        trend_windows_days = [30, 60, 365]
        times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)

        for col in displacement_cols:
            if col not in df.columns:
                continue

            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

            for window_days in trend_windows_days:
                window_years = window_days / 365.25
                slopes = self._rolling_slope(times, y, window_years, min_points=3)
                df[f'{col}_trend_slope_{window_days}d'] = slopes

        return df

    def _add_periodic_features_minimal(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Add periodic features (annual and semi-annual cycles)"""
        t = pd.to_numeric(df[time_col], errors="coerce").astype("float64")
        df['sin_annual'] = np.sin(2 * np.pi * t)
        df['cos_annual'] = np.cos(2 * np.pi * t)
        df['sin_semiannual'] = np.sin(4 * np.pi * t)
        df['cos_semiannual'] = np.cos(4 * np.pi * t)
        return df


class SpatialFeatureBuilder:
    """Spatial feature builder"""

    def __init__(self, n_clusters: int = 10, logger: Optional[logging.Logger] = None):
        self.n_clusters = n_clusters
        self.logger = logger or logging.getLogger("FeatureEngineering")
        self.station_coords = None
        self.cluster_model = None

    def extract_station_coordinates(self, full_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract station coordinates"""
        coord_list = []
        for station, df in full_data_dict.items():
            if len(df) == 0:
                continue

            row = df.iloc[0]
            coord_dict = {'station': station}

            if 'latitude' in df.columns and 'longitude' in df.columns:
                coord_dict['latitude'] = row['latitude']
                coord_dict['longitude'] = row['longitude']
            elif 'e0' in df.columns and 'n0' in df.columns:
                coord_dict['latitude'] = row.get('n0', 0)
                coord_dict['longitude'] = row.get('e0', 0)
            else:
                continue

            coord_list.append(coord_dict)

        self.station_coords = pd.DataFrame(coord_list)
        self.logger.info(f"Extracted station coordinates: {len(self.station_coords)}")
        return self.station_coords

    def compute_spatial_clusters(self) -> Dict[str, int]:
        """Compute spatial clusters"""
        if self.station_coords is None:
            raise ValueError("Please call extract_station_coordinates first")

        coords = self.station_coords[['latitude', 'longitude']].values
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = self.cluster_model.fit_predict(coords)

        self.station_coords['cluster'] = labels
        cluster_dict = dict(zip(self.station_coords['station'], labels))

        self.logger.info(f"Spatial clustering completed: K={self.n_clusters}")
        return cluster_dict

    def compute_morans_i(self, full_data_dict: Dict[str, pd.DataFrame],
                         displacement_col: str = 'east',
                         k_neighbors: int = 5) -> float:
        """Calculate Moran's I spatial autocorrelation index"""
        if self.station_coords is None or len(self.station_coords) < 2:
            return 0.0

        station_means = {}
        for station, df in full_data_dict.items():
            if displacement_col in df.columns:
                mean_val = pd.to_numeric(df[displacement_col], errors="coerce").mean()
                if not np.isnan(mean_val):
                    station_means[station] = mean_val

        if len(station_means) < 2:
            return 0.0

        valid_stations = [s for s in self.station_coords['station'] if s in station_means]
        coords_df = self.station_coords[self.station_coords['station'].isin(valid_stations)].copy()

        coords_rad = np.radians(coords_df[['latitude', 'longitude']].values)
        tree = BallTree(coords_rad, metric='haversine')
        _, indices = tree.query(coords_rad, k=min(k_neighbors + 1, len(coords_rad)))

        n = len(valid_stations)
        values = np.array([station_means[s] for s in coords_df['station']])
        mean_val = values.mean()

        numerator = 0.0
        W = 0.0

        for i in range(n):
            for j in range(1, min(k_neighbors + 1, len(indices[i]))):
                neighbor_idx = indices[i][j]
                weight = 1.0
                W += weight
                numerator += weight * (values[i] - mean_val) * (values[neighbor_idx] - mean_val)

        denominator = np.sum((values - mean_val) ** 2)
        if denominator == 0 or W == 0:
            return 0.0

        return (n / W) * (numerator / denominator)


class AdvancedFeatureSelector:
    """Advanced feature selector"""

    def __init__(self,
                 missing_threshold: float = 0.3,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.98,
                 importance_threshold: float = 0.001,
                 stability_threshold: float = 0.5,
                 min_keep_per_target: int = 8,
                 cross_dir_importance_multiplier: float = 2.0,
                 cross_dir_stability_multiplier: float = 2.0,
                 logger: Optional[logging.Logger] = None):

        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.stability_threshold = stability_threshold

        self.min_keep_per_target = int(min_keep_per_target)
        self.cross_dir_importance_multiplier = float(cross_dir_importance_multiplier)
        self.cross_dir_stability_multiplier = float(cross_dir_stability_multiplier)

        self.logger = logger or logging.getLogger("FeatureEngineering")

        self.selected_features = []
        self.feature_importance_by_target: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, float] = {}
        self.removed_features_log: Dict[str, List[Tuple[str, str]]] = {}

    def select_features(self,
                        train_data: pd.DataFrame,
                        target_cols: List[str],
                        exclude_cols: List[str],
                        station_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """Execute feature selection"""
        self.logger.info("=" * 80)
        self.logger.info("Starting feature selection")

        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [c for c in numeric_cols if c not in exclude_cols and c not in target_cols]
        self.logger.info(f"Candidate features: {len(candidate_features)}")

        features = self._filter_missing(train_data, candidate_features)
        self.logger.info(f"After missing rate filter: {len(features)}")

        features = self._filter_low_variance(train_data, features)
        self.logger.info(f"After variance filter: {len(features)}")

        features = self._filter_high_correlation(train_data, features)
        self.logger.info(f"After correlation filter: {len(features)}")

        train_data_clean = self._handle_outliers(train_data, features)

        per_target_selected = self._filter_by_importance_per_target(train_data_clean, features, target_cols)
        union_feats = sorted(set().union(*per_target_selected.values())) if per_target_selected else []
        self.logger.info(f"After importance screening: {len(union_feats)}")

        if station_data is not None and union_feats:
            union_feats = self._filter_by_stability_per_target(station_data, union_feats, target_cols)
            self.logger.info(f"After stability screening: {len(union_feats)}")

        self.selected_features = union_feats
        self._build_global_importance(target_cols)

        self.logger.info(f"Final retained features: {len(self.selected_features)}")
        self.logger.info("=" * 80)
        return self.selected_features

    def _filter_missing(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Filter features with high missing rates"""
        kept, removed = [], []
        n = max(len(df), 1)
        for feat in features:
            mr = df[feat].isna().sum() / n
            if mr <= self.missing_threshold:
                kept.append(feat)
            else:
                removed.append((feat, f"missing_rate={mr:.2%}"))
        self.removed_features_log['missing'] = removed
        return kept

    def _filter_low_variance(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Filter low variance features"""
        kept, removed = [], []
        for feat in features:
            n_unique = df[feat].nunique(dropna=True)
            if n_unique <= 1:
                removed.append((feat, f"constant(n_unique={n_unique})"))
                continue
            var = df[feat].var(skipna=True)
            if pd.isna(var) or var < self.variance_threshold:
                removed.append((feat, f"low_variance={var}"))
                continue
            kept.append(feat)
        self.removed_features_log['low_variance'] = removed
        return kept

    def _filter_high_correlation(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Filter highly correlated features"""
        if len(features) == 0:
            return features

        corr_matrix = df[features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        removed = []
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            high_corr = upper_tri[column] > self.correlation_threshold
            if high_corr.any():
                correlated_features = upper_tri.index[high_corr].tolist()
                for feat in correlated_features:
                    to_drop.add(feat)
                    removed.append((feat, f"high_corr_with_{column}(r>{self.correlation_threshold})"))

        kept = [f for f in features if f not in to_drop]
        self.removed_features_log['high_correlation'] = removed
        return kept

    def _handle_outliers(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Handle outliers in features"""
        df_clean = df.copy()
        for feat in features:
            inf_mask = np.isinf(df_clean[feat])
            if inf_mask.any():
                df_clean.loc[inf_mask, feat] = np.nan

            if df_clean[feat].notna().any():
                q_low = df_clean[feat].quantile(0.05)
                q_high = df_clean[feat].quantile(0.95)
                df_clean[feat] = df_clean[feat].clip(lower=q_low, upper=q_high)

        return df_clean

    def _filter_by_importance_per_target(self,
                                         df: pd.DataFrame,
                                         features: List[str],
                                         target_cols: List[str]) -> Dict[str, List[str]]:
        """Filter features based on importance"""
        X = df[features].copy()
        X = X.fillna(X.median(numeric_only=True))

        selected: Dict[str, List[str]] = {}
        removed_all = []

        self.feature_importance_by_target = {}

        for target in target_cols:
            if target not in df.columns:
                continue

            y = pd.to_numeric(df[target], errors="coerce")
            y = y.fillna(y.median())

            rf = RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )

            try:
                rf.fit(X, y)
            except Exception:
                selected[target] = []
                self.feature_importance_by_target[target] = {}
                continue

            imp = rf.feature_importances_
            imp_map = {f: float(v) for f, v in zip(features, imp)}
            self.feature_importance_by_target[target] = imp_map

            base_th = self.importance_threshold
            cross_th = self.importance_threshold * self.cross_dir_importance_multiplier

            keep = []
            for f, v in imp_map.items():
                if self._is_same_direction_feature(f, target):
                    if v >= base_th:
                        keep.append(f)
                    else:
                        removed_all.append((f, f"{target}: low_importance={v:.6f}"))
                else:
                    if v >= cross_th:
                        keep.append(f)
                    else:
                        removed_all.append((f, f"{target}: cross_dir_low_importance={v:.6f}"))

            keep = self._ensure_min_features_for_target(target, keep, imp_map)
            selected[target] = sorted(set(keep))

        self.removed_features_log['low_importance_per_target'] = removed_all
        return selected

    def _ensure_min_features_for_target(self,
                                        target: str,
                                        kept: List[str],
                                        imp_map: Dict[str, float]) -> List[str]:
        """Ensure minimum number of features retained for each target"""
        need = max(0, self.min_keep_per_target - len(set(kept)))
        if need == 0:
            return kept

        cross_soft_th = (self.importance_threshold * self.cross_dir_importance_multiplier) * 0.5

        same = [(f, v) for f, v in imp_map.items() if self._is_same_direction_feature(f, target)]
        same.sort(key=lambda x: x[1], reverse=True)

        cross = [(f, v) for f, v in imp_map.items() if not self._is_same_direction_feature(f, target)]
        cross.sort(key=lambda x: x[1], reverse=True)

        kept_set = set(kept)

        for f, v in same:
            if f in kept_set:
                continue
            if v <= 0:
                continue
            kept_set.add(f)
            need -= 1
            if need == 0:
                return list(kept_set)

        for f, v in cross:
            if f in kept_set:
                continue
            if v >= cross_soft_th:
                kept_set.add(f)
                need -= 1
                if need == 0:
                    break

        return list(kept_set)

    def _is_same_direction_feature(self, feat_name: str, target: str) -> bool:
        """Check if feature is in same direction as target"""
        if feat_name.startswith("east_"):
            return target == "east"
        if feat_name.startswith("north_"):
            return target == "north"
        if feat_name.startswith("up_"):
            return target == "up"
        return True

    def _filter_by_stability_per_target(self,
                                        station_data: Dict[str, pd.DataFrame],
                                        features: List[str],
                                        target_cols: List[str]) -> List[str]:
        """Filter features based on stability"""
        if len(station_data) < 3 or len(features) == 0:
            return features

        per_target_feat_imps: Dict[str, Dict[str, List[float]]] = {
            t: {f: [] for f in features} for t in target_cols
        }

        for _, sdf in station_data.items():
            if sdf is None or len(sdf) < 50:
                continue

            feats_here = [f for f in features if f in sdf.columns]
            if not feats_here:
                continue

            Xs = sdf[feats_here].copy()
            Xs = Xs.fillna(Xs.median(numeric_only=True))

            for t in target_cols:
                if t not in sdf.columns:
                    continue

                ys = pd.to_numeric(sdf[t], errors="coerce")
                ys = ys.fillna(ys.median())

                rf = RandomForestRegressor(
                    n_estimators=40,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                try:
                    rf.fit(Xs, ys)
                except Exception:
                    continue

                for f, imp in zip(feats_here, rf.feature_importances_):
                    per_target_feat_imps[t][f].append(float(imp))

        removed = []
        per_target_kept = {}

        for t in target_cols:
            base_th = self.importance_threshold
            cross_th = self.importance_threshold * self.cross_dir_stability_multiplier

            base_stab = self.stability_threshold
            cross_stab = min(0.95, self.stability_threshold * self.cross_dir_stability_multiplier)

            kept_t = []
            for f in features:
                imps = per_target_feat_imps.get(t, {}).get(f, [])
                if len(imps) == 0:
                    removed.append((f, f"{t}: no_station_importance"))
                    continue

                th = base_th if self._is_same_direction_feature(f, t) else cross_th
                ratio = float(np.mean([v > th for v in imps]))

                stab_req = base_stab if self._is_same_direction_feature(f, t) else cross_stab
                if ratio >= stab_req:
                    kept_t.append(f)
                else:
                    removed.append((f, f"{t}: unstable(ratio={ratio:.2%} < {stab_req:.2%})"))

            kept_t = self._ensure_min_stable_for_target(t, kept_t, per_target_feat_imps[t])
            per_target_kept[t] = set(kept_t)

        self.removed_features_log['unstable_per_target'] = removed
        union = sorted(set().union(*per_target_kept.values())) if per_target_kept else features
        return union

    def _ensure_min_stable_for_target(self,
                                      target: str,
                                      kept: List[str],
                                      feat_imps: Dict[str, List[float]]) -> List[str]:
        """Ensure minimum number of stable features retained for each target"""
        need = max(0, self.min_keep_per_target - len(set(kept)))
        if need == 0:
            return kept

        base_th = self.importance_threshold
        cross_th = self.importance_threshold * self.cross_dir_stability_multiplier

        scores_same = []
        scores_cross = []
        for f, imps in feat_imps.items():
            if len(imps) == 0:
                continue
            th = base_th if self._is_same_direction_feature(f, target) else cross_th
            ratio = float(np.mean([v > th for v in imps]))
            if self._is_same_direction_feature(f, target):
                scores_same.append((f, ratio))
            else:
                scores_cross.append((f, ratio))

        scores_same.sort(key=lambda x: x[1], reverse=True)
        scores_cross.sort(key=lambda x: x[1], reverse=True)

        kept_set = set(kept)

        for f, _ in scores_same:
            if f in kept_set:
                continue
            kept_set.add(f)
            need -= 1
            if need == 0:
                return list(kept_set)

        for f, r in scores_cross:
            if f in kept_set:
                continue
            if r >= self.stability_threshold:
                kept_set.add(f)
                need -= 1
                if need == 0:
                    break

        return list(kept_set)

    def _build_global_importance(self, target_cols: List[str]) -> None:
        """Build global feature importance"""
        global_map = {}
        for t in target_cols:
            imp_map = self.feature_importance_by_target.get(t, {})
            for f, v in imp_map.items():
                global_map[f] = max(global_map.get(f, 0.0), float(v))
        self.feature_importance = global_map

    def save_selection_report(self, output_dir: str):
        """Save feature selection report"""
        report_path = os.path.join(output_dir, "feature_selection_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Feature Selection Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("Filtering Statistics:\n")
            f.write("-" * 80 + "\n")
            for layer, removed_list in self.removed_features_log.items():
                f.write(f"{layer}: Removed {len(removed_list)} features\n")
            f.write(f"\nFinally retained: {len(self.selected_features)} features\n\n")

            for layer, removed_list in self.removed_features_log.items():
                if len(removed_list) > 0:
                    f.write(f"\n{layer} - Partial removal reasons (Top 20):\n")
                    f.write("-" * 80 + "\n")
                    for feat, reason in removed_list[:20]:
                        f.write(f"  {feat}: {reason}\n")
                    if len(removed_list) > 20:
                        f.write(f"  ... and {len(removed_list) - 20} more\n")

            if self.feature_importance:
                f.write("\n\nFeature Importance (Top 30, max over targets):\n")
                f.write("-" * 80 + "\n")
                sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_importance[:30]:
                    f.write(f"  {feat}: {imp:.6f}\n")

        self.logger.info(f"Feature selection report saved: {report_path}")


class FeatureEngineer:
    """Main feature engineering class"""

    def __init__(self,
                 preprocessed_dir: str,
                 output_dir: str,
                 window_sizes_days: List[int] = [7, 30, 60, 365],
                 n_clusters: int = 10,
                 **selector_kwargs):

        self.preprocessed_dir = preprocessed_dir
        self.output_dir = output_dir
        self.window_sizes_days = window_sizes_days
        os.makedirs(output_dir, exist_ok=True)

        self.logger = setup_feature_logging(output_dir)

        self.temporal_builder = TemporalFeatureBuilder(
            window_sizes_days=window_sizes_days,
            logger=self.logger
        )

        self.spatial_builder = SpatialFeatureBuilder(
            n_clusters=n_clusters,
            logger=self.logger
        )

        self.feature_selector = AdvancedFeatureSelector(
            logger=self.logger,
            **selector_kwargs
        )

        self.displacement_cols = ['east', 'north', 'up']

    def load_preprocessed_data(self) -> Tuple[Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame]]:
        """Load preprocessed data"""
        self.logger.info("=" * 80)
        self.logger.info("Loading preprocessed data...")

        train_data = {}
        val_data = {}
        test_data = {}

        stations = [d for d in os.listdir(self.preprocessed_dir)
                    if os.path.isdir(os.path.join(self.preprocessed_dir, d))]

        for station in tqdm(stations, desc="Loading station data"):
            station_dir = os.path.join(self.preprocessed_dir, station)

            train_path = os.path.join(station_dir, 'train.csv')
            val_path = os.path.join(station_dir, 'val.csv')
            test_path = os.path.join(station_dir, 'test.csv')

            if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
                train_data[station] = pd.read_csv(train_path, low_memory=False)
                val_data[station] = pd.read_csv(val_path, low_memory=False)
                test_data[station] = pd.read_csv(test_path, low_memory=False)

        self.logger.info(f"Number of stations: {len(train_data)}")
        self.logger.info(f"Training samples: {sum(len(df) for df in train_data.values())}")
        self.logger.info(f"Validation samples: {sum(len(df) for df in val_data.values())}")
        self.logger.info(f"Test samples: {sum(len(df) for df in test_data.values())}")

        return train_data, val_data, test_data

    def build_all_features(self) -> Dict:
        self.logger.info("=" * 80)
        self.logger.info("Starting Feature Engineering")
        self.logger.info("=" * 80)

        train_data, val_data, test_data = self.load_preprocessed_data()

        self.logger.info("Computing spatial clusters")
        station_coords = self.spatial_builder.extract_station_coordinates(train_data)
        cluster_labels = self.spatial_builder.compute_spatial_clusters()

        self.logger.info("Computing Moran's I")
        morans_dict = {}
        for col in self.displacement_cols:
            morans_dict[col] = self.spatial_builder.compute_morans_i(
                train_data,
                displacement_col=col,
                k_neighbors=5
            )
        self.logger.info(
            f"Moran's I: east={morans_dict['east']:.4f}, "
            f"north={morans_dict['north']:.4f}, up={morans_dict['up']:.4f}"
        )

        self.logger.info("Initializing feature builder")
        enhanced_builder = TemporalFeatureBuilder(
            window_sizes_days=self.window_sizes_days,
            adaptive_windows=True,  # Enable adaptive windows
            add_coverage_features=True,  # Enable coverage features
            logger=self.logger
        )

        self.logger.info("Building temporal features")

        featured_train = {}
        featured_val = {}
        featured_test = {}

        from tqdm import tqdm
        for station in tqdm(train_data.keys(), desc="Building features"):
            # Add station and cluster information to all datasets
            for df in [train_data[station], val_data[station], test_data[station]]:
                if 'station' not in df.columns:
                    df['station'] = station
                if station in cluster_labels:
                    df['spatial_cluster'] = cluster_labels[station]

            self.logger.debug(f"Processing training set - station {station}")
            featured_train[station] = enhanced_builder.build_temporal_features(
                train_data[station].copy(),
                self.displacement_cols,
                history_df=None
            )
            featured_train[station], train_stats = self._check_and_fix_data_quality(
                featured_train[station]
            )

            self.logger.debug(f"Processing validation set - station {station}")
            featured_val[station] = enhanced_builder.build_temporal_features(
                val_data[station].copy(),
                self.displacement_cols,
                history_df=train_data[station]
            )
            featured_val[station], _ = self._check_and_fix_data_quality(
                featured_val[station],
                train_stats
            )

            self.logger.debug(f"Processing test set - station {station}")
            train_val_combined = pd.concat([
                train_data[station],
                val_data[station]
            ], ignore_index=True).sort_values('decimal_year').reset_index(drop=True)

            featured_test[station] = enhanced_builder.build_temporal_features(
                test_data[station].copy(),
                self.displacement_cols,
                history_df=train_val_combined
            )
            featured_test[station], _ = self._check_and_fix_data_quality(
                featured_test[station],
                train_stats
            )

        # ========================================================================
        # Feature quality report
        # ========================================================================
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Feature Quality Statistics")
        self.logger.info("=" * 80)

        sample_station = list(featured_train.keys())[0]
        all_features = [c for c in featured_train[sample_station].columns
                        if c not in ['station', 'decimal_year', 'mjd'] + self.displacement_cols]

        self.logger.info(f"Total features: {len(all_features)}")
        self.logger.info(f"  - Rolling window features: {sum(1 for f in all_features if 'roll' in f)}")
        self.logger.info(
            f"  - Difference/velocity features: {sum(1 for f in all_features if any(k in f for k in ['diff', 'velocity', 'acceleration', 'momentum']))}")
        self.logger.info(f"  - Trend features: {sum(1 for f in all_features if 'trend' in f)}")
        self.logger.info(f"  - Periodic features: {sum(1 for f in all_features if any(k in f for k in ['sin', 'cos']))}")
        self.logger.info(f"  - Coverage features: {sum(1 for f in all_features if 'coverage' in f)}")
        self.logger.info(f"  - Adaptive features: {sum(1 for f in all_features if 'adaptive' in f)}")

        # Check NaN ratios
        for split_name, split_data in [('Training set', featured_train),
                                       ('Validation set', featured_val),
                                       ('Test set', featured_test)]:
            all_df = pd.concat([split_data[s] for s in split_data.keys()], ignore_index=True)

            # Check NaN ratio in rolling features
            roll_features = [f for f in all_features if 'roll_mean' in f]
            if roll_features:
                nan_ratios = all_df[roll_features].isna().mean()
                max_nan = nan_ratios.max()
                avg_nan = nan_ratios.mean()
                self.logger.info(f"{split_name} - Rolling mean features NaN ratio: avg={avg_nan:.2%}, max={max_nan:.2%}")

        featured_split = {}
        for station in featured_train.keys():
            featured_split[station] = {
                'train': featured_train[station],
                'val': featured_val[station],
                'test': featured_test[station],
                'full': pd.concat([
                    featured_train[station],
                    featured_val[station],
                    featured_test[station]
                ], ignore_index=True)
            }

        self.logger.info("\nFeature selection")
        all_train = pd.concat([featured_train[s] for s in featured_train.keys()], ignore_index=True)
        station_train = {s: featured_train[s] for s in featured_train.keys()}

        exclude_cols = ['station', 'decimal_year', 'mjd'] + self.displacement_cols
        selected_features = self.feature_selector.select_features(
            all_train,
            target_cols=self.displacement_cols,
            exclude_cols=exclude_cols,
            station_data=station_train
        )

        self.logger.info("Saving results")
        self._save_featured_data(featured_split, selected_features)
        self.feature_selector.save_selection_report(self.output_dir)

        self.logger.info("Generating comprehensive report")
        self._generate_comprehensive_report(featured_split, selected_features, morans_dict)

        self.logger.info("=" * 80)
        self.logger.info("Feature Engineering Completed")
        self.logger.info("=" * 80)

        return {
            'featured_data': featured_split,
            'selected_features': selected_features,
            'station_coords': station_coords,
            'morans_i': morans_dict,
            'feature_importance': self.feature_selector.feature_importance
        }

    def _check_and_fix_data_quality(self, df: pd.DataFrame,
                                    reference_stats: Optional[Dict] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """Check and fix data quality issues"""
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df, None

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        exclude = {'decimal_year', 'mjd', 'year'}
        cols_check = [c for c in numeric_cols if c not in exclude]

        if reference_stats is None:
            stats = {}
            for c in cols_check:
                q_low = df[c].quantile(0.005)
                q_high = df[c].quantile(0.995)
                stats[c] = {'q_low': q_low, 'q_high': q_high}
                df[c] = df[c].clip(lower=q_low, upper=q_high)
            return df, stats

        else:
            for c in cols_check:
                if c in reference_stats:
                    df[c] = df[c].clip(
                        lower=reference_stats[c]['q_low'],
                        upper=reference_stats[c]['q_high']
                    )
            return df, None

    def _save_featured_data(self, featured_split: Dict, selected_features: List[str]):
        """Save featured data"""
        for station in tqdm(featured_split.keys(), desc="Saving featured data"):
            station_dir = os.path.join(self.output_dir, station)
            os.makedirs(station_dir, exist_ok=True)

            for split_name, df in featured_split[station].items():
                cols_to_save = (
                        ['station', 'decimal_year'] +
                        self.displacement_cols +
                        [f for f in selected_features if f in df.columns]
                )
                save_path = os.path.join(station_dir, f'{split_name}_featured.csv')
                df[cols_to_save].to_csv(save_path, index=False)

        feature_list_path = os.path.join(self.output_dir, 'selected_features.txt')
        with open(feature_list_path, 'w', encoding='utf-8') as f:
            for feat in selected_features:
                f.write(f"{feat}\n")

        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Retained features: {len(selected_features)}")

    def _generate_comprehensive_report(self, featured_split: Dict,
                                       selected_features: List[str],
                                       morans_dict: Dict):
        """Generate comprehensive report"""
        report_path = os.path.join(self.output_dir, 'feature_engineering_comprehensive_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GNSS Time Series Prediction Feature Engineering Comprehensive Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-" * 80 + "\n")
            f.write("Data Statistics\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of stations: {len(featured_split)}\n")
            f.write(f"Selected features: {len(selected_features)}\n")

            total_train = sum(len(featured_split[s]['train']) for s in featured_split)
            total_val = sum(len(featured_split[s]['val']) for s in featured_split)
            total_test = sum(len(featured_split[s]['test']) for s in featured_split)

            f.write(f"\nTraining samples: {total_train}\n")
            f.write(f"Validation samples: {total_val}\n")
            f.write(f"Test samples: {total_test}\n\n")

            f.write("-" * 80 + "\n")
            f.write("Spatial Autocorrelation (Moran's I)\n")
            f.write("-" * 80 + "\n")
            for col, value in morans_dict.items():
                f.write(f"  {col}: {value:.4f}\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("Selected Feature Categories (by keywords)\n")
            f.write("-" * 80 + "\n")

            temporal_feats = [ft for ft in selected_features if any(
                kw in ft for kw in ['roll', 'diff', 'trend', 'sin', 'cos',
                                    'time', 'velocity', 'acceleration', 'year']
            )]
            spatial_feats = [ft for ft in selected_features if any(
                kw in ft for kw in ['cluster', 'latitude', 'longitude', 'spatial', 'height']
            )]
            other_feats = [ft for ft in selected_features if ft not in temporal_feats and ft not in spatial_feats]

            f.write(f"\nTemporal features: {len(temporal_feats)}\n")
            f.write(f"Spatial features: {len(spatial_feats)}\n")
            f.write(f"Other features: {len(other_feats)}\n\n")

            all_train = pd.concat([featured_split[s]['train'] for s in featured_split], ignore_index=True)
            missing_rates = {}
            for feat in selected_features:
                if feat in all_train.columns:
                    missing_rates[feat] = all_train[feat].isnull().sum() / max(len(all_train), 1)

            high_missing = [(ft, r) for ft, r in missing_rates.items() if r > 0.05]
            if high_missing:
                f.write("Features with missing rate >5% (Top 10):\n")
                for feat, rate in sorted(high_missing, key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"  {feat}: {rate:.2%}\n")
            else:
                f.write("Missing rate check: All <=5%\n")

        self.logger.info(f"Comprehensive report saved: {report_path}")


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("GNSS Time Series Data Feature Engineering")
    print("=" * 80)

    preprocessed_dir = r"preprocessed_dir"
    output_dir = r"output_dir"

    print("\nInitializing...")
    engineer = FeatureEngineer(
        preprocessed_dir=preprocessed_dir,
        output_dir=output_dir,
        window_sizes_days=[7, 30, 60, 365],
        n_clusters=10,
        missing_threshold=0.3,
        variance_threshold=0.01,
        correlation_threshold=0.98,
        importance_threshold=0.001,
        stability_threshold=0.5,
        min_keep_per_target=8,
        cross_dir_importance_multiplier=2.0,
        cross_dir_stability_multiplier=2.0
    )

    print("\nRunning...")
    results = engineer.build_all_features()

    print("\n" + "=" * 80)
    print("Completed")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of stations: {len(results['featured_data'])}")
    print(f"Retained features: {len(results['selected_features'])}")
    print("\nMoran's I:")
    for col, value in results['morans_i'].items():
        print(f"  {col}: {value:.4f}")

    print("\nReport files:")
    print("  - feature_engineering_comprehensive_report.txt")
    print("  - feature_selection_report.txt")
    print("  - feature_engineering_log_*.txt")


if __name__ == "__main__":
    main()