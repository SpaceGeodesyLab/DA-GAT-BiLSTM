import os
import pickle
import sys
from typing import Optional
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cartopy.feature import ShapelyFeature

# ---------------------------------------------------------------------------
# ── Hardcoded paths ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

BASE_DIR      = r"C:\Users\朱思宇\Desktop\a"
NE_DIR        = os.path.join(BASE_DIR, "ne_shapefiles")   # 本地 shapefile 目录
CKPT_PATH     = os.path.join(BASE_DIR, "model.pt")
STATIONS_CSV  = os.path.join(BASE_DIR, "stations.csv")
LOADER_PKL    = os.path.join(BASE_DIR, "loader.pkl")
FAULT_GEOJSON = os.path.join(BASE_DIR, "san_andreas_fault.geojson")

OUT_MAP   = os.path.join(BASE_DIR, "fig_attention_san_andreas.png")
OUT_STATS = os.path.join(BASE_DIR, "attention_stats.csv")
OUT_FAULT = os.path.join(BASE_DIR, "attention_vs_fault.csv")

REGION         = "san_andreas"
TOP_PERCENTILE = 20   # show top 20 % of attention edges
N_BATCHES      = 10   # number of time-window batches to average over

# ---------------------------------------------------------------------------
# ── model.py must be in the same directory ───────────────────────────────────
# ---------------------------------------------------------------------------
sys.path.insert(0, BASE_DIR)
from model import DAGATBiLSTM   # noqa: E402


# ---------------------------------------------------------------------------
# 0. Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")
    return device


# ---------------------------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> DAGATBiLSTM:
    print(f"[load_model] Loading checkpoint: {ckpt_path}")
    model = DAGATBiLSTM.from_pretrained(ckpt_path, device=str(device))
    model.to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"[load_model] Parameters: {total:,}  |  GAT layers: {len(model.gat_layers)}")
    return model


# ---------------------------------------------------------------------------
# 2. Extract attention weights from a single batch
# ---------------------------------------------------------------------------

def get_attention_weights(model, batch, device, layer_idx=-1):
    with torch.no_grad():
        x          = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        n_layers   = len(model.gat_layers)
        target_k   = n_layers + layer_idx

        for k, gat in enumerate(model.gat_layers):
            if k == target_k:
                _, (ei, alpha) = gat(
                    x, edge_index, return_attention_weights=True
                )
                return ei.cpu().numpy(), alpha.cpu().numpy()
            x = gat(x, edge_index)

    raise RuntimeError(f"layer_idx={layer_idx} not reached (n_layers={n_layers}).")


# ---------------------------------------------------------------------------
# 3. Aggregate attention across multiple time-window batches
# ---------------------------------------------------------------------------

def aggregate_attention(loader, model, device, n_batches=10, layer_idx=-1):
    print(f"[aggregate_attention] Aggregating over {n_batches} batches …")
    all_alpha  = []
    edge_index = None

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ei, alpha = get_attention_weights(model, batch, device, layer_idx)
        if edge_index is None:
            edge_index = ei
        all_alpha.append(alpha.mean(axis=1))
        print(f"  batch {i+1}/{n_batches}  edges={ei.shape[1]}  "
              f"alpha_range=[{alpha.min():.4f}, {alpha.max():.4f}]")

    if edge_index is None:
        raise RuntimeError("Loader is empty – no batches found.")

    alpha_mean = np.mean(all_alpha, axis=0)
    print(f"[aggregate_attention] Done.  alpha_mean: "
          f"min={alpha_mean.min():.4f}  max={alpha_mean.max():.4f}")
    return edge_index, alpha_mean


# ---------------------------------------------------------------------------
# 辅助：从本地 shapefile 构建 cartopy feature（完全不联网）
# ---------------------------------------------------------------------------

def _local_feature(shp_name: str, **kwargs) -> ShapelyFeature:
    """
    从 NE_DIR 目录读取 Natural Earth shapefile，返回 cartopy ShapelyFeature。
    shp_name 示例: 'ne_10m_land.shp'
    """
    shp_path = os.path.join(NE_DIR, shp_name)
    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"本地 shapefile 不存在: {shp_path}\n"
            f"请将 Natural Earth 10m shapefile 放入: {NE_DIR}"
        )
    reader = shapereader.Reader(shp_path)
    geometries = list(reader.geometries())
    return ShapelyFeature(
        geometries,
        ccrs.PlateCarree(),
        **kwargs
    )


# ---------------------------------------------------------------------------
# 4. Map visualisation
# ---------------------------------------------------------------------------

def plot_attention_map(
    stations: pd.DataFrame,
    edge_index: np.ndarray,
    alpha_mean: np.ndarray,
    region: str = "san_andreas",
    top_percentile: int = 20,
    fault_geojson: Optional[str] = None,
    out_path: str = "fig_attention_san_andreas.png",
):
    """
    Fix [3]: region_ids 使用 ORIGINAL 行号，不先 reset_index。
    Fix [网络]: 所有地图要素改为从本地 shapefile 读取，不触发任何下载。
    """
    if region == "san_andreas":
        extent = [-125, -114, 32, 42]
        title  = "Learned GAT attention weights — San Andreas region"
    elif region == "anatolia":
        extent = [25, 45, 35, 43]
        title  = "Learned GAT attention weights — Anatolia region"
    else:
        raise ValueError(f"Unknown region: '{region}'")

    lon_min, lon_max, lat_min, lat_max = extent

    # Fix [3]: 保留原始索引
    mask = (
        (stations.lon > lon_min) & (stations.lon < lon_max) &
        (stations.lat > lat_min) & (stations.lat < lat_max)
    )
    region_ids      = set(stations[mask].index.tolist())
    region_stations = stations[mask].reset_index(drop=True)
    print(f"[plot_attention_map] Stations in region: {len(region_ids)}")

    # 过滤 region 内的边
    src, dst  = edge_index[0], edge_index[1]
    in_region = np.array(
        [int(s) in region_ids and int(d) in region_ids
         for s, d in zip(src, dst)],
        dtype=bool
    )
    src_r, dst_r, alpha_r = src[in_region], dst[in_region], alpha_mean[in_region]

    if len(alpha_r) == 0:
        print("[plot_attention_map] WARNING: no edges found in region.")
        return
    print(f"[plot_attention_map] Region edges: {in_region.sum()}")

    # 只保留 top-percentile
    thr  = np.percentile(alpha_r, 100 - top_percentile)
    keep = alpha_r >= thr
    src_r, dst_r, alpha_r = src_r[keep], dst_r[keep], alpha_r[keep]
    print(f"[plot_attention_map] Edges after top-{top_percentile}% threshold: {keep.sum()}")

    # ── 构建图形 ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 9))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # ── 本地地图要素（完全不联网）────────────────────────────────────────────
    try:
        ax.add_feature(
            _local_feature("ne_10m_land.shp", facecolor="#f5f5f5", zorder=0)
        )
    except FileNotFoundError as e:
        print(f"[plot_attention_map] WARNING: {e}")

    try:
        ax.add_feature(
            _local_feature("ne_10m_ocean.shp", facecolor="#d6e9f5", zorder=0)
        )
    except FileNotFoundError as e:
        print(f"[plot_attention_map] WARNING: {e}")

    try:
        ax.add_feature(
            _local_feature("ne_10m_coastline.shp",
                           facecolor="none", edgecolor="black",
                           linewidth=0.5, zorder=1)
        )
    except FileNotFoundError as e:
        print(f"[plot_attention_map] WARNING: {e}")

    try:
        ax.add_feature(
            _local_feature("ne_10m_admin_1_states_provinces_lakes.shp",
                           facecolor="none", edgecolor="gray",
                           linewidth=0.3, zorder=1)
        )
    except FileNotFoundError as e:
        print(f"[plot_attention_map] WARNING: {e}")

    try:
        ax.add_feature(
            _local_feature("ne_10m_admin_0_boundary_lines_land.shp",
                           facecolor="none", edgecolor="gray",
                           linewidth=0.4, zorder=1)
        )
    except FileNotFoundError as e:
        print(f"[plot_attention_map] WARNING: {e}")

    ax.gridlines(draw_labels=True, linewidth=0.3,
                 color="gray", alpha=0.5, linestyle="--")

    # ── 断层线（可选）────────────────────────────────────────────────────────
    if fault_geojson is not None and os.path.exists(fault_geojson):
        import geopandas as gpd
        faults = gpd.read_file(fault_geojson)
        faults.plot(
            ax=ax, color="crimson", linewidth=1.2, alpha=0.85,
            transform=ccrs.PlateCarree(), label="Mapped faults (USGS)",
            zorder=2
        )
        print(f"[plot_attention_map] Fault traces loaded: {len(faults)} features")
    elif fault_geojson is not None:
        print(f"[plot_attention_map] WARNING: fault file not found: {fault_geojson}")

    # ── 注意力边 ─────────────────────────────────────────────────────────────
    norm = mcolors.Normalize(vmin=alpha_r.min(), vmax=alpha_r.max())
    cmap = plt.cm.plasma

    for s, d, a in zip(src_r, dst_r, alpha_r):
        lat_s = stations.iloc[int(s)]["lat"]
        lon_s = stations.iloc[int(s)]["lon"]
        lat_d = stations.iloc[int(d)]["lat"]
        lon_d = stations.iloc[int(d)]["lon"]
        ax.plot(
            [lon_s, lon_d], [lat_s, lat_d],
            color=cmap(norm(a)),
            linewidth=0.4 + 2.5 * norm(a),
            alpha=0.78,
            transform=ccrs.PlateCarree(),
            zorder=3
        )

    # ── 台站标记 ─────────────────────────────────────────────────────────────
    ax.scatter(
        region_stations.lon, region_stations.lat,
        s=40, c="black", marker="^",
        transform=ccrs.PlateCarree(), zorder=4,
        edgecolors="white", linewidths=0.7,
        label="GNSS station"
    )

    if "name" in region_stations.columns:
        for _, row in region_stations.iterrows():
            ax.text(
                row.lon + 0.05, row.lat + 0.05, row["name"],
                fontsize=4, color="#333333",
                transform=ccrs.PlateCarree(), zorder=5
            )

    # ── 色条 ─────────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      shrink=0.70, pad=0.07, aspect=30)
    cb.set_label(f"Mean GAT attention weight (top {top_percentile} %)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_attention_map] Saved: {out_path}")


# ---------------------------------------------------------------------------
# 5. Save per-edge statistics CSV
# ---------------------------------------------------------------------------

def save_attention_stats(stations, edge_index, alpha_mean, out_path):
    src, dst = edge_index[0], edge_index[1]

    def haversine_km(lat1, lon1, lat2, lon2):
        R    = 6371.0
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    rows = []
    for s, d, a in zip(src, dst, alpha_mean):
        rs = stations.iloc[int(s)]
        rd = stations.iloc[int(d)]
        rows.append({
            "src_name"    : rs.get("name", int(s)),
            "dst_name"    : rd.get("name", int(d)),
            "src_lat"     : rs.lat,
            "src_lon"     : rs.lon,
            "dst_lat"     : rd.lat,
            "dst_lon"     : rd.lon,
            "alpha_mean"  : float(a),
            "edge_dist_km": float(haversine_km(rs.lat, rs.lon, rd.lat, rd.lon)),
        })

    df = pd.DataFrame(rows).sort_values("alpha_mean", ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[save_attention_stats] Saved: {out_path}  ({len(df)} edges)")
    return df


# ---------------------------------------------------------------------------
# 6. Attention vs fault distance
# ---------------------------------------------------------------------------

def attention_vs_fault(stations, edge_index, alpha_mean, fault_geojson, out_path):
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    if not os.path.exists(fault_geojson):
        print(f"[attention_vs_fault] Fault file not found: {fault_geojson}")
        return None, None

    faults = gpd.read_file(fault_geojson)

    # Fix [4]: 同时处理 LineString 和 MultiLineString
    fault_pts = []
    for geom in faults.geometry:
        if geom is None:
            continue
        if isinstance(geom, LineString):
            fault_pts.extend(list(geom.coords))
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                fault_pts.extend(list(line.coords))
        else:
            try:
                fault_pts.extend(list(geom.coords))
            except (AttributeError, NotImplementedError):
                pass

    if not fault_pts:
        print("[attention_vs_fault] WARNING: no fault coordinates extracted.")
        return None, None

    fault_lonlat = np.array(fault_pts, dtype=np.float64)
    fault_tree   = cKDTree(fault_lonlat)

    src, dst = edge_index[0], edge_index[1]
    midpoints = np.array([
        [
            (stations.iloc[int(s)].lon + stations.iloc[int(d)].lon) / 2.0,
            (stations.iloc[int(s)].lat + stations.iloc[int(d)].lat) / 2.0,
        ]
        for s, d in zip(src, dst)
    ])

    dist_deg, _ = fault_tree.query(midpoints, k=1)
    dist_km     = dist_deg * 111.0

    df = pd.DataFrame({
        "src"             : src,
        "dst"             : dst,
        "alpha"           : alpha_mean,
        "dist_to_fault_km": dist_km,
    })
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[attention_vs_fault] Saved: {out_path}")

    rho, p = (lambda r: (r.statistic, r.pvalue))(
        spearmanr(df.alpha, df.dist_to_fault_km)
    )
    print(f"[attention_vs_fault] Spearman ρ = {rho:.4f} p = {p:.3e}")
    if rho < 0:
        print("  → Negative correlation: model attends MORE near the fault  ✓")
    else:
        print("  → No negative correlation detected.")
    return rho, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("DA-GAT-BiLSTM Attention Visualisation")
    print("=" * 65)

    device = get_device()

    print(f"\n[main] Reading stations: {STATIONS_CSV}")
    stations = pd.read_csv(STATIONS_CSV).reset_index(drop=True)
    print(f"       {len(stations)} stations loaded  (columns: {list(stations.columns)})")

    model = load_model(CKPT_PATH, device)

    print(f"\n[main] Loading loader: {LOADER_PKL}")
    with open(LOADER_PKL, "rb") as f:
        loader = pickle.load(f)
    print(f"       {len(loader)} batches")

    print()
    edge_index, alpha_mean = aggregate_attention(
        loader=loader, model=model, device=device,
        n_batches=N_BATCHES, layer_idx=-1,
    )

    print()
    save_attention_stats(stations, edge_index, alpha_mean, OUT_STATS)

    print()
    plot_attention_map(
        stations=stations, edge_index=edge_index, alpha_mean=alpha_mean,
        region=REGION, top_percentile=TOP_PERCENTILE,
        fault_geojson=FAULT_GEOJSON, out_path=OUT_MAP,
    )

    print()
    attention_vs_fault(
        stations=stations, edge_index=edge_index, alpha_mean=alpha_mean,
        fault_geojson=FAULT_GEOJSON, out_path=OUT_FAULT,
    )

    print("\n" + "=" * 65)
    print("All outputs written to:", BASE_DIR)
    print("  ·", os.path.basename(OUT_MAP))
    print("  ·", os.path.basename(OUT_STATS))
    print("  ·", os.path.basename(OUT_FAULT))
    print("=" * 65)


if __name__ == "__main__":
    main()