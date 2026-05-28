# -*- coding: utf-8 -*-
"""GNSS tenv3 preprocessing pipeline."""

import glob
import os
import re
import traceback
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CONFIG = {
    "INPUT_DIR": r"INPUT_DIR",
    "OUTPUT_DIR": r"OUTPUT_DIR",

    "STATION_FILES": None,
    "MAX_STATIONS": None,
    "FILE_PATTERN": "*.txt",

    "TRAIN_END": "2015-01-01",
    "VAL_END": "2020-01-01",

    "DISPLACEMENT_UNIT": "m",
    "WRITE_FULL_ARCHIVE": True,

    "JUMP_WIN": 60,
    "JUMP_WIN_STEP": 15,
    "JUMP_K": 3.5,
    "JUMP_MIN_MM": 5.0,
    "JUMP_MERGE": 30,
    "JUMP_MAX_ITER": 8,

    "JUMP_RESIDUAL_PASS": True,
    "JUMP_RESID_K": 3.0,
    "JUMP_RESID_MIN_MM": 3.0,

    "IQR_K": 1.5,
    "MAX_SHORT_GAP": 30,
    "TARGET_MAX_MISS": 2.0,
    "ROLL_WIN": 30,
    "CENTER_ON_TRAIN_MEAN": True,
}

COMPONENTS = ("E", "N", "U")
COMP_OUT = {"E": "east", "N": "north", "U": "up"}
SIG_COL = {"E": "sigE", "N": "sigN", "U": "sigU"}
MJD0 = pd.Timestamp("1858-11-17")

PIPELINE_COLS = [
    "station",
    "decimal_year",
    "mjd",
    "east",
    "north",
    "up",
    "latitude",
    "longitude",
    "height",
]


def parse_tenv3(filepath: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Parse one NGL tenv3 text file. Displacements and sigmas are returned in mm."""
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=1,
        header=None,
        engine="python",
        comment=None,
    )

    def col(idx: int) -> pd.Series:
        if idx in df.columns:
            return pd.to_numeric(df[idx], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    mjd = col(3).round().astype("Int64")
    date = MJD0 + pd.to_timedelta(mjd.astype("float64"), unit="D")

    out = pd.DataFrame(
        {
            "date": date.values,
            "decimal_year": col(2).values,
            "mjd": mjd.astype("float64").values,
            "E": (col(7) + col(8)).values * 1000.0,
            "N": (col(9) + col(10)).values * 1000.0,
            "U": (col(11) + col(12)).values * 1000.0,
            "sigE": col(14).values * 1000.0,
            "sigN": col(15).values * 1000.0,
            "sigU": col(16).values * 1000.0,
        }
    )

    out = (
        out.dropna(subset=["date"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    coords = {
        "latitude": float(np.nanmedian(col(20))) if 20 in df.columns else np.nan,
        "longitude": float(np.nanmedian(col(21))) if 21 in df.columns else np.nan,
        "height": float(np.nanmedian(col(22))) if 22 in df.columns else np.nan,
    }
    return out, coords


def to_daily_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Reindex observations to a continuous daily grid."""
    full_idx = pd.date_range("2000-01-01", df["date"].max(), freq="D")
    grid = df.set_index("date").reindex(full_idx)
    grid.index.name = "date"
    missing_ratio = float(grid[list(COMPONENTS)].isna().any(axis=1).mean())
    return grid, missing_ratio


def to_decimal_year(dt_index: Iterable[pd.Timestamp]) -> np.ndarray:
    dt = pd.DatetimeIndex(dt_index)
    years = dt.year.to_numpy()
    start = pd.to_datetime(pd.Series(years).astype(str) + "-01-01").to_numpy()
    end = pd.to_datetime((pd.Series(years) + 1).astype(str) + "-01-01").to_numpy()
    return years + (dt.to_numpy() - start) / (end - start)


def to_mjd(dt_index: Iterable[pd.Timestamp]) -> np.ndarray:
    return (pd.DatetimeIndex(dt_index) - MJD0).days.astype("int64")


def natkey(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def mad(x: Iterable[float]) -> float:
    values = np.asarray(x, dtype="float64")
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(values - np.median(values))) + 1e-9


def detect_jumps_single_pass(
    y: np.ndarray,
    win_det: int,
    jump_k: float,
    jump_min_mm: float,
    jump_merge: int,
    exclude_positions: Optional[set] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Detect jump candidates without modifying the input sequence."""
    n = len(y)
    exclude_positions = exclude_positions or set()

    yv = y.copy()
    global_mad = mad(yv)
    global_median = np.nanmedian(yv)
    if global_mad > 0 and not np.isnan(global_mad):
        yv[np.abs(yv - global_median) > 8 * global_mad] = np.nan

    diff_score = np.full(n, np.nan)
    raw_step = np.full(n, np.nan)

    for idx in range(win_det, n - win_det):
        before = yv[idx - win_det:idx]
        after = yv[idx:idx + win_det]
        if np.sum(~np.isnan(before)) < win_det * 0.4:
            continue
        if np.sum(~np.isnan(after)) < win_det * 0.4:
            continue

        step = np.nanmedian(after) - np.nanmedian(before)
        local_mad = np.mean([mad(before), mad(after)])
        if np.isnan(local_mad) or local_mad <= 0:
            continue

        diff_score[idx] = step / local_mad
        raw_step[idx] = step

    valid = (np.abs(diff_score) > jump_k) & (np.abs(raw_step) > jump_min_mm)
    candidates = [
        idx for idx in np.where(valid)[0]
        if not any(abs(idx - prev) <= jump_merge for prev in exclude_positions)
    ]
    if not candidates:
        return [], diff_score, raw_step

    groups: List[List[int]] = [[candidates[0]]]
    for idx in candidates[1:]:
        if idx - groups[-1][-1] <= jump_merge:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    jumps = [group[int(np.argmax(np.abs(diff_score[group])))] for group in groups]
    jumps.sort(key=lambda idx: -abs(raw_step[idx]) if not np.isnan(raw_step[idx]) else 0)
    return jumps, diff_score, raw_step


def remove_jumps_iterative(
    series: pd.Series,
    dates_arr: np.ndarray,
    cfg: Dict,
    jump_k: Optional[float] = None,
    jump_min_mm: Optional[float] = None,
) -> Tuple[pd.Series, List[pd.Timestamp]]:
    """Apply iterative jump correction and return corrected values plus jump dates."""
    y = np.asarray(series, dtype="float64").copy()
    n = len(y)
    if np.sum(~np.isnan(y)) < 4 * cfg["JUMP_WIN"]:
        return pd.Series(y, index=series.index), []

    win_det = cfg["JUMP_WIN"]
    win_step = cfg.get("JUMP_WIN_STEP", max(14, win_det // 3))
    k = cfg["JUMP_K"] if jump_k is None else jump_k
    min_mm = cfg["JUMP_MIN_MM"] if jump_min_mm is None else jump_min_mm

    jump_dates: List[pd.Timestamp] = []
    corrected_positions = set()

    for _ in range(cfg.get("JUMP_MAX_ITER", 8)):
        jumps, _, _ = detect_jumps_single_pass(
            y,
            win_det,
            k,
            min_mm,
            cfg["JUMP_MERGE"],
            exclude_positions=corrected_positions,
        )
        if not jumps:
            break

        updated = False
        for jump_idx in jumps:
            before = y[max(0, jump_idx - win_step):jump_idx]
            after = y[jump_idx:min(n, jump_idx + win_step)]
            if np.sum(~np.isnan(before)) < max(3, win_step // 4):
                continue
            if np.sum(~np.isnan(after)) < max(3, win_step // 4):
                continue

            step = np.nanmedian(after) - np.nanmedian(before)
            if np.isnan(step) or abs(step) < min_mm * 0.5:
                continue

            y[jump_idx:] -= step
            corrected_positions.add(jump_idx)
            jump_dates.append(pd.Timestamp(dates_arr[jump_idx]))
            updated = True

        if not updated:
            break

    unique_dates = sorted({date.strftime("%Y-%m-%d"): date for date in jump_dates}.values())
    return pd.Series(y, index=series.index), unique_dates


def rolling_baseline(values: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Build a robust annual rolling-median baseline."""
    return (
        pd.Series(values, index=dates)
        .rolling(365, center=True, min_periods=60)
        .median()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )


def fill_gaps_controlled(
    series: pd.Series,
    baseline: pd.Series,
    cfg: Dict,
) -> Tuple[pd.Series, np.ndarray]:
    """Fill short gaps first, then fill the smallest remaining gaps until the target is met."""
    target_ratio = cfg.get("TARGET_MAX_MISS", 2.0) / 100.0
    interpolated = series.interpolate(
        method="time",
        limit=cfg["MAX_SHORT_GAP"],
        limit_area="inside",
    )
    fill_flag = (series.isna() & interpolated.notna()).to_numpy(copy=True)
    arr = interpolated.to_numpy(dtype="float64", copy=True)

    if np.isnan(arr).mean() <= target_ratio:
        return pd.Series(arr, index=series.index), fill_flag

    gaps = []
    is_missing = np.isnan(arr)
    idx = 0
    while idx < len(is_missing):
        if not is_missing[idx]:
            idx += 1
            continue
        end = idx
        while end < len(is_missing) and is_missing[end]:
            end += 1
        gaps.append((idx, end, end - idx))
        idx = end

    for start, end, _ in sorted(gaps, key=lambda item: item[2]):
        if np.isnan(arr).mean() <= target_ratio:
            break
        arr[start:end] = baseline.values[start:end]
        fill_flag[start:end] = True

    return pd.Series(arr, index=series.index), fill_flag


def process_component(
    grid: pd.DataFrame,
    dates: pd.DatetimeIndex,
    train_mask: np.ndarray,
    component: str,
    cfg: Dict,
) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    """Process one displacement component."""
    out_name = COMP_OUT[component]
    y0 = grid[component].copy()

    # Jump correction.
    y_nojump, jump_dates = remove_jumps_iterative(y0, dates.values, cfg)
    jump_set = {date.strftime("%Y-%m-%d") for date in jump_dates}

    # Baseline and residual cleanup.
    baseline = rolling_baseline(y_nojump.values, dates)
    resid = y_nojump.values - baseline.values

    if cfg.get("JUMP_RESIDUAL_PASS", True):
        residual_corrected, residual_jumps = remove_jumps_iterative(
            pd.Series(resid, index=dates),
            dates.values,
            cfg,
            jump_k=cfg.get("JUMP_RESID_K", 4.0),
            jump_min_mm=cfg.get("JUMP_RESID_MIN_MM", 3.0),
        )
        y_nojump = pd.Series(y_nojump.values + residual_corrected.values - resid, index=dates)
        baseline = rolling_baseline(y_nojump.values, dates)
        resid = y_nojump.values - baseline.values
        jump_set.update(date.strftime("%Y-%m-%d") for date in residual_jumps)

    reference_resid = resid[train_mask & ~np.isnan(resid)]
    if reference_resid.size < 30:
        reference_resid = resid[~np.isnan(resid)]

    q1, q3 = np.percentile(reference_resid, [25, 75])
    iqr = q3 - q1
    low = q1 - cfg["IQR_K"] * iqr
    high = q3 + cfg["IQR_K"] * iqr
    is_outlier = (resid < low) | (resid > high)

    cleaned = y_nojump.values.copy()
    cleaned[is_outlier] = np.nan
    filled, fill_flag = fill_gaps_controlled(pd.Series(cleaned, index=dates), baseline, cfg)

    if cfg["CENTER_ON_TRAIN_MEAN"]:
        ref = np.nanmean(filled.values[train_mask])
        if np.isnan(ref):
            ref = np.nanmean(filled.values)
        filled = filled - ref
        raw_display = y0.values - ref
    else:
        raw_display = y0.values

    component_df = pd.DataFrame(
        {
            out_name: filled.values,
            f"{out_name}_raw": raw_display,
            f"{out_name}_filled": fill_flag,
            f"{out_name}_sig": grid[SIG_COL[component]].values,
        },
        index=dates,
    )

    raw_std = (
        pd.Series(y0.values, index=dates)
        .rolling(cfg["ROLL_WIN"], min_periods=cfg["ROLL_WIN"] // 2)
        .std()
    )
    proc_std = (
        pd.Series(filled.values, index=dates)
        .rolling(cfg["ROLL_WIN"], min_periods=cfg["ROLL_WIN"] // 2)
        .std()
    )
    mean_raw = np.nanmean(raw_std.values)
    mean_proc = np.nanmean(proc_std.values)
    std_drop = 100 * (1 - mean_proc / mean_raw) if mean_raw and not np.isnan(mean_raw) else np.nan

    metrics = {
        f"{component}_n_jumps": len(jump_set),
        f"{component}_n_outliers": int(np.nansum(is_outlier)),
        f"{component}_RollSTD_raw": round(mean_raw, 3),
        f"{component}_RollSTD_proc": round(mean_proc, 3),
        f"{component}_RollSTD_drop_%": round(std_drop, 2),
    }
    return component_df, metrics, sorted(jump_set)


def unit_factor(cfg: Dict) -> float:
    unit = str(cfg.get("DISPLACEMENT_UNIT", "m")).lower()
    if unit not in {"m", "mm"}:
        raise ValueError("DISPLACEMENT_UNIT must be either 'm' or 'mm'.")
    return 1e-3 if unit == "m" else 1.0


def preprocess_station(filepath: str, cfg: Dict) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    name = os.path.splitext(os.path.basename(filepath))[0]
    raw, coords = parse_tenv3(filepath)
    if len(raw) < 200:
        return None, None

    grid, missing_before = to_daily_grid(raw)
    dates = grid.index
    train_mask = np.asarray(dates < pd.Timestamp(cfg["TRAIN_END"]))
    if train_mask.sum() < 200:
        fallback_count = int(len(dates) * 0.6)
        train_mask = np.zeros(len(dates), dtype=bool)
        train_mask[:fallback_count] = True

    out = pd.DataFrame(index=dates)
    out.index.name = "date"

    report = {
        "station": name,
        "n_raw_obs": len(raw),
        "n_grid_days": len(dates),
        "missing_before_%": round(100 * missing_before, 3),
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "height": coords["height"],
    }
    all_jump_dates = {}

    # Component-level preprocessing.
    for component in COMPONENTS:
        component_df, metrics, jumps = process_component(grid, dates, train_mask, component, cfg)
        out = out.join(component_df)
        report.update(metrics)
        all_jump_dates[component] = jumps

    factor = unit_factor(cfg)
    if factor != 1.0:
        for component in COMPONENTS:
            out_name = COMP_OUT[component]
            out[out_name] *= factor
            out[f"{out_name}_raw"] *= factor
            out[f"{out_name}_sig"] *= factor

    processed_cols = [COMP_OUT[component] for component in COMPONENTS]
    missing_after = float(out[processed_cols].isna().any(axis=1).mean())
    report["missing_after_%"] = round(100 * missing_after, 3)
    report["missing_drop_pts"] = round(100 * (missing_before - missing_after), 3)
    report["jump_dates"] = str(all_jump_dates)

    # Metadata for downstream feature engineering.
    split = np.where(
        dates < pd.Timestamp(cfg["TRAIN_END"]),
        "train",
        np.where(dates < pd.Timestamp(cfg["VAL_END"]), "val", "test"),
    )
    out.insert(0, "decimal_year", to_decimal_year(dates))
    out.insert(1, "mjd", to_mjd(dates))
    out.insert(2, "split", split)
    out["latitude"] = coords["latitude"]
    out["longitude"] = coords["longitude"]
    out["height"] = coords["height"]
    out["station"] = name

    return out.reset_index(), report


def write_station_outputs(station_dir: str, full_df: pd.DataFrame, cfg: Dict) -> None:
    os.makedirs(station_dir, exist_ok=True)
    station_name = full_df["station"].iloc[0]

    for split_name in ("train", "val", "test"):
        split_df = full_df[full_df["split"] == split_name]
        cols = [col for col in PIPELINE_COLS if col in split_df.columns]
        split_df[cols].to_csv(
            os.path.join(station_dir, f"{split_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if cfg.get("WRITE_FULL_ARCHIVE", True):
        full_df.to_csv(
            os.path.join(station_dir, f"{station_name}_full.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def collect_input_files(cfg: Dict) -> List[str]:
    if cfg["STATION_FILES"]:
        files = [os.path.join(cfg["INPUT_DIR"], filename) for filename in cfg["STATION_FILES"]]
    else:
        files = sorted(
            glob.glob(os.path.join(cfg["INPUT_DIR"], cfg["FILE_PATTERN"])),
            key=lambda path: natkey(os.path.basename(path)),
        )

    if cfg["MAX_STATIONS"]:
        return files[:cfg["MAX_STATIONS"]]
    return files


def print_station_summary(index: int, total: int, name: str, full_df: pd.DataFrame, report: Dict) -> None:
    train_count = int((full_df["split"] == "train").sum())
    val_count = int((full_df["split"] == "val").sum())
    test_count = int((full_df["split"] == "test").sum())

    print(
        f"[{index}/{total}] {name}: "
        f"missing {report['missing_before_%']}% -> {report['missing_after_%']}% | "
        f"split train/val/test = {train_count}/{val_count}/{test_count} | "
        f"RollSTD drop E {report['E_RollSTD_drop_%']}%, "
        f"N {report['N_RollSTD_drop_%']}%, "
        f"U {report['U_RollSTD_drop_%']}% | "
        f"jumps E {report['E_n_jumps']}, "
        f"N {report['N_n_jumps']}, "
        f"U {report['U_n_jumps']}"
    )

    if val_count == 0 or test_count == 0:
        print(f"    Warning: {name} has an empty validation or test split. Check TRAIN_END/VAL_END.")


def save_quality_report(reports: List[Dict], output_dir: str) -> None:
    if not reports:
        print("No valid station was processed.")
        return

    report_df = pd.DataFrame(reports)
    report_path = os.path.join(output_dir, "quality_report.csv")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nQuality report saved to: {report_path}")

    summary_cols = [
        "station",
        "latitude",
        "longitude",
        "height",
        "missing_before_%",
        "missing_after_%",
        "E_RollSTD_drop_%",
        "N_RollSTD_drop_%",
        "U_RollSTD_drop_%",
        "E_n_jumps",
        "N_n_jumps",
        "U_n_jumps",
    ]
    summary_cols = [col for col in summary_cols if col in report_df.columns]
    print(report_df[summary_cols].to_string(index=False))


def main(cfg: Dict = CONFIG) -> List[Dict]:
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    files = collect_input_files(cfg)
    reports = []

    for index, filepath in enumerate(files, start=1):
        station_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            full_df, report = preprocess_station(filepath, cfg)
            if full_df is None or report is None:
                print(f"[{index}/{len(files)}] {station_name}: too few observations, skipped.")
                continue

            station_dir = os.path.join(cfg["OUTPUT_DIR"], station_name)
            write_station_outputs(station_dir, full_df, cfg)
            reports.append(report)
            print_station_summary(index, len(files), station_name, full_df, report)

        except Exception as exc:
            print(f"[{index}/{len(files)}] {station_name}: failed -> {exc}")
            traceback.print_exc()

    save_quality_report(reports, cfg["OUTPUT_DIR"])
    return reports


if __name__ == "__main__":
    main()