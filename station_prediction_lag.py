"""
Station prediction visualization and lag diagnostics.

Required CSV columns:
    时间
    东向真实值(m), 北向真实值(m), 垂直向真实值(m)
    东向预测值(m), 北向预测值(m), 垂直向预测值(m)

Generated files:
    fig_station_prediction.png
    fig_lag_diagnostic.png
    lag_table.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.stats import pearsonr

plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

REQUIRED_COLUMNS = [
    '时间',
    '东向真实值(m)', '北向真实值(m)', '垂直向真实值(m)',
    '东向预测值(m)', '北向预测值(m)', '垂直向预测值(m)',
]


def decimal_year_to_dates(dy: np.ndarray) -> pd.DatetimeIndex:
    """Convert decimal years to dates."""
    year = dy.astype(int)
    frac = dy - year
    days_in_year = np.where(year % 4 == 0, 366, 365)
    day_of_year = (frac * days_in_year).astype(int)
    return pd.to_datetime(
        [f'{y}-{d + 1:03d}' for y, d in zip(year, day_of_year)],
        format='%Y-%j'
    )


def clean_path(path: str) -> str:
    """Remove common quote characters around a pasted file path."""
    return path.strip().strip('"').strip("'")


def resolve_csv_path(csv_arg: Optional[str]) -> str:
    """Resolve the CSV path from the command line or interactive input."""
    if csv_arg:
        csv_path = clean_path(csv_arg)
    else:
        print('CSV path was not provided.')
        try:
            csv_path = clean_path(input('Enter CSV file path: '))
        except EOFError as exc:
            raise SystemExit(
                'CSV path was not provided. Example:\n'
                'python station_prediction_lag.py --csv "C:/path/to/prediction.csv"'
            ) from exc

    if not csv_path:
        raise SystemExit('CSV path is empty.')
    if not os.path.isfile(csv_path):
        raise SystemExit(f'CSV file not found: {csv_path}')
    return csv_path


def load_csv(path: str):
    """Load dates, observed values, and predicted values from the CSV file."""
    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError('Missing required columns: ' + ', '.join(missing_columns))

    dates = decimal_year_to_dates(df['时间'].values)
    obs = df[['东向真实值(m)', '北向真实值(m)', '垂直向真实值(m)']].values * 1000
    pred = df[['东向预测值(m)', '北向预测值(m)', '垂直向预测值(m)']].values * 1000
    print(f'Loaded CSV: {path}')
    print(f'Rows loaded: {len(df)}')
    return dates, obs, pred


def plot_station(dates, obs, pred, station_name: str, out_path: str):
    """Create the observed-versus-predicted time-series figure."""
    comps = ['East (E)', 'North (N)', 'Up (U)']
    units = 'mm'

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.patch.set_facecolor('#f8f8f8')

    for i, ax in enumerate(axes):
        ax.set_facecolor('white')
        ax.plot(dates, obs[:, i], color='#222222', lw=0.6,
                label='Observed', alpha=0.85, zorder=3)
        ax.plot(dates, pred[:, i], color='#d63031', lw=0.8,
                label='Predicted', alpha=0.85, zorder=4)
        ax.fill_between(dates, obs[:, i], pred[:, i],
                        color='#d63031', alpha=0.10, zorder=2)

        r, _ = pearsonr(obs[:, i], pred[:, i])
        rmse = np.sqrt(np.mean((obs[:, i] - pred[:, i]) ** 2))
        mae = np.mean(np.abs(obs[:, i] - pred[:, i]))

        ax.set_ylabel(f'{comps[i]} ({units})', fontsize=10)
        ax.text(
            0.01, 0.96,
            f'r = {r:.3f}   RMSE = {rmse:.3f} mm   MAE = {mae:.3f} mm',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='#cccccc',
                      alpha=0.9, boxstyle='round,pad=0.3')
        )
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=9)
    axes[0].set_title(
        f'Station {station_name} — Test period prediction',
        fontsize=12, fontweight='bold', pad=8
    )
    axes[-1].set_xlabel('Date', fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def lag_diagnostic(obs, pred, max_lag: int = 10) -> dict:
    """Calculate normalized cross-correlation within the selected lag window."""
    results = {}
    for i, c in enumerate(['E', 'N', 'U']):
        o = obs[:, i] - obs[:, i].mean()
        p = pred[:, i] - pred[:, i].mean()

        ccf = correlate(o, p, mode='full')
        ccf = ccf / (np.std(o) * np.std(p) * len(o))
        lags = np.arange(-len(o) + 1, len(o))

        mask = (lags >= -max_lag) & (lags <= max_lag)
        ccf_window = ccf[mask]
        lags_window = lags[mask]

        peak_lag = lags_window[np.argmax(ccf_window)]
        peak_value = ccf_window.max()
        results[c] = {
            'lags': lags_window,
            'ccf': ccf_window,
            'peak_lag': int(peak_lag),
            'peak_value': float(peak_value),
        }
    return results


def plot_lag_diagnostic(lag_results: dict, out_path: str):
    """Create the lag diagnostic figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.patch.set_facecolor('#f8f8f8')

    for i, (c, res) in enumerate(lag_results.items()):
        ax = axes[i]
        ax.set_facecolor('white')
        markerline, stemlines, _ = ax.stem(
            res['lags'], res['ccf'],
            basefmt=' ', linefmt=f'C{i}-', markerfmt=f'C{i}o'
        )
        markerline.set_markersize(4)
        stemlines.set_linewidth(0.8)
        ax.axvline(0, color='#636e72', linestyle='--',
                   linewidth=0.9, alpha=0.7)
        ax.axvline(
            res['peak_lag'], color='#d63031', linestyle=':', linewidth=1.2,
            label=f"peak lag = {res['peak_lag']} d\nCCF = {res['peak_value']:.3f}"
        )
        ax.set_title(f'Component {c}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Lag (days)', fontsize=9)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Normalized CCF', fontsize=10)
    fig.suptitle(
        'Cross-correlation: observed vs predicted series',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def save_lag_table(lag_results: dict, station_name: str, out_path: str):
    """Save the peak lag summary table."""
    rows = [
        {
            'station': station_name,
            'component': c,
            'peak_lag_days': res['peak_lag'],
            'peak_ccf': res['peak_value'],
        }
        for c, res in lag_results.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    print('Lag summary:')
    print(df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize station prediction results and lag diagnostics.'
    )
    parser.add_argument('--csv', default=None,
                        help='Path to the prediction result CSV file.')
    parser.add_argument('--station', default='Station-1',
                        help='Station name used in the figure title.')
    parser.add_argument('--outdir', default='.',
                        help='Output directory.')
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Maximum lag in days for diagnostics.')
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates, obs, pred = load_csv(csv_path)
    plot_station(
        dates, obs, pred,
        station_name=args.station,
        out_path=str(outdir / 'fig_station_prediction.png')
    )
    lag_results = lag_diagnostic(obs, pred, max_lag=args.max_lag)
    plot_lag_diagnostic(
        lag_results,
        out_path=str(outdir / 'fig_lag_diagnostic.png')
    )
    save_lag_table(
        lag_results,
        station_name=args.station,
        out_path=str(outdir / 'lag_table.csv')
    )
    print('Done.')


if __name__ == '__main__':
    main()
