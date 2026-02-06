import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIRS = [
    Path('haul/performance_resuts'),
    Path('plot/results'),
    Path('haul/performance_results'),
]
BATCH_SIZE = 64  # Update if you used a different batch size for the scan.

DECODE_MODES = {
    "decode_on_demand": {
        "label": "Decode On Demand",
        "colors": {
            "prefetch": "#2980b9",
            "no_prefetch": "#3498db",
            "base": "#2980b9",
        },
    },
    "decode_all": {
        "label": "Decode All",
        "colors": {
            "prefetch": "#c0392b",
            "no_prefetch": "#e74c3c",
            "base": "#c0392b",
        },
    },
}

powers_of_2 = [1, 2, 4, 8, 16, 32, 64]
linear_ticks = [1, 8, 16, 24, 32, 40, 48, 56, 64]


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')


def _setup_log_x(ax: plt.Axes) -> None:
    ax.set_xscale('log', base=2)
    ax.set_xticks(powers_of_2)
    ax.set_xticklabels([str(x) for x in powers_of_2])
    ax.set_xlim(0.8, 80)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')


def _setup_log_xy(ax: plt.Axes) -> None:
    _setup_log_x(ax)
    ax.set_yscale('log', base=2)
    powers_of_2_y = [100, 50, 25, 12.5, 6.25, 3.125]
    ax.set_yticks(powers_of_2_y)
    ax.set_yticklabels([f'{y:.2f}%' if y < 10 else f'{int(y)}%' for y in powers_of_2_y])
    ax.set_ylim(2, 120)


def _setup_linear_x(ax: plt.Axes) -> None:
    ax.set_xticks(linear_ticks)
    ax.set_xlim(0, 65)
    ax.grid(True, alpha=0.3, linestyle='--')


def _find_crossing(x: pd.Series, y: pd.Series, target: float = 1.0) -> float | None:
    for i in range(1, len(x)):
        y0 = y.iloc[i - 1]
        y1 = y.iloc[i]
        if (y0 - target) == 0:
            return float(x.iloc[i - 1])
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            t = (target - y0) / (y1 - y0)
            return float(x.iloc[i - 1] + t * (x.iloc[i] - x.iloc[i - 1]))
    return None


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(name, dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")


def _resolve_csv(filename: str) -> Path:
    for root in RESULTS_DIRS:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return RESULTS_DIRS[0] / filename


def _load_pair(decode_tag: str) -> dict[str, pd.DataFrame]:
    prefetch_path = _resolve_csv(f"haul_scan_{decode_tag}_prefetch_profiler.csv")
    no_prefetch_path = _resolve_csv(f"haul_scan_{decode_tag}_no_prefetch_profiler.csv")

    # Backward-compatible fallback for older decode-on-demand filenames.
    if decode_tag == "decode_on_demand":
        legacy_prefetch = _resolve_csv("haul_scan_prefetch_profiler.csv")
        legacy_no_prefetch = _resolve_csv("haul_scan_no_prefetch_profiler.csv")
        if not prefetch_path.exists() and legacy_prefetch.exists():
            prefetch_path = legacy_prefetch
        if not no_prefetch_path.exists() and legacy_no_prefetch.exists():
            no_prefetch_path = legacy_no_prefetch

    missing = []
    if not prefetch_path.exists():
        missing.append(str(prefetch_path))
    if not no_prefetch_path.exists():
        missing.append(str(no_prefetch_path))

    if missing:
        raise SystemExit(
            "Missing CSVs for decode mode "
            f"'{decode_tag}':\n" + "\n".join(missing)
        )

    prefetch = pd.read_csv(prefetch_path)
    no_prefetch = pd.read_csv(no_prefetch_path)

    numeric_cols = [
        'frame_skip', 'runtime',
        'prefetch_total_batches',
        'prefetch_avg_wait_ms', 'prefetch_avg_infer_ms', 'prefetch_avg_prepare_ms',
        'prefetch_wait_ratio', 'prefetch_steady_avg_wait_ms', 'prefetch_steady_avg_infer_ms',
        'prefetch_steady_avg_prepare_ms', 'prefetch_steady_wait_ratio',
        'sync_total_prepare_ms',
        'sync_avg_prepare_ms', 'sync_avg_infer_ms', 'sync_wait_ratio',
        'sync_steady_avg_prepare_ms', 'sync_steady_avg_infer_ms', 'sync_steady_wait_ratio',
        'sync_ideal_serial_ms', 'sync_ideal_overlap_ms', 'sync_ideal_saved_ms',
        'sync_total_batches',
    ]

    _to_numeric(prefetch, numeric_cols)
    _to_numeric(no_prefetch, numeric_cols)

    merged = prefetch.merge(no_prefetch, on='frame_skip', suffixes=('_pf', '_no'))

    merged['speedup'] = merged['runtime_no'] / merged['runtime_pf']
    merged['saved_s'] = merged['runtime_no'] - merged['runtime_pf']

    merged['tc_ms'] = merged['sync_steady_avg_prepare_ms']
    merged['tg_ms'] = merged['sync_steady_avg_infer_ms']
    merged['N_b'] = merged['sync_total_batches']

    merged['T_serial_pred_s'] = (merged['tc_ms'] + merged['tg_ms']) * merged['N_b'] / 1000.0
    merged['T_overlap_pred_s'] = (
        (merged['tc_ms'] + merged['tg_ms']) / 1000.0
        + (merged['N_b'] - 1) * merged[['tc_ms', 'tg_ms']].max(axis=1) / 1000.0
    )
    merged['T_saved_pred_s'] = merged['T_serial_pred_s'] - merged['T_overlap_pred_s']
    merged['T_saved_obs_s'] = merged['saved_s']
    if 'sync_ideal_saved_ms' in merged.columns and merged['sync_ideal_saved_ms'].notna().any():
        merged['T_saved_pred_s'] = merged['sync_ideal_saved_ms'] / 1000.0
        if 'sync_ideal_overlap_ms' in merged.columns and merged['sync_ideal_overlap_ms'].notna().any():
            merged['T_overlap_pred_s'] = merged['sync_ideal_overlap_ms'] / 1000.0
    merged['saved_ratio'] = merged['T_saved_obs_s'] / merged['T_saved_pred_s']
    merged['overlap_err_pct'] = (
        (merged['T_overlap_pred_s'] - merged['runtime_pf']) / merged['runtime_pf'] * 100.0
    )
    merged['runtime_diff_s'] = merged['runtime_pf'] - merged['T_overlap_pred_s']
    merged['runtime_diff_pct'] = (
        merged['runtime_diff_s'] / merged['T_overlap_pred_s'].replace(0, np.nan) * 100.0
    )

    merged['gpu_idle_s'] = merged['prefetch_steady_wait_ratio'] * merged['runtime_pf']
    merged['prefetch_wait_s'] = (
        merged['prefetch_avg_wait_ms'] * merged['prefetch_total_batches'] / 1000.0
    )
    merged['sync_wait_s'] = merged['sync_total_prepare_ms'] / 1000.0
    merged['gpu_wait_saved_s'] = merged['sync_wait_s'] - merged['prefetch_wait_s']

    merged['prepare_per_infer_frame_ms'] = merged['prefetch_steady_avg_prepare_ms'] / BATCH_SIZE
    merged['prepare_per_advanced_frame_ms'] = (
        merged['prefetch_steady_avg_prepare_ms'] / (BATCH_SIZE * merged['frame_skip'])
    )
    merged['infer_per_infer_frame_ms'] = merged['prefetch_steady_avg_infer_ms'] / BATCH_SIZE

    merged['tc_tg_ratio'] = (
        merged['prefetch_steady_avg_prepare_ms'] / merged['prefetch_steady_avg_infer_ms']
    )

    return {
        "prefetch": prefetch,
        "no_prefetch": no_prefetch,
        "merged": merged,
    }


def _runtime_pct(df: pd.DataFrame) -> pd.Series:
    baseline = df.loc[df['frame_skip'] == 1, 'runtime']
    if baseline.empty:
        raise ValueError('frame_skip=1 missing in runtime data')
    return (df['runtime'] / float(baseline.iloc[0])) * 100.0


pairs = {tag: _load_pair(tag) for tag in DECODE_MODES}

# === Figure 1: Runtime Scaling vs Ideal (4 curves) ===
fig, ax = plt.subplots(figsize=(10, 6))

ideal_pct = None
for tag, cfg in DECODE_MODES.items():
    pair = pairs[tag]
    prefetch = pair['prefetch']
    no_prefetch = pair['no_prefetch']

    prefetch_pct = _runtime_pct(prefetch)
    no_prefetch_pct = _runtime_pct(no_prefetch)

    if ideal_pct is None:
        ideal_pct = (1 / prefetch['frame_skip']) * 100

    ax.plot(
        prefetch['frame_skip'],
        no_prefetch_pct,
        'o-',
        color=cfg['colors']['no_prefetch'],
        linewidth=2,
        markersize=4,
        label=f"{cfg['label']}, Serial",
    )
    ax.plot(
        prefetch['frame_skip'],
        prefetch_pct,
        's-',
        color=cfg['colors']['prefetch'],
        linewidth=2,
        markersize=4,
        label=f"{cfg['label']}, Overlapped (Prefetch)",
    )

ax.plot(
    prefetch['frame_skip'],
    ideal_pct,
    '--',
    color='#2ecc71',
    linewidth=2,
    alpha=0.7,
    label='Ideal (runtime ∝ 1/skip)',
)

_setup_log_xy(ax)
ax.set_xlabel('Frame Skip', fontsize=12)
ax.set_ylabel('Runtime (% of skip=1)', fontsize=12)
ax.set_title('Runtime Scaling: Decode Strategy × Overlap (Mac Studio M4 Max)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

_save(fig, 'mac_studio_scaling_decode_strategies.png')

# === Figure 2: Observed vs Ideal Saved Time (two panels) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2)
legend_handles: list = []
legend_labels: list = []

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    merged = pairs[tag]['merged']
    ax.plot(
        merged['frame_skip'],
        merged['T_saved_obs_s'],
        'o-',
        color=cfg['colors']['base'],
        linewidth=2,
        markersize=4,
        label='Observed Saved',
    )
    ax.plot(
        merged['frame_skip'],
        merged['T_saved_pred_s'],
        's--',
        color='#e67e22',
        linewidth=2,
        markersize=4,
        label='Ideal Saved by Formula',
    )
    _setup_log_x(ax)
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('Time Saved (seconds)', fontsize=12)
    ax.set_title(f"{cfg['label']}", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

fig.suptitle('Observed vs Ideal Saved Time (Mac Studio M4 Max)',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88)

_save(fig, 'mac_studio_prefetch_saved_vs_ideal_by_decode.png')

# === Figure 3: Tc / Tg Ratio (prefetch steady) ===
fig, ax = plt.subplots(figsize=(10, 6))

for tag, cfg in DECODE_MODES.items():
    merged = pairs[tag]['merged']
    crossing = _find_crossing(merged['frame_skip'], merged['tc_tg_ratio'])
    label = cfg['label']
    if crossing is not None:
        label = f"{cfg['label']} (CPU>=GPU @ {crossing:.1f})"
    ax.plot(
        merged['frame_skip'],
        merged['tc_tg_ratio'],
        'o-',
        color=cfg['colors']['base'],
        linewidth=2,
        markersize=4,
        label=label,
    )

ax.axhline(1.0, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7, label='CPU = GPU')
_setup_log_x(ax)
ax.set_xlabel('Frame Skip', fontsize=12)
ax.set_ylabel('Tc / Tg', fontsize=12)
ax.set_title('CPU vs GPU Bottleneck (Overlap Steady)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

_save(fig, 'mac_studio_prefetch_tc_tg_ratio.png')

# === Figure 4: GPU Idle vs Wait Ratio (two panels) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2)
legend_handles: list = []
legend_labels: list = []

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    merged = pairs[tag]['merged']

    bar_color = cfg['colors']['base']
    ax.bar(merged['frame_skip'], merged['gpu_idle_s'], color=bar_color, alpha=0.6,
           label='GPU Idle (seconds)')
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('GPU Idle Time (seconds)', fontsize=12, color=bar_color)
    ax.tick_params(axis='y', labelcolor=bar_color)
    _setup_linear_x(ax)

    ax2 = ax.twinx()
    ax2.plot(
        merged['frame_skip'],
        merged['prefetch_steady_wait_ratio'],
        'o-',
        color='#e67e22',
        linewidth=2,
        markersize=4,
        label='Wait Ratio (steady)',
    )
    ax2.set_ylabel('Wait Ratio', fontsize=12, color='#e67e22')
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax2.set_ylim(0, max(0.35, merged['prefetch_steady_wait_ratio'].max() * 1.2))

    ax.set_title(cfg['label'], fontsize=13, fontweight='bold')

    if not legend_handles:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend_handles = lines1 + lines2
        legend_labels = labels1 + labels2

fig.legend(legend_handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10)
fig.suptitle('Overlap GPU Idle vs Wait Ratio (Mac Studio M4 Max)',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88, bottom=0.12)

_save(fig, 'mac_studio_prefetch_idle_vs_wait_by_decode.png')

# === Figure 5: Prepare/Infer Cost per Frame (two panels) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2, sharey=True)
frame_cost_values: list[float] = []

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    merged = pairs[tag]['merged']
    frame_cost_values.extend(merged['prepare_per_infer_frame_ms'].dropna().tolist())
    # Only plot per inferred-frame metrics for clarity
    frame_cost_values.extend(merged['infer_per_infer_frame_ms'].dropna().tolist())
    ax.plot(
        merged['frame_skip'],
        merged['prepare_per_infer_frame_ms'],
        'o-',
        color='#e67e22',
        linewidth=2,
        markersize=4,
        label='Prepare per inferred frame',
    )
    ax.plot(
        merged['frame_skip'],
        merged['infer_per_infer_frame_ms'],
        's-',
        color='#3498db',
        linewidth=2,
        markersize=4,
        label='Infer per inferred frame',
    )
    _setup_log_x(ax)
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('Cost per Frame (ms)', fontsize=12)
    ax.set_title(cfg['label'], fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

if frame_cost_values:
    ymin = max(0.0, min(frame_cost_values) * 0.9)
    ymax = max(frame_cost_values) * 1.1
    axes[0].set_ylim(ymin, ymax)

fig.suptitle('Prepare/Infer Cost per Frame (Overlap Steady)', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88)

_save(fig, 'mac_studio_prepare_infer_batch_and_frame_by_decode.png')

# === Figure 6: Overlap steady vs overall deltas (two panels) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    prefetch = pairs[tag]['prefetch'].copy()
    prefetch['delta_prepare_ms'] = prefetch['prefetch_avg_prepare_ms'] - prefetch['prefetch_steady_avg_prepare_ms']
    prefetch['delta_infer_ms'] = prefetch['prefetch_avg_infer_ms'] - prefetch['prefetch_steady_avg_infer_ms']
    prefetch['delta_wait_ms'] = prefetch['prefetch_avg_wait_ms'] - prefetch['prefetch_steady_avg_wait_ms']

    ax.plot(prefetch['frame_skip'], prefetch['delta_prepare_ms'], 'o-',
            color='#e67e22', linewidth=2, markersize=4, label='Prepare: avg - steady')
    ax.plot(prefetch['frame_skip'], prefetch['delta_infer_ms'], 's-',
            color='#3498db', linewidth=2, markersize=4, label='Infer: avg - steady')
    ax.plot(prefetch['frame_skip'], prefetch['delta_wait_ms'], 'd-',
            color='#2ecc71', linewidth=2, markersize=4, label='Wait: avg - steady')

    _setup_log_x(ax)
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('Delta (ms)', fontsize=12)
    ax.set_title(cfg['label'], fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

fig.suptitle('Overlap Steady vs Overall Deltas (Mac Studio M4 Max)',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88)

_save(fig, 'mac_studio_prefetch_steady_deltas_by_decode.png')

# === Figure 7: Ideal vs Observed Gap (Absolute + Relative) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2)
legend_handles: list = []
legend_labels: list = []

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    merged = pairs[tag]['merged']
    gap_s = merged['T_saved_pred_s'] - merged['T_saved_obs_s']
    rel_gap = gap_s / merged['T_saved_pred_s'].replace(0, np.nan)

    bar_color = cfg['colors']['base']
    ax.bar(
        merged['frame_skip'],
        gap_s,
        color=bar_color,
        alpha=0.5,
        label='Ideal - Observed (gap)',
    )
    ax.axhline(0, color='#666666', linewidth=1, alpha=0.6)
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('Gap (seconds)', fontsize=12)
    _setup_linear_x(ax)

    ax2 = ax.twinx()
    ax2.plot(
        merged['frame_skip'],
        rel_gap * 100.0,
        'o-',
        color='#e67e22',
        linewidth=2,
        markersize=4,
        label='Gap / Ideal (%)',
    )
    ax2.axhline(0, color='#e67e22', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_ylabel('Relative Gap (%)', fontsize=12, color='#e67e22')
    ax2.tick_params(axis='y', labelcolor='#e67e22')

    ax.set_title(cfg['label'], fontsize=13, fontweight='bold')

    if ax is axes[0]:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend_handles = lines1 + lines2
        legend_labels = labels1 + labels2

fig.legend(legend_handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10)
fig.suptitle('Saved Time Diff: Observed vs Predicted (Mac Studio M4 Max)',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88, bottom=0.12)

_save(fig, 'mac_studio_prefetch_gap_vs_relative_error.png')

# === Figure 8: Runtime Diff (Observed vs Formula) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2)
legend_handles = []
legend_labels = []

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    merged = pairs[tag]['merged']
    bar_color = cfg['colors']['base']
    ax.bar(
        merged['frame_skip'],
        merged['runtime_diff_s'],
        color=bar_color,
        alpha=0.5,
        label='Observed - Predicted (s)',
    )
    ax.axhline(0.0, color='#666666', linewidth=1, alpha=0.6)
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('Diff (s)', fontsize=12)
    _setup_linear_x(ax)

    ax2 = ax.twinx()
    ax2.plot(
        merged['frame_skip'],
        merged['runtime_diff_pct'],
        'o-',
        color='#e67e22',
        linewidth=2,
        markersize=4,
        label='Diff (%)',
    )
    ax2.axhline(0.0, color='#e67e22', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_ylabel('Diff (%)', fontsize=12, color='#e67e22')
    ax2.tick_params(axis='y', labelcolor='#e67e22')

    ax.set_title(cfg['label'], fontsize=13, fontweight='bold')

    if not legend_handles:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend_handles = lines1 + lines2
        legend_labels = labels1 + labels2

fig.legend(legend_handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10)
fig.suptitle('Runtime Diff: Observed vs Predicted',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88, bottom=0.12)

_save(fig, 'mac_studio_runtime_vs_formula_by_decode.png')

# === Figure 9: GPU Wait (Serial vs Overlapped) ===
fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for ax, (tag, cfg) in zip(axes, DECODE_MODES.items()):
    merged = pairs[tag]['merged']
    ax.plot(
        merged['frame_skip'],
        merged['sync_wait_s'],
        'o-',
        color='#7f8c8d',
        linewidth=2,
        markersize=4,
        label='Serial',
    )
    ax.plot(
        merged['frame_skip'],
        merged['prefetch_wait_s'],
        's-',
        color=cfg['colors']['base'],
        linewidth=2,
        markersize=4,
        label='Overlapped (Prefetch)',
    )
    ax.axhline(0.0, color='#666666', linewidth=1, alpha=0.6)
    _setup_linear_x(ax)
    ax.set_xlabel('Frame Skip', fontsize=12)
    ax.set_ylabel('GPU Wait s', fontsize=12)

    ax.set_title(cfg['label'], fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

fig.suptitle('GPU Wait: Serial vs Overlapped',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.88)

_save(fig, 'mac_studio_gpu_wait_saved_by_decode.png')
