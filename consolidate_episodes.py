"""Consolidate episode summaries from all parallel envs and plot training curves.

Per env we read every ``episodes_summary_*.csv`` under ``env_<id>_<YYYYMMDD>_<HHMMSS>``
folders, sort them chronologically, and build:

* ``global_steps``: cumulative environment steps across *all* envs combined, so that the
  x-axis is the total amount of experience generated (comparable to wall-clock time).
* Per-env step-function interpolation (``kind='previous'``) onto a shared global-steps
  grid — reward is known only at episode end, so we hold the last observed value rather
  than linearly interpolating between episodes.
* Mean ± std across envs at every grid point, as a filled band.
* Scattered dots at actual episode-end positions so you can see per-env termination density.
* Episode-length and status-mix plots on the same x-axis.
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d


# ------------------------- Discovery & loading -------------------------------

_FOLDER_RE = re.compile(r'env_(\d+)_(\d{8})_(\d{6})')


def _discover_envs(base_path: Path) -> dict[int, list[tuple[datetime, Path]]]:
    """Return ``{env_id: [(timestamp, folder), ...]}`` sorted by timestamp."""
    envs: dict[int, list[tuple[datetime, Path]]] = {}
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        m = _FOLDER_RE.match(item.name)
        if not m:
            continue
        env_id = int(m.group(1))
        try:
            ts = datetime.strptime(m.group(2) + m.group(3), '%Y%m%d%H%M%S')
        except ValueError:
            print(f"  Warning: bad timestamp in {item.name}")
            continue
        envs.setdefault(env_id, []).append((ts, item))
    for env_id in envs:
        envs[env_id].sort(key=lambda x: x[0])
    return envs


def _load_env(folder_entries: list[tuple[datetime, Path]], env_id: int) -> pd.DataFrame | None:
    """Concatenate every ``episodes_summary_*.csv`` for this env in chronological order."""
    parts: list[pd.DataFrame] = []
    for _, folder in folder_entries:
        for csv_path in sorted(folder.glob("episodes_summary_*.csv")):
            try:
                parts.append(pd.read_csv(csv_path))
            except Exception as exc:
                print(f"  Warning: failed to read {csv_path}: {exc}")
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df['env_id'] = env_id
    if 'steps' in df.columns:
        df['env_steps'] = df['steps'].astype(float).cumsum()
    if 'episode' in df.columns:
        df['episode'] = np.arange(1, len(df) + 1)
    return df


def _add_episode_end_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'status' not in df.columns:
        return df
    s = df['status'].astype(str).str.strip().str.lower()
    df = df.copy()
    df['ended_terminated'] = s.isin(['terminated', 'both'])
    df['ended_truncated'] = s.isin(['truncated', 'both'])
    return df


# ------------------------- Global steps ---------------------------------------

def _add_global_steps(env_dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Add ``global_steps = env_steps * num_envs`` to each DataFrame (in-place copy).

    This converts per-env cumulative steps to a proxy for total environment-steps
    generated across all parallel envs — equivalent to wall-clock time when all envs
    run in lock-step.
    """
    num_envs = len(env_dfs)
    out = []
    for df in env_dfs:
        d = df.copy()
        if 'env_steps' in d.columns:
            d['global_steps'] = d['env_steps'] * num_envs
        out.append(d)
    return out


# ------------------------- Aggregation ---------------------------------------

def _common_grid(env_dfs: list[pd.DataFrame], n_points: int) -> np.ndarray:
    """Dense ``global_steps`` grid over the overlap range of all envs.

    Overlap ``[max(first_global_step), min(last_global_step)]`` ensures every env has
    data on both sides of every grid point, so step-function interpolation is defined.
    """
    firsts, lasts = [], []
    for df in env_dfs:
        if 'global_steps' not in df.columns or df.empty:
            continue
        firsts.append(float(df['global_steps'].iloc[0]))
        lasts.append(float(df['global_steps'].iloc[-1]))
    if not firsts or not lasts:
        return np.array([])
    x_min, x_max = max(firsts), min(lasts)
    if x_max <= x_min:
        return np.array([])
    return np.linspace(x_min, x_max, n_points)


def _step_interp(x_known: np.ndarray, y_known: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Step-function (hold-previous) interpolation.

    Reward is only observed at episode end; between episodes the last known value is held.
    Grid points before the first observation return NaN (env hasn't finished its first
    episode yet — no information available).
    """
    f = interp1d(x_known, y_known, kind='previous', bounds_error=False, fill_value=(np.nan, y_known[-1]))
    return f(x_query)


def _aggregate_across_envs(
    env_dfs: list[pd.DataFrame],
    columns: list[str],
    grid: np.ndarray,
) -> dict[str, dict]:
    """Per env, step-interpolate each metric onto ``grid``; compute mean/std across envs.

    Returns ``{col: {'mean', 'std', 'grid', 'per_env_dots'}}``.
    ``per_env_dots`` is a list of ``(global_steps_array, values_array)`` for scatter plots.
    """
    if grid.size < 2:
        return {}
    out: dict[str, dict] = {}
    for col in columns:
        per_env_interp: list[np.ndarray] = []
        for df in env_dfs:
            if col not in df.columns or 'global_steps' not in df.columns:
                continue
            x = df['global_steps'].to_numpy(dtype=float)
            y = df[col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 1:
                continue
            x, y = x[mask], y[mask]
            order = np.argsort(x)
            x, y = x[order], y[order]
            per_env_interp.append(_step_interp(x, y, grid))
        if not per_env_interp:
            continue
        stack = np.vstack(per_env_interp).astype(float)
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
        out[col] = {'mean': mean, 'std': std, 'grid': grid}
    return out




# ------------------------- Plotting ------------------------------------------

_COLORS = {
    'progress': 'tab:purple',
    'progress_per_step': 'tab:purple',
    'energy_efficiency': 'tab:green',
    'energy_efficiency_per_step': 'tab:green',
    'smoothness': 'tab:orange',
    'smoothness_per_step': 'tab:orange',
    'total_reward': 'black',
    'total_reward_per_step': 'black',
}


def _pretty(name: str) -> str:
    return name.replace('cum_', '').replace('_', ' ').title()


def _color_for(col: str) -> str:
    key = col.lower().replace('cum_', '')
    return _COLORS.get(key, 'tab:blue')


def _setup_pub_style() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.2,
    })


def _save(fig: plt.Figure, base: Path, name: str) -> None:
    fig.savefig(base / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {name}.png")


def _plot_metric(
    agg: dict[str, dict],
    cols: list[str],
    base: Path,
    name: str,
    ylabel_suffix: str = '',
) -> None:
    valid = [c for c in cols if c in agg]
    if not valid:
        return
    fig, axes = plt.subplots(nrows=len(valid), ncols=1, figsize=(11, 3.2 * len(valid)), sharex=True)
    if len(valid) == 1:
        axes = [axes]
    for ax, col in zip(axes, valid):
        d = agg[col]
        x = d['grid']
        m = d['mean']
        s = d['std']
        c = _color_for(col)

        ax.fill_between(x, m - s, m + s, color=c, alpha=0.20, edgecolor='none', label='±1 SD across envs')
        ax.step(x, m, where='post', color=c, linewidth=1.2, label='mean across envs')
        # pretty = _pretty(col) + (' / step' if ylabel_suffix == '/step' else '')
        pretty = _pretty(col)
        ax.set_title(pretty, fontweight='bold')
        ax.set_ylabel(pretty)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], color='gray', linewidth=1.2, label='mean across envs'),
        plt.matplotlib.patches.Patch(color='gray', alpha=0.20, label='±1 SD across envs'),
    ]
    # axes[0].legend(handles=handles, loc='best', frameon=True, framealpha=0.9, edgecolor='gray')
    axes[-1].set_xlabel('Global environment steps (all envs combined)')
    fig.tight_layout()
    _save(fig, base, name)


def _plot_steps(agg_steps: dict | None, base: Path) -> None:
    if agg_steps is None:
        return
    d = agg_steps
    x = d['grid']
    m = d['mean']
    s = d['std']
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.fill_between(x, m - s, m + s, color='gray', alpha=0.25, edgecolor='none', label='±1 SD across envs')
    ax.step(x, m, where='post', color='black', linewidth=1.2, label='mean across envs')
    ax.set_title('Episode length', fontweight='bold')
    ax.set_xlabel('Global environment steps (all envs combined)')
    ax.set_ylabel('Steps per episode')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    # ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray')
    fig.tight_layout()
    _save(fig, base, 'episode_length')




# ------------------------- Top level -----------------------------------------

def consolidate_episodes(base_folder: str, n_points: int = 200) -> str:
    """Discover envs, aggregate across them, write plots and an Excel summary.

    X-axis is global environment steps (env_steps × num_envs), so it represents total
    experience generated, comparable across runs with different numbers of parallel envs.
    Interpolation is step-function / hold-previous: reward is only known at episode end.

    Returns the path to the Excel file.
    """
    base_path = Path(base_folder)
    if not base_path.exists():
        raise ValueError(f"Folder does not exist: {base_folder}")

    discovered = _discover_envs(base_path)
    if not discovered:
        raise ValueError(f"No env_<id>_<YYYYMMDD>_<HHMMSS> folders found in {base_folder}")

    raw_dfs: list[pd.DataFrame] = []
    for env_id in sorted(discovered):
        df = _load_env(discovered[env_id], env_id)
        if df is None or df.empty:
            print(f"  env_{env_id}: no episodes loaded")
            continue
        print(f"  env_{env_id}: {len(df)} episodes, env_steps span = {df['env_steps'].iloc[-1]:.0f}")
        raw_dfs.append(df)
    if not raw_dfs:
        raise ValueError("No episode CSVs were successfully loaded")

    num_envs = len(raw_dfs)
    print(f"Loaded {num_envs} envs — multiplying env_steps by {num_envs} for global_steps")
    env_dfs = _add_global_steps(raw_dfs)

    sample_cols = env_dfs[0].columns.tolist()
    reward_cols = [
        c for c in sample_cols
        if (c.lower().startswith('cum_') or 'reward' in c.lower())
        and c not in {'episode', 'steps', 'env_id', 'env_steps', 'global_steps', 'status'}
    ]
    print(f"Reward columns: {reward_cols}")

    grid = _common_grid(env_dfs, n_points=n_points)
    if grid.size < 2:
        raise ValueError(
            "Could not build a common global_steps grid; envs don't overlap or 'steps' column is missing"
        )
    print(f"Grid: {grid.size} points over global_steps ∈ [{grid[0]:.0f}, {grid[-1]:.0f}]")

    agg = _aggregate_across_envs(env_dfs, reward_cols, grid)

    per_step_dfs: list[pd.DataFrame] = []
    for df in env_dfs:
        if 'steps' not in df.columns:
            continue
        derived = df.copy()
        denom = derived['steps'].replace(0, np.nan).astype(float)
        for c in reward_cols:
            if c not in derived.columns:
                continue
            derived[f'{c}_per_step'] = derived[c] / denom
        per_step_dfs.append(derived)
    per_step_cols = [f'{c}_per_step' for c in reward_cols]
    agg_per_step = _aggregate_across_envs(per_step_dfs, per_step_cols, grid) if per_step_dfs else {}

    agg_steps_dict = _aggregate_across_envs(env_dfs, ['steps'], grid)
    agg_steps = agg_steps_dict.get('steps')

    _setup_pub_style()
    combined_cols = [c for c in reward_cols if c.lower() not in {'cum_total_reward', 'total_reward', 'reward'}]
    if not combined_cols:
        combined_cols = reward_cols

    _plot_metric(agg, combined_cols, base_path, 'rewards_combined_mean_std')
    if 'cum_total_reward' in agg:
        _plot_metric(agg, ['cum_total_reward'], base_path, 'total_reward_mean_std')
    if agg_per_step:
        per_step_combined = [f'{c}_per_step' for c in combined_cols if f'{c}_per_step' in agg_per_step]
        _plot_metric(agg_per_step, per_step_combined, base_path, 'rewards_combined_per_step', ylabel_suffix='/step')
        if 'cum_total_reward_per_step' in agg_per_step:
            _plot_metric(agg_per_step, ['cum_total_reward_per_step'], base_path, 'total_reward_per_step', ylabel_suffix='/step')
    _plot_steps(agg_steps, base_path)

    pooled = pd.concat(env_dfs, ignore_index=True)
    pooled = _add_episode_end_columns(pooled)
    excel_path = base_path / 'consolidated_episodes.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pooled.to_excel(writer, sheet_name='episodes', index=False)
        for col, d in agg.items():
            pd.DataFrame({'mean': d['mean'], 'std': d['std']}, index=d['grid']).to_excel(
                writer, sheet_name=f'agg_{col[:25]}'
            )
        for col, d in agg_per_step.items():
            pd.DataFrame({'mean': d['mean'], 'std': d['std']}, index=d['grid']).to_excel(
                writer, sheet_name=f'per_step_{col[:20]}'
            )
        if agg_steps is not None:
            pd.DataFrame({'mean': agg_steps['mean'], 'std': agg_steps['std']}, index=agg_steps['grid']).to_excel(
                writer, sheet_name='agg_steps'
            )
    print(f"Excel file created: {excel_path}")
    return str(excel_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python consolidate_episodes.py <folder_path> [n_points]")
        sys.exit(1)
    folder = sys.argv[1]
    n_points = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
    try:
        out = consolidate_episodes(folder, n_points=n_points)
        print(f"\nDone: {out}")
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
