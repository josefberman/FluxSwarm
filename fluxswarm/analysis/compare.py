"""RL vs baselines aggregation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_episode_summary(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    pq = run_dir / "episodes.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = run_dir / "episodes.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No episodes file in {run_dir}")


def compare_runs(run_dirs: Sequence[Path], labels: Sequence[str] | None = None) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(run_dirs):
        label = labels[i] if labels else Path(d).name
        eps = load_episode_summary(d)
        rows.append(
            {
                "run": label,
                "n_episodes": len(eps),
                "mean_steps": float(eps["steps"].mean()) if "steps" in eps else None,
                "mean_cum_progress": float(eps["cum_progress"].mean()) if "cum_progress" in eps else None,
            }
        )
    return pd.DataFrame(rows)
