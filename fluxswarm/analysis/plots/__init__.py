"""Figure generators writing to runs_new/<run>/figures/ and _comparisons/."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_sidecar(path: Path, meta: dict[str, Any]) -> None:
    with path.with_suffix(".json").open("w") as f:
        json.dump({**meta, "git_commit": _git_commit()}, f, indent=2)


def _load_steps(run_dir: Path) -> Optional[pd.DataFrame]:
    pq = run_dir / "steps.parquet"
    return pd.read_parquet(pq) if pq.exists() else None


def _load_episodes(run_dir: Path) -> Optional[pd.DataFrame]:
    pq = run_dir / "episodes.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = run_dir / "episodes.csv"
    return pd.read_csv(csv) if csv.exists() else None


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if p.exists():
        with p.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def comparison_dir(output_root: Path, slug_parts: Sequence[str]) -> Path:
    raw = "|".join(slug_parts)
    h = hashlib.sha1(raw.encode()).hexdigest()[:10]
    slug = "_".join(slug_parts)[:80] + "_" + h
    d = output_root / "_comparisons" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_reward_per_episode(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "reward_per_episode.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    eps = _load_episodes(run_dir)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    metrics = ["progress", "energy_efficiency", "smoothness"]
    colors = ["#2F67B1", "#2E8B57", "#E07A2F"]
    if steps is not None and "episode" in steps.columns:
        for ax, m, c in zip(axes, metrics, colors):
            if m not in steps.columns:
                continue
            g = steps.groupby("episode")[m].mean()
            roll = g.rolling(5, center=True, min_periods=1).mean()
            ax.plot(roll.index, roll.values, color=c, label=m)
            ax.set_ylabel(m)
            ax.grid(True, alpha=0.3)
    elif eps is not None:
        for ax, m, c in zip(axes, metrics, colors):
            col = f"cum_{m}" if f"cum_{m}" in eps.columns else m
            if col in eps.columns:
                ax.plot(eps.index, eps[col], color=c)
                ax.set_ylabel(col)
    axes[-1].set_xlabel("episode")
    fig.suptitle(f"Reward per episode — {run_dir.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "reward_per_episode", "run_ids": [run_dir.name]})
    return out


def plot_episode_length(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "episode_length_per_episode.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = _load_episodes(run_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    if eps is not None and "steps" in eps.columns:
        roll = eps["steps"].rolling(5, center=True, min_periods=1).mean()
        std = eps["steps"].rolling(5, min_periods=1).std().fillna(0)
        ax.plot(eps.index, roll, color="gray")
        ax.fill_between(eps.index, roll - std, roll + std, color="gray", alpha=0.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"Episode length — {run_dir.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "episode_length_per_episode", "run_ids": [run_dir.name]})
    return out


def plot_center_of_mass(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "center_of_mass.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    if steps is not None:
        xs = [c for c in steps.columns if c.startswith("location_") and c.endswith("_x")]
        if xs and "episode_time" in steps.columns:
            com_x = steps[xs].mean(axis=1)
            rog = np.sqrt(((steps[xs].sub(com_x, axis=0)) ** 2).mean(axis=1))
            n = len(steps)
            early, late = slice(0, max(1, n // 10)), slice(max(0, n - n // 10), n)
            for ax, sl, title in zip(axes, [early, late], ["Before learning", "After learning"]):
                t = steps["episode_time"].iloc[sl]
                ax.plot(t, com_x.iloc[sl], color="#2F67B1")
                ax.fill_between(t, (com_x - rog).iloc[sl], (com_x + rog).iloc[sl], color="#2F67B1", alpha=0.2)
                ax.set_title(title)
                ax.set_xlabel("time (s)")
                ax.set_ylabel("COM x (mm)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "center_of_mass", "run_ids": [run_dir.name]})
    return out


def plot_dispersion(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "dispersion.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if steps is not None and "episode_time" in steps.columns:
        xs = [c for c in steps.columns if c.startswith("location_") and c.endswith("_x")]
        ys = [c for c in steps.columns if c.startswith("location_") and c.endswith("_y")]
        sx = steps[xs].std(axis=1) if xs else None
        sy = steps[ys].std(axis=1) if ys else None
        n = len(steps)
        early, late = slice(0, max(1, n // 10)), slice(max(0, n - n // 10), n)
        specs = [
            (0, early, "σx before", "#2F67B1", sx),
            (1, late, "σx after", "#2F67B1", sx),
            (2, early, "σy before", "#2E8B57", sy),
            (3, late, "σy after", "#2E8B57", sy),
        ]
        for col, sl, title, color, series in specs:
            ax = axes.flat[col]
            if series is not None:
                ax.plot(steps["episode_time"].iloc[sl], series.iloc[sl], color=color)
            ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "dispersion", "run_ids": [run_dir.name]})
    return out


def plot_velocity_temporal_profile(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "velocity_temporal_profile.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted((run_dir / "fields").glob("step_*.npz")) if (run_dir / "fields").exists() else []
    fig, ax = plt.subplots(figsize=(10, 4))
    if fields:
        ts, mean_vx, mean_vy = [], [], []
        for f in fields:
            d = np.load(f)
            ts.append(float(d["timestep"]))
            mean_vx.append(float(np.mean(d["vx"])))
            mean_vy.append(float(np.mean(d["vy"])))
        ax.plot(ts, mean_vx, label="mean vx")
        ax.plot(ts, mean_vy, label="mean vy")
        ax.legend()
    else:
        steps = _load_steps(run_dir)
        if steps is not None:
            vxs = [c for c in steps.columns if c.startswith("velocity_") and c.endswith("_x")]
            if vxs:
                ax.plot(steps["episode_time"], steps[vxs].mean(axis=1), label="swarm mean vx")
                ax.legend()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity (mm/s)")
    ax.set_title(f"Velocity temporal profile — {run_dir.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "velocity_temporal_profile", "run_ids": [run_dir.name]})
    return out


def plot_domain(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "domain.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(run_dir)
    sim, swarm = cfg.get("sim", {}), cfg.get("swarm", {})
    Lx, Ly = sim.get("length_x", 100), sim.get("length_y", 2)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, linewidth=2))
    ax.annotate("", xy=(15, Ly / 2), xytext=(5, Ly / 2), arrowprops=dict(arrowstyle="->", color="C0"))
    ax.text(8, Ly * 0.7, "inflow", color="C0")
    nx, ny = swarm.get("num_x", 8), swarm.get("num_y", 2)
    r = swarm.get("member_radius", 0.25)
    gap = (Ly - 2 * ny * r) / (ny + 1)
    left = Lx / 2 - (2 * (nx - 1) * r + (nx - 1) * gap) / 2
    bottom = gap + r
    for iy in range(ny):
        for ix in range(nx):
            ax.add_patch(
                plt.Circle(
                    (left + ix * (2 * r + gap), bottom + iy * (2 * r + gap)),
                    r,
                    color="0.3",
                    alpha=0.8,
                )
            )
    ax.set_xlim(-2, Lx + 2)
    ax.set_ylim(-0.3, Ly + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Domain")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "domain", "run_ids": [run_dir.name]})
    return out


def plot_architecture(run_dir: Path, which: str = "actor", out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    name = "actor_architecture" if which == "actor" else "critic_architecture"
    out = out or (run_dir / "figures" / f"{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(run_dir)
    hidden = cfg.get("train", {}).get("hidden_sizes", [256, 256])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    if which == "actor":
        labels = ["local obs"] + [f"Linear+Tanh\n{h}" for h in hidden] + ["μ (2)", "tanh+σ"]
    else:
        labels = ["joint obs"] + [f"Linear+Tanh\n{h}" for h in hidden] + ["V_p / V_e / V_s"]
    for i, lab in enumerate(labels):
        ax.add_patch(plt.Rectangle((i * 1.5, 0.25), 1.3, 0.5, facecolor="0.92", edgecolor="k"))
        ax.text(i * 1.5 + 0.65, 0.5, lab, ha="center", va="center", fontsize=8)
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=((i + 1) * 1.5, 0.5),
                xytext=(i * 1.5 + 1.3, 0.5),
                arrowprops=dict(arrowstyle="->"),
            )
    ax.set_xlim(-0.2, len(labels) * 1.5)
    ax.set_ylim(0, 1)
    ax.set_title(name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": name, "run_ids": [run_dir.name]})
    return out


def plot_pcgrad_comparison(with_dir: Path, without_dir: Path, out_dir: Optional[Path] = None) -> Path:
    with_dir, without_dir = Path(with_dir), Path(without_dir)
    out_dir = out_dir or comparison_dir(with_dir.parent, ["pcgrad", with_dir.name, without_dir.name])
    out = Path(out_dir) / "reward_with_without_PCGrad.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, m in zip(axes, ["progress", "energy_efficiency", "smoothness"]):
        for run, color, label in [
            (with_dir, "#2F67B1", "with PCGrad"),
            (without_dir, "#BF2C23", "without PCGrad"),
        ]:
            steps = _load_steps(run)
            if steps is None or m not in steps.columns:
                continue
            g = steps.groupby("episode")[m].mean().rolling(5, center=True, min_periods=1).mean()
            ax.plot(g.index, g.values, color=color, label=label)
        ax.set_ylabel(m)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("episode")
    fig.suptitle("PCGrad comparison")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "reward_with_without_PCGrad", "run_ids": [with_dir.name, without_dir.name]})
    return out


def generate_all_for_run(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    paths = [
        plot_reward_per_episode(run_dir),
        plot_episode_length(run_dir),
        plot_center_of_mass(run_dir),
        plot_dispersion(run_dir),
        plot_velocity_temporal_profile(run_dir),
        plot_domain(run_dir),
        plot_architecture(run_dir, "actor"),
        plot_architecture(run_dir, "critic"),
    ]
    try:
        from fluxswarm.analysis.swarm_viz import animate_swarm_from_run

        anim = animate_swarm_from_run(run_dir)
        if anim is not None:
            paths.append(anim)
    except Exception:
        pass
    return paths
