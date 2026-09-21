"""Lightweight swarm-in-domain rendering for TensorBoard and post-hoc animation."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style_axis(
    ax,
    length_x: float,
    length_y: float,
    success_x: float,
    failure_x: float,
    *,
    axis_aspect: str | float = "equal",
) -> None:
    ax.set_xlim(0.0, length_x)
    ax.set_ylim(0.0, length_y)
    ax.set_aspect(axis_aspect)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axvline(success_x, color="#2E8B57", ls="--", lw=1.0, alpha=0.8, label="success")
    ax.axvline(failure_x, color="#C0392B", ls="--", lw=1.0, alpha=0.8, label="failure")
    ax.add_patch(
        plt.Rectangle((0, 0), length_x, length_y, fill=False, linewidth=1.2, edgecolor="0.2")
    )


def render_swarm_frame(
    pos: np.ndarray,
    *,
    length_x: float,
    length_y: float,
    success_x: float,
    failure_x: float,
    radius: float,
    trail: Optional[Sequence[np.ndarray]] = None,
    title: str = "",
    dpi: int = 80,
    fig_scale: float = 1.0,
    display_wh_ratio: float | None = None,
) -> np.ndarray:
    """Render agent positions to an RGB uint8 array shaped (H, W, 3).

    ``pos`` is (N, 2). Optional ``trail`` is a sequence of older (N, 2) arrays
    (oldest first) drawn with increasing transparency.
    """
    pos = np.asarray(pos, dtype=np.float64)
    scale = max(0.5, float(fig_scale))
    fig_h = 2.5 * scale
    if display_wh_ratio is not None:
        wh = max(1.0, float(display_wh_ratio))
        fig_w = wh * fig_h
        axis_aspect = "auto"
    else:
        aspect = length_x / max(length_y, 1e-6)
        fig_w = min(14.0, max(6.0, aspect * 0.14)) * scale
        axis_aspect = "equal"
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    _style_axis(
        ax,
        length_x,
        length_y,
        success_x,
        failure_x,
        axis_aspect=axis_aspect,
    )

    pt_scale = scale * dpi / 72.0
    if trail:
        n_t = len(trail)
        for i, tp in enumerate(trail):
            alpha = 0.08 + 0.35 * ((i + 1) / n_t)
            ax.scatter(
                tp[:, 0],
                tp[:, 1],
                s=8 * pt_scale,
                c="#4C78A8",
                alpha=alpha,
                linewidths=0,
            )

    if pos.size:
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=max(12.0, 40.0 * radius) * pt_scale,
            c="#1F4E79",
            zorder=3,
        )
        for i, (x, y) in enumerate(pos):
            ax.add_patch(plt.Circle((x, y), radius, color="#1F4E79", alpha=0.35, zorder=2))

    if title:
        ax.set_title(title, fontsize=max(9, int(9 * scale)))
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = np.ascontiguousarray(buf[:, :, :3])
    plt.close(fig)
    return rgb


def rgb_to_chw_float(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (3,H,W) float32 in [0, 1] for SummaryWriter.add_image."""
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


class SwarmTrailRenderer:
    """Ring-buffer trail + TB-ready frames for env-0 swarm positions."""

    def __init__(
        self,
        length_x: float,
        length_y: float,
        success_x: float,
        failure_x: float,
        radius: float,
        trail_len: int = 0,
        dpi: int = 150,
        fig_scale: float = 1.5,
        display_wh_ratio: float = 4.0,
    ):
        self.length_x = float(length_x)
        self.length_y = float(length_y)
        self.success_x = float(success_x)
        self.failure_x = float(failure_x)
        self.radius = float(radius)
        self.dpi = int(dpi)
        self.fig_scale = float(fig_scale)
        self.display_wh_ratio = float(display_wh_ratio)
        self.trail_len = max(0, int(trail_len))
        self._trail: Deque[np.ndarray] = deque(maxlen=max(1, self.trail_len)) if self.trail_len > 0 else deque()

    def reset_trail(self) -> None:
        self._trail.clear()

    def render(self, pos: np.ndarray, title: str = "") -> np.ndarray:
        """Return CHW float image; updates the trail with ``pos`` (N, 2)."""
        pos = np.asarray(pos, dtype=np.float64)
        trail = list(self._trail) if self.trail_len > 0 else None
        rgb = render_swarm_frame(
            pos,
            length_x=self.length_x,
            length_y=self.length_y,
            success_x=self.success_x,
            failure_x=self.failure_x,
            radius=self.radius,
            trail=trail,
            title=title,
            dpi=self.dpi,
            fig_scale=self.fig_scale,
            display_wh_ratio=self.display_wh_ratio,
        )
        if self.trail_len > 0:
            self._trail.append(pos.copy())
        return rgb_to_chw_float(rgb)


def animate_swarm_from_run(
    run_dir: Path,
    out: Optional[Path] = None,
    *,
    max_frames: int = 400,
    fps: int = 12,
    dpi: int = 72,
) -> Optional[Path]:
    """Build a GIF of env-0 swarm motion from ``steps.parquet``.

    Subsamples rows so the GIF has at most ``max_frames`` frames. Returns
    ``None`` if there is no usable step log or Pillow is unavailable.
    """
    import pandas as pd
    import yaml

    run_dir = Path(run_dir)
    steps_path = run_dir / "steps.parquet"
    if not steps_path.exists():
        return None
    steps = pd.read_parquet(steps_path)
    xs = sorted(c for c in steps.columns if c.startswith("location_") and c.endswith("_x"))
    ys = sorted(c for c in steps.columns if c.startswith("location_") and c.endswith("_y"))
    if not xs or len(xs) != len(ys):
        return None

    cfg: dict = {}
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
    sim = cfg.get("sim", {})
    task = cfg.get("task", {})
    swarm = cfg.get("swarm", {})
    length_x = float(sim.get("length_x", 100.0))
    length_y = float(sim.get("length_y", 2.0))
    success_x = float(task.get("success_x", 20.0))
    failure_x = float(task.get("failure_x", 80.0))
    radius = float(swarm.get("member_radius", 0.25))

    n = len(steps)
    stride = max(1, int(np.ceil(n / max(1, max_frames))))
    idx = np.arange(0, n, stride)
    if len(idx) > max_frames:
        idx = idx[:max_frames]

    out = out or (run_dir / "figures" / "swarm_motion.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    for i in idx:
        row = steps.iloc[int(i)]
        pos = np.stack(
            [np.array([row[c] for c in xs], dtype=np.float64), np.array([row[c] for c in ys], dtype=np.float64)],
            axis=1,
        )
        title = f"step {int(row.get('global_step', i))}  t={float(row.get('episode_time', 0.0)):.2f}s"
        rgb = render_swarm_frame(
            pos,
            length_x=length_x,
            length_y=length_y,
            success_x=success_x,
            failure_x=failure_x,
            radius=radius,
            title=title,
            dpi=dpi,
            display_wh_ratio=4.0,
        )
        frames.append(rgb)

    if not frames:
        return None

    try:
        from PIL import Image
    except ImportError:
        k = min(8, len(frames))
        picks = [frames[int(j)] for j in np.linspace(0, len(frames) - 1, k)]
        h, w, _ = picks[0].shape
        cols = min(4, k)
        rows = (k + cols - 1) // cols
        sheet = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for j, fr in enumerate(picks):
            r, c = divmod(j, cols)
            sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = fr
        png = out.with_suffix(".png")
        plt.imsave(png, sheet)
        return png

    imgs = [Image.fromarray(f) for f in frames]
    duration_ms = max(1, int(1000 / max(1, fps)))
    imgs[0].save(
        out,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out
