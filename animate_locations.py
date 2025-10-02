import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib as mpl

mpl.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\Assaf\\anaconda3\\envs\\flow_swarm\\Library\\bin\\ffmpeg.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 animation of swarm member trajectories from a locations CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to locations.csv (with columns timestep, location_i_x, location_i_y)")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the animation (default: 24)")
    parser.add_argument("--radius", type=float, default=0.25, help="Circle radius in axis units (default: 0.25)")
    return parser.parse_args()


def read_locations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestep" in df.columns:
        df = df.sort_values("timestep").reset_index(drop=True)
    return df


def extract_member_ids(df: pd.DataFrame) -> List[int]:
    member_ids: List[int] = []
    for col in df.columns:
        match = re.match(r"location_(\d+)_x$", col)
        if match and f"location_{match.group(1)}_y" in df.columns:
            member_ids.append(int(match.group(1)))
    member_ids.sort()
    return member_ids


def compute_axis_limits(df: pd.DataFrame, member_ids: List[int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for mid in member_ids:
        xs.append(df[f"location_{mid}_x"].to_numpy())
        ys.append(df[f"location_{mid}_y"].to_numpy())
    if not xs or not ys:
        return (0.0, 1.0), (0.0, 1.0)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    # Add 5% padding
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def build_colors(n: int) -> List[str]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def create_animation(df: pd.DataFrame, output_path: str, fps: int, radius: float) -> None:
    member_ids = extract_member_ids(df)
    if not member_ids:
        raise ValueError("No member columns found. Expected columns like 'location_0_x' and 'location_0_y'.")

    (x_min, x_max), (y_min, y_max) = compute_axis_limits(df, member_ids)

    fig, ax = plt.subplots(figsize=(12, 3))
    # y:x data aspect ratio = 4:1
    ax.set_aspect(4.0, adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Swarm Motion")
    ax.grid(False)

    # Timestep overlay (updated each frame)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha='left', va='top')

    colors = build_colors(len(member_ids))
    circles: List[Circle] = []
    for idx, mid in enumerate(member_ids):
        x0 = float(df[f"location_{mid}_x"].iloc[0])
        y0 = float(df[f"location_{mid}_y"].iloc[0])
        circ = Circle((x0, y0), radius=radius, color=colors[idx])
        ax.add_patch(circ)
        circles.append(circ)

    # Add a dashed white smaller circle for the center of mass
    # Initial center of mass
    xs0 = [float(df[f"location_{mid}_x"].iloc[0]) for mid in member_ids]
    ys0 = [float(df[f"location_{mid}_y"].iloc[0]) for mid in member_ids]
    com_x0 = np.mean(xs0)
    com_y0 = np.mean(ys0)
    com_radius = radius * 0.5
    com_circle = Circle(
        (com_x0, com_y0),
        radius=com_radius,
        edgecolor='white',
        facecolor='none',
        linestyle='dashed',
        linewidth=2,
        zorder=10,
    )
    ax.add_patch(com_circle)

    num_frames = len(df)

    def update(frame_idx: int):
        xs = []
        ys = []
        for circ, mid in zip(circles, member_ids):
            x = float(df[f"location_{mid}_x"].iloc[frame_idx])
            y = float(df[f"location_{mid}_y"].iloc[frame_idx])
            circ.center = (x, y)
            xs.append(x)
            ys.append(y)
        # Update center of mass circle
        com_x = np.mean(xs)
        com_y = np.mean(ys)
        com_circle.center = (com_x, com_y)
        # Update timestep text
        if "timestep" in df.columns:
            t = float(df["timestep"].iloc[frame_idx])
            time_text.set_text(f"t = {t:.2f}")
        else:
            # Fallback: derive from frame index and 0.05 s per frame
            time_text.set_text(f"t = {frame_idx * 0.05:.2f}")
        return [*circles, com_circle, time_text]

    # Each frame is 0.05 seconds → 20 fps
    fps_used = 20
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps_used, codec='h264', bitrate=-1)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FFmpeg is required to write MP4. Please install FFmpeg and ensure it is on PATH.") from exc

    ani.save(output_path, writer=writer, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = read_locations(args.csv)
    create_animation(df=df, output_path=args.output, fps=args.fps, radius=args.radius)


if __name__ == "__main__":
    main()


