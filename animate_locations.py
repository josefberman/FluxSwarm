import argparse
import math
import os
import re
from typing import List, Optional, Tuple

from train_cli import MEMBER_RADIUS, SIM_LENGTH_X, SIM_LENGTH_Y

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import shutil
import matplotlib as mpl
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path


def parse_args() -> argparse.Namespace:
    """
    Defaults for domain size and member circle radius match ``train_cli`` / ``main.py``.
    Other training CLI defaults are in ``train_cli.training_cli_defaults()`` if needed.
    """
    parser = argparse.ArgumentParser(
        description="Create an MP4 animation of swarm member trajectories from a locations CSV."
    )
    parser.add_argument(
        "--locations",
        required=True,
        help="Path to locations CSV (timestep, location_i_x, location_i_y, …)",
    )
    parser.add_argument(
        "--forces",
        default=None,
        help="Optional path to forces.csv (timestep, force_i_x, force_i_y); must align with locations rows. Draws force direction arrows.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path (default: same directory as --locations; multi-field adds _vx/_vy/_p before .mp4)",
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the animation (default: 24)")
    parser.add_argument(
        "--radius",
        type=float,
        default=MEMBER_RADIUS,
        help=f"Circle radius in axis units (default: {MEMBER_RADIUS}, same as main.py swarm)",
    )
    parser.add_argument(
        "--length_x",
        type=float,
        default=SIM_LENGTH_X,
        help=f"Simulation domain length in x-direction (default: {SIM_LENGTH_X}, same as main.py)",
    )
    parser.add_argument(
        "--length_y",
        type=float,
        default=SIM_LENGTH_Y,
        help=f"Simulation domain length in y-direction (default: {SIM_LENGTH_Y}, same as main.py)",
    )
    parser.add_argument(
        "--fields",
        required=True,
        help="Directory of step_*.npz field snapshots (or a single combined .npz) for background visualization",
    )
    parser.add_argument("--field_type", type=str, default=None, choices=['vx', 'vy', 'p'], help="Field type to display (vx, vy, or p)")
    return parser.parse_args()


def read_locations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestep" in df.columns:
        df = df.sort_values("timestep").reset_index(drop=True)
    # Drop the first row so locations lag one frame behind the field data.
    # Locations are recorded after the physics step, so they correspond to
    # the *end* of timestep t, while the field snapshot is from the *start*;
    # shifting by one frame re-aligns them.
    if len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def read_forces(csv_path: str) -> pd.DataFrame:
    """Same row trim as locations so forces stay frame-aligned."""
    df = pd.read_csv(csv_path)
    if "timestep" in df.columns:
        df = df.sort_values("timestep").reset_index(drop=True)
    if len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)
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


def load_fields(fields_path: str) -> dict:
    """
    Load fields from a single combined .npz file (shape: (timesteps, x, y)).
    """
    data = np.load(fields_path)
    
    if 'timesteps' in data:
        key = 'timesteps'
    elif 'timestep' in data and data['timestep'].ndim > 0:
        key = 'timestep'
    else:
        raise ValueError("Single step file provided. Use a directory of step_*.npz files instead.")
    
    return {
        'vx': data['vx'],
        'vy': data['vy'],
        'p':  data['p'],
        'timestep': data[key],
        'length_x': float(data['length_x']),
        'length_y': float(data['length_y'])
    }


def scan_fields_directory(fields_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Scan a directory of step_*.npz files and return:
      - sorted array of timestep values
      - corresponding sorted list of file paths
    Does NOT load any field data — just reads the 'timestep' scalar from each file.
    """
    import glob
    files = sorted(glob.glob(os.path.join(fields_dir, "step_*.npz")))
    if not files:
        raise ValueError(f"No step_*.npz files found in {fields_dir}")

    timesteps = []
    for f in files:
        d = np.load(f)
        timesteps.append(float(d['timestep']))
    return np.array(timesteps, dtype=np.float64), files


def load_fields_for_frames(fields_dir: str, frame_times: np.ndarray) -> dict:
    """
    For each timestamp in `frame_times` (the CSV timesteps),
    find the nearest field file and load it. Returns a dict of stacked arrays
    perfectly aligned to `frame_times`.
    """
    field_timesteps, field_files = scan_fields_directory(fields_dir)
    print(f"Field directory contains {len(field_files)} snapshots covering "
          f"t=[{field_timesteps[0]:.2f}, {field_timesteps[-1]:.2f}]")

    vx_list, vy_list, p_list = [], [], []
    length_x = length_y = None

    for t in frame_times:
        idx = int(np.argmin(np.abs(field_timesteps - t)))
        data = np.load(field_files[idx])
        vx_list.append(data['vx'])
        vy_list.append(data['vy'])
        p_list.append(data['p'])
        if length_x is None:
            length_x = float(data['length_x'])
            length_y = float(data['length_y'])

    print(f"  Loaded {len(vx_list)} field frames matching {len(frame_times)} animation frames")
    return {
        'vx': np.stack(vx_list, axis=0),
        'vy': np.stack(vy_list, axis=0),
        'p':  np.stack(p_list,  axis=0),
        'timestep': frame_times,
        'length_x': length_x,
        'length_y': length_y,
    }



def build_colors(n: int) -> List[str]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def create_animation(
    df: pd.DataFrame,
    output_path: str,
    fps: int,
    radius: float,
    length_x: float,
    length_y: float,
    fields_data: dict = None,
    field_type: str = None,
    forces_df: Optional[pd.DataFrame] = None,
) -> None:
    member_ids = extract_member_ids(df)
    if not member_ids:
        raise ValueError("No member columns found. Expected columns like 'location_0_x' and 'location_0_y'.")

    n_frames = len(df)

    # Use fixed simulation domain dimensions
    x_min, x_max = 0.0, length_x
    y_min, y_max = 0.0, length_y

    # Figure size: wide enough for the x-domain; axes aspect below makes y/x display 4:1.
    target_max_pixels = 2000
    fig_width = 12.0
    fig_height = max(fig_width * (length_y / length_x) * 4.0, 3.0)
    max_fig_dim = max(fig_width, fig_height)
    dpi = int(target_max_pixels / max_fig_dim)
    dpi = max(50, min(dpi, 300))
    final_pixel_width = fig_width * dpi
    final_pixel_height = fig_height * dpi
    if final_pixel_width > target_max_pixels or final_pixel_height > target_max_pixels:
        scale = min(target_max_pixels / final_pixel_width, target_max_pixels / final_pixel_height)
        fig_width *= scale
        fig_height *= scale

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Visual y/x scale 4:1 (y data unit draws 4× the length of one x unit); not 1:1 "equal".
    ax.set_aspect(4.0, adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.grid(False)
    
    # Setup field background if provided
    field_img = None
    colorbar = None
    if fields_data and field_type:
        field_array = fields_data[field_type]  # Shape: (timesteps, x, y)
        # Compute global min/max for constant colorbar
        vmin, vmax = float(field_array.min()), float(field_array.max())
        
        # Display first frame as background
        field_img = ax.imshow(
            field_array[0].T,  # Transpose to match (x,y) -> (row, col) convention
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='bwr',
            vmin=-max(abs(vmin), abs(vmax)),
            vmax=max(abs(vmin), abs(vmax)),
            aspect="auto",
            zorder=0,
            interpolation='bilinear'
        )
        # Horizontal colorbar below axes — extra pad + ample bottom rect so xlabel stays above colorbar.
        colorbar = fig.colorbar(
            field_img, ax=ax, orientation='horizontal', pad=0.26, shrink=0.85, aspect=28
        )
        colorbar_labels = {
            'vx': 'Velocity X [mm/s]',
            'vy': 'Velocity Y [mm/s]',
            'p': 'Pressure [mg/(mm·s²)]'
        }
        colorbar.set_label(colorbar_labels.get(field_type, field_type))

    colors = build_colors(len(member_ids))
    circles: List[Circle] = []
    for idx, mid in enumerate(member_ids):
        x0 = float(df[f"location_{mid}_x"].iloc[0])
        y0 = float(df[f"location_{mid}_y"].iloc[0])
        circ = Circle(
            (x0, y0),
            radius=radius,
            facecolor=colors[idx],
            edgecolor='black',
            linewidth=0.35,
            zorder=5,
        )
        ax.add_patch(circ)
        circles.append(circ)

    # Add a black smaller circle for the center of mass
    # Initial center of mass
    xs0 = [float(df[f"location_{mid}_x"].iloc[0]) for mid in member_ids]
    ys0 = [float(df[f"location_{mid}_y"].iloc[0]) for mid in member_ids]
    com_x0 = np.mean(xs0)
    com_y0 = np.mean(ys0)
    com_radius = radius * 0.3
    com_circle = Circle(
        (com_x0, com_y0),
        radius=com_radius,
        edgecolor='black',
        facecolor='black',
        linewidth=0.5,
        zorder=10,
    )
    ax.add_patch(com_circle)

    arrows: List[FancyArrowPatch] = []
    if forces_df is not None:
        for mid in member_ids:
            cols = (f"force_{mid}_x", f"force_{mid}_y")
            if cols[0] not in forces_df.columns or cols[1] not in forces_df.columns:
                raise ValueError(f"Forces CSV must include columns {cols[0]} and {cols[1]}.")

        base_arrow_mm = max(radius * 2.25, length_y * 0.06, length_x * 0.015)
        force_mag_max = 0.0
        for i in range(len(forces_df)):
            for mid in member_ids:
                fx_i = float(forces_df[f"force_{mid}_x"].iloc[i])
                fy_i = float(forces_df[f"force_{mid}_y"].iloc[i])
                force_mag_max = max(force_mag_max, math.hypot(fx_i, fy_i))
        force_mag_max = float(max(force_mag_max, 1e-30))

        def arrow_length_mm(mag: float) -> float:
            return base_arrow_mm * (mag / force_mag_max)

        for mid in member_ids:
            x0 = float(df[f"location_{mid}_x"].iloc[0])
            y0 = float(df[f"location_{mid}_y"].iloc[0])
            fx0 = float(forces_df[f"force_{mid}_x"].iloc[0])
            fy0 = float(forces_df[f"force_{mid}_y"].iloc[0])
            m0 = math.hypot(fx0, fy0)
            L0 = arrow_length_mm(m0)
            if m0 < 1.0:
                p0, p1 = (x0, y0), (x0, y0)
                hide0 = True
            else:
                hide0 = False
                if L0 <= 0:
                    p0, p1 = (x0, y0), (x0, y0)
                else:
                    p0 = (x0, y0)
                    p1 = (x0 + (fx0 / m0) * L0, y0 + (fy0 / m0) * L0)
            arr = FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=4.2,
                mutation_aspect=0.32,
                linewidth=0.4,
                edgecolor="black",
                facecolor="black",
                zorder=6,
                clip_on=True,
                visible=not hide0,
            )
            ax.add_patch(arr)
            arrows.append(arr)

    def update(frame_idx: int):
        # Update field background if present — fields_data is already aligned to df rows
        artists = []
        if field_img is not None and fields_data is not None:
            field_array = fields_data[field_type]
            # fields_data was built exactly matching df rows, so direct index is correct
            if frame_idx < len(field_array):
                field_img.set_data(field_array[frame_idx].T)
                artists.append(field_img)

        # With a field overlay, members (and arrows) lag by one CSV row vs the field shown.
        swarm_frame = (
            max(0, frame_idx - 1)
            if field_img is not None and fields_data is not None
            else frame_idx
        )

        xs = []
        ys = []
        for circ, mid in zip(circles, member_ids):
            x = float(df[f"location_{mid}_x"].iloc[swarm_frame])
            y = float(df[f"location_{mid}_y"].iloc[swarm_frame])
            circ.center = (x, y)
            xs.append(x)
            ys.append(y)
        # Update center of mass circle
        com_x = np.mean(xs)
        com_y = np.mean(ys)
        com_circle.center = (com_x, com_y)
        out_art = [*artists, *circles, com_circle]
        if arrows:
            for arr, mid in zip(arrows, member_ids):
                x = float(df[f"location_{mid}_x"].iloc[swarm_frame])
                y = float(df[f"location_{mid}_y"].iloc[swarm_frame])
                fx = float(forces_df[f"force_{mid}_x"].iloc[swarm_frame])
                fy = float(forces_df[f"force_{mid}_y"].iloc[swarm_frame])
                m = math.hypot(fx, fy)
                if m < 1.0:
                    arr.set_visible(False)
                else:
                    L = arrow_length_mm(m)
                    arr.set_visible(True)
                    if L <= 0:
                        arr.set_positions((x, y), (x, y))
                    else:
                        ux, uy = fx / m, fy / m
                        arr.set_positions((x, y), (x + ux * L, y + uy * L))
                out_art.append(arr)
        return out_art

    fps_used = max(1, int(fps))
    frame_interval_ms = max(1, int(1000 / fps_used))
    print(f"  Animation: {n_frames} frames at {fps_used} fps (interval {frame_interval_ms} ms)")
    if colorbar is not None:
        fig.tight_layout(rect=(0, 0.22, 1, 1))
    else:
        fig.tight_layout()
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=frame_interval_ms, blit=True
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps_used, codec='h264', bitrate=-1)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FFmpeg is required to write MP4. Please install FFmpeg and ensure it is on PATH.") from exc

    # Use calculated DPI to keep video dimensions reasonable
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def _output_base(locations_path: str, output_override: Optional[str]) -> str:
    """Base path for MP4(s): same directory as locations when output is omitted."""
    if output_override is None:
        return os.path.splitext(os.path.abspath(locations_path))[0]
    base = output_override.rsplit(".", 1)[0] if "." in os.path.basename(output_override) else output_override
    return os.path.abspath(base)


def main() -> None:
    args = parse_args()
    df = read_locations(args.locations)
    forces_df = read_forces(args.forces) if args.forces else None
    if forces_df is not None and len(forces_df) != len(df):
        raise ValueError(
            f"forces row count ({len(forces_df)}) must match locations ({len(df)}) after alignment trim."
        )
    out_base = _output_base(args.locations, args.output)

    # Gather the actual simulation timestamps we'll be animating
    if "timestep" in df.columns:
        frame_times = df["timestep"].to_numpy(dtype=np.float64)
    else:
        frame_times = None

    print(f"Loading fields from {args.fields}...")
    if os.path.isdir(args.fields):
        if frame_times is not None:
            fields_data = load_fields_for_frames(args.fields, frame_times)
        else:
            fields_data = load_fields_for_frames(
                args.fields, np.linspace(0, 1, len(df))
            )
    else:
        fields_data = load_fields(args.fields)

    print(f"  Loaded {len(fields_data['timestep'])} field snapshots")
    print(f"  VX range: [{fields_data['vx'].min():.2e}, {fields_data['vx'].max():.2e}]")
    print(f"  VY range: [{fields_data['vy'].min():.2e}, {fields_data['vy'].max():.2e}]")
    print(f"  P range:  [{fields_data['p'].min():.2e}, {fields_data['p'].max():.2e}]")

    if args.field_type is None:
        print("\nCreating animations for all three fields...")
        for field_type in ["vx", "vy", "p"]:
            output_path = f"{out_base}_{field_type}.mp4"
            print(f"Creating {field_type} animation -> {output_path}")
            create_animation(
                df=df,
                output_path=output_path,
                fps=args.fps,
                radius=args.radius,
                length_x=args.length_x,
                length_y=args.length_y,
                fields_data=fields_data,
                field_type=field_type,
                forces_df=forces_df,
            )
    else:
        output_path = f"{out_base}.mp4"
        print(f"Creating animation -> {output_path}")
        create_animation(
            df=df,
            output_path=output_path,
            fps=args.fps,
            radius=args.radius,
            length_x=args.length_x,
            length_y=args.length_y,
            fields_data=fields_data,
            field_type=args.field_type,
            forces_df=forces_df,
        )


if __name__ == "__main__":
    main()


