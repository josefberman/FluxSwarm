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

mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\Josef\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 animation of swarm member trajectories from a locations CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to locations.csv (with columns timestep, location_i_x, location_i_y)")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the animation (default: 24)")
    parser.add_argument("--radius", type=float, default=0.25, help="Circle radius in axis units (default: 0.25)")
    parser.add_argument("--length_x", type=float, default=100.0, help="Simulation domain length in x-direction (default: 100.0)")
    parser.add_argument("--length_y", type=float, default=4.0, help="Simulation domain length in y-direction (default: 4.0)")
    parser.add_argument("--fields", type=str, default=None, help="Path to fields npz file for background visualization")
    parser.add_argument("--field_type", type=str, default=None, choices=['vx', 'vy', 'p'], help="Field type to display (vx, vy, or p)")
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


def load_fields(fields_path: str) -> dict:
    """
    Load fields from compressed npz file.
    
    Handles two formats:
    1. Combined file: single npz with arrays of shape (timesteps, x, y)
    2. Directory pattern: path to directory containing step_*.npz files (will be handled by caller)
    """
    data = np.load(fields_path)
    
    # Check if this is a combined file (has 'timesteps' plural) or single step file
    if 'timesteps' in data:
        # Combined file format
        return {
            'vx': data['vx'],  # Shape: (timesteps, x, y)
            'vy': data['vy'],  # Shape: (timesteps, x, y)
            'p': data['p'],    # Shape: (timesteps, x, y)
            'timestep': data['timesteps'],  # Array of timesteps
            'length_x': float(data['length_x']),
            'length_y': float(data['length_y'])
        }
    elif 'timestep' in data:
        # Single step file - check if it's a scalar or array
        timestep_data = data['timestep']
        if timestep_data.ndim == 0:
            # Single scalar - this is a single step file, need to combine
            raise ValueError("Single step file provided. Use load_fields_from_directory() or provide combined file.")
        else:
            # Array - combined format with 'timestep' key
            return {
                'vx': data['vx'],
                'vy': data['vy'],
                'p': data['p'],
                'timestep': timestep_data,
                'length_x': float(data['length_x']),
                'length_y': float(data['length_y'])
            }
    else:
        raise ValueError(f"Unknown field file format: missing 'timestep' or 'timesteps' key")


def load_fields_from_directory(fields_dir: str) -> dict:
    """Load and combine all step_*.npz files from a directory."""
    import glob
    
    # Find all step files
    pattern = os.path.join(fields_dir, "step_*.npz")
    fields_files = sorted(glob.glob(pattern))
    
    if not fields_files:
        raise ValueError(f"No step_*.npz files found in {fields_dir}")
    
    print(f"Loading {len(fields_files)} field step files from {fields_dir}...")
    
    vx_list = []
    vy_list = []
    p_list = []
    timesteps_list = []
    length_x = None
    length_y = None
    
    for field_file in fields_files:
        data = np.load(field_file)
        vx_list.append(data['vx'])  # Shape: (x, y)
        vy_list.append(data['vy'])  # Shape: (x, y)
        p_list.append(data['p'])    # Shape: (x, y)
        timesteps_list.append(float(data['timestep']))
        
        # Store metadata from first file
        if length_x is None:
            length_x = float(data['length_x'])
            length_y = float(data['length_y'])
    
    # Stack into arrays: (timesteps, x, y)
    vx_array = np.stack(vx_list, axis=0)
    vy_array = np.stack(vy_list, axis=0)
    p_array = np.stack(p_list, axis=0)
    timesteps_array = np.array(timesteps_list, dtype=np.float64)
    
    print(f"  Combined into arrays with shape: vx={vx_array.shape}, vy={vy_array.shape}, p={p_array.shape}")
    
    return {
        'vx': vx_array,  # Shape: (timesteps, x, y)
        'vy': vy_array,  # Shape: (timesteps, x, y)
        'p': p_array,    # Shape: (timesteps, x, y)
        'timestep': timesteps_array,  # Array of timesteps
        'length_x': length_x,
        'length_y': length_y
    }


def build_colors(n: int) -> List[str]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def create_animation(df: pd.DataFrame, output_path: str, fps: int, radius: float, length_x: float, length_y: float, 
                     fields_data: dict = None, field_type: str = None) -> None:
    member_ids = extract_member_ids(df)
    if not member_ids:
        raise ValueError("No member columns found. Expected columns like 'location_0_x' and 'location_0_y'.")

    # Use fixed simulation domain dimensions
    x_min, x_max = 0.0, length_x
    y_min, y_max = 0.0, length_y

    # Calculate proper figure size to match simulation dimensions
    # Cap dimensions to prevent FFmpeg errors (max ~2000 pixels per dimension)
    target_max_pixels = 2000
    aspect_ratio = length_x / length_y
    
    # Start with reasonable figure size - cap width to prevent extreme aspect ratios
    fig_height = 4.0
    max_fig_width = 10.0  # Cap figure width to 10 inches
    fig_width = min(fig_height * aspect_ratio, max_fig_width)
    
    # Calculate DPI to keep pixel dimensions within target_max_pixels
    # Calculate what DPI would give us the target max pixels for the larger dimension
    max_fig_dim = max(fig_width, fig_height)
    dpi = int(target_max_pixels / max_fig_dim)
    dpi = max(50, min(dpi, 300))  # Clamp DPI between 50 and 300
    
    # Verify final pixel dimensions
    final_pixel_width = fig_width * dpi
    final_pixel_height = fig_height * dpi
    
    if final_pixel_width > target_max_pixels or final_pixel_height > target_max_pixels:
        # Further scale down if needed
        scale = min(target_max_pixels / final_pixel_width, target_max_pixels / final_pixel_height)
        fig_width *= scale
        fig_height *= scale
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio for mm units
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    
    # Set title based on field type
    if field_type:
        field_names = {'vx': 'X-Velocity', 'vy': 'Y-Velocity', 'p': 'Pressure'}
        ax.set_title(f"Swarm Motion - {field_names.get(field_type, field_type)} Field")
    else:
        ax.set_title("Swarm Motion")
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
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            zorder=0,
            interpolation='bilinear'
        )
        # Add colorbar
        colorbar = fig.colorbar(field_img, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
        colorbar_labels = {
            'vx': 'Velocity X [mm/s]',
            'vy': 'Velocity Y [mm/s]',
            'p': 'Pressure [mg/(mm·s²)]'
        }
        colorbar.set_label(colorbar_labels.get(field_type, field_type))

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

    num_frames = len(df)

    def update(frame_idx: int):
        # Update field background if present
        artists = []
        if field_img and fields_data:
            field_array = fields_data[field_type]
            if frame_idx < len(field_array):
                field_img.set_data(field_array[frame_idx].T)
                artists.append(field_img)
        
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
            time_text.set_text(f"t = {t:.2f} s")
        else:
            # Fallback: derive from frame index and 0.05 s per frame
            time_text.set_text(f"t = {frame_idx * 0.05:.2f} s")
        
        return [*artists, *circles, com_circle, time_text]

    # Each frame is 0.05 seconds → 20 fps
    fps_used = 20
    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps_used, codec='h264', bitrate=-1)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FFmpeg is required to write MP4. Please install FFmpeg and ensure it is on PATH.") from exc

    # Use calculated DPI to keep video dimensions reasonable
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = read_locations(args.csv)
    
    # Load fields if provided
    fields_data = None
    if args.fields:
        print(f"Loading fields from {args.fields}...")
        # Check if it's a directory or a file
        if os.path.isdir(args.fields):
            fields_data = load_fields_from_directory(args.fields)
        else:
            try:
                fields_data = load_fields(args.fields)
            except ValueError as e:
                if "Single step file" in str(e):
                    # Try treating it as a directory pattern
                    fields_dir = os.path.dirname(args.fields)
                    if os.path.isdir(fields_dir):
                        fields_data = load_fields_from_directory(fields_dir)
                    else:
                        raise
                else:
                    raise
        
        print(f"  Loaded {len(fields_data['timestep'])} field snapshots")
        print(f"  VX range: [{fields_data['vx'].min():.2e}, {fields_data['vx'].max():.2e}]")
        print(f"  VY range: [{fields_data['vy'].min():.2e}, {fields_data['vy'].max():.2e}]")
        print(f"  P range: [{fields_data['p'].min():.2e}, {fields_data['p'].max():.2e}]")
        
        # If field_type is not specified but fields are provided, create all three animations
        if args.field_type is None:
            print("\nCreating animations for all three fields...")
            base_output = args.output.rsplit('.', 1)[0]  # Remove extension
            
            for field_type in ['vx', 'vy', 'p']:
                output_path = f"{base_output}_{field_type}.mp4"
                print(f"Creating {field_type} animation -> {output_path}")
                create_animation(
                    df=df, 
                    output_path=output_path, 
                    fps=args.fps, 
                    radius=args.radius, 
                    length_x=args.length_x, 
                    length_y=args.length_y,
                    fields_data=fields_data,
                    field_type=field_type
                )
        else:
            # Create single animation with specified field type
            create_animation(
                df=df, 
                output_path=args.output, 
                fps=args.fps, 
                radius=args.radius, 
                length_x=args.length_x, 
                length_y=args.length_y,
                fields_data=fields_data,
                field_type=args.field_type
            )
    else:
        # No fields, create basic animation
        create_animation(
            df=df, 
            output_path=args.output, 
            fps=args.fps, 
            radius=args.radius, 
            length_x=args.length_x, 
            length_y=args.length_y
        )


if __name__ == "__main__":
    main()


