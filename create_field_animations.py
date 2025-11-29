"""
Helper script to create field animations after training.
This script automatically finds the fields files and creates three animations 
(vx, vy, p) with the proper simulation dimensions.
"""
import argparse
import os
import glob
import numpy as np
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Create field animations from training run"
    )
    parser.add_argument("--run_folder", required=True, help="Path to run folder (e.g., ../runs/2025-1-1_12-30-45)")
    parser.add_argument("--subfolder", default="MOMAPPO", help="Subfolder within run (default: MOMAPPO)")
    parser.add_argument("--length_x", type=float, default=100.0, help="Simulation domain length in x (default: 100.0)")
    parser.add_argument("--length_y", type=float, default=4.0, help="Simulation domain length in y (default: 4.0)")
    parser.add_argument("--radius", type=float, default=0.25, help="Circle radius (default: 0.25)")
    
    args = parser.parse_args()
    
    # Find the most recent MOMAPPO run folder
    momappo_folders = sorted(glob.glob(f"{args.run_folder}/{args.subfolder}/*"))
    if not momappo_folders:
        print(f"Error: No {args.subfolder} folders found in {args.run_folder}")
        return
    
    latest_folder = momappo_folders[-2]
    print(f"Using folder: {latest_folder}")
    
    # Find locations.csv
    locations_csv = f"{latest_folder}/locations.csv"
    if not os.path.exists(locations_csv):
        print(f"Error: locations.csv not found at {locations_csv}")
        return
    
    # Find fields directory
    fields_dir = f"{args.run_folder}/fields"
    if not os.path.isdir(fields_dir):
        print(f"Error: Fields directory not found: {fields_dir}")
        print("Make sure you ran training with save_fields=True")
        return
    
    # Check for step files
    fields_pattern = os.path.join(fields_dir, "step_*.npz")
    fields_files = sorted(glob.glob(fields_pattern))
    if not fields_files:
        print(f"Error: No step_*.npz files found in {fields_dir}")
        print("Make sure you ran training with save_fields=True")
        return
    
    print(f"Found {len(fields_files)} field step files in {fields_dir}")
    
    # Output path
    output_base = f"{latest_folder}/animation"
    
    # Pass directory directly to animate_locations.py (it can handle it now)
    cmd = (
        f'python animate_locations.py '
        f'--csv "{locations_csv}" '
        f'--output "{output_base}.mp4" '
        f'--fields "{fields_dir}" '
        f'--length_x {args.length_x} '
        f'--length_y {args.length_y} '
        f'--radius {args.radius}'
    )
    
    print(f"\nRunning command:\n{cmd}\n")
    os.system(cmd)
    
    print(f"\nAnimations created:")
    print(f"  {output_base}_vx.mp4  (X-velocity)")
    print(f"  {output_base}_vy.mp4  (Y-velocity)")
    print(f"  {output_base}_p.mp4   (Pressure)")


if __name__ == "__main__":
    main()

