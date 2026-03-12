import argparse
import os
import glob
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Extract mean fluid velocity at y=2 (tube center) over time.")
    parser.add_argument("--run_folder", type=str, required=True, help="Path to run folder (e.g., runs/2025-1-1_12-30-45)")
    parser.add_argument("--output", type=str, default="center_velocity.csv", help="Output CSV file name")
    return parser.parse_args()

def load_fields_from_directory(fields_dir: str) -> dict:
    # Find all step files
    pattern = os.path.join(fields_dir, "step_*.npz")
    fields_files = sorted(glob.glob(pattern))
    
    if not fields_files:
        raise ValueError(f"No step_*.npz files found in {fields_dir}")
    
    print(f"Loading {len(fields_files)} field step files from {fields_dir}...")
    
    vx_list = []
    vy_list = []
    timesteps_list = []
    length_y = None
    
    for field_file in fields_files:
        data = np.load(field_file)
        vx_list.append(data['vx'])  # Shape: (x, y)
        vy_list.append(data['vy'])  # Shape: (x, y)
        timesteps_list.append(float(data['timestep']))
        
        # Store metadata from first file
        if length_y is None:
            length_y = float(data['length_y'])
            
    # Stack into arrays: (timesteps, x, y)
    vx_array = np.stack(vx_list, axis=0)
    vy_array = np.stack(vy_list, axis=0)
    timesteps_array = np.array(timesteps_list, dtype=np.float64)
    
    return {
        'vx': vx_array,  # Shape: (timesteps, x, y)
        'vy': vy_array,  # Shape: (timesteps, x, y)
        'timestep': timesteps_array,  # Array of timesteps
        'length_y': length_y
    }

def main():
    args = parse_args()
    
    fields_dir = os.path.join(args.run_folder, "fields")
    if not os.path.isdir(fields_dir):
        print(f"Error: Fields directory not found: {fields_dir}")
        return
        
    try:
        fields_data = load_fields_from_directory(fields_dir)
    except Exception as e:
        print(f"Error loading fields: {e}")
        return
        
    vx = fields_data['vx']
    vy = fields_data['vy']
    length_y = fields_data['length_y']
    timesteps = fields_data['timestep']
    
    y_dim = vx.shape[2] # Shape is (timesteps, x, y)
    
    # Calculate index for y = 2
    # Assuming the grid spans from 0 to length_y
    y_target = 2.0
    y_idx = int((y_target / length_y) * y_dim)
    
    # Ensure y_idx is within bounds
    y_idx = min(max(y_idx, 0), y_dim - 1)
    
    print(f"Simulation length_y: {length_y}")
    print(f"Grid y-dimension: {y_dim}")
    print(f"Extracting data at y={y_target} (index {y_idx})")
    
    # Extract velocities at y=2
    # Shape of vx_at_center will be (timesteps, x)
    vx_at_center = vx[:, :, y_idx]
    vy_at_center = vy[:, :, y_idx]
    
    # Calculate mean over x-axis (axis=1) for each timestep
    mean_vx = np.mean(vx_at_center, axis=1)
    mean_vy = np.mean(vy_at_center, axis=1)
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'timestep': timesteps,
        'mean_vx': mean_vx,
        'mean_vy': mean_vy
    })
    
    # Sort by timestep just in case
    df = df.sort_values('timestep').reset_index(drop=True)
    
    output_path = os.path.join(args.run_folder, args.output)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully saved mean velocities to {output_path}")
    print(df.head())

if __name__ == "__main__":
    main()
