import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime


def consolidate_episodes(base_folder):
    """
    Consolidate episode summary CSV files from multiple training session folders.
    
    Args:
        base_folder (str): Path to the folder containing env_0_YYYYMMDD_{id} folders
    
    Returns:
        str: Path to the created Excel file
    """
    base_path = Path(base_folder)
    
    if not base_path.exists():
        raise ValueError(f"Folder does not exist: {base_folder}")
    
    # Pattern to match folder names: env_0_YYYYMMDD_{id}
    folder_pattern = re.compile(r'env_0_(\d{8})_(.+)')
    
    # Find all matching folders with their dates
    folders_with_dates = []
    for item in base_path.iterdir():
        if item.is_dir():
            match = folder_pattern.match(item.name)
            if match:
                date_str = match.group(1)
                env_id = match.group(2)
                try:
                    # Parse the date for sorting
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    folders_with_dates.append((date_obj, date_str, env_id, item))
                except ValueError:
                    print(f"Warning: Could not parse date from folder: {item.name}")
    
    if not folders_with_dates:
        raise ValueError(f"No folders matching 'env_0_YYYYMMDD_{{id}}' pattern found in {base_folder}")
    
    # Sort by date chronologically
    folders_with_dates.sort(key=lambda x: x[0])
    
    print(f"Found {len(folders_with_dates)} folders to process:")
    for date_obj, date_str, env_id, folder_path in folders_with_dates:
        print(f"  - {folder_path.name} (Date: {date_str}, ID: {env_id})")
    
    # Load and concatenate CSV files
    all_dataframes = []
    for date_obj, date_str, env_id, folder_path in folders_with_dates:
        # Look for any episodes_summary_*.csv file in the folder
        csv_files = list(folder_path.glob("episodes_summary_*.csv"))
        
        if not csv_files:
            print(f"Warning: No episodes_summary_*.csv file found in {folder_path.name}")
            continue
        
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found in {folder_path.name}, using first one: {csv_files[0].name}")
        
        csv_path = csv_files[0]
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {csv_path.name}")
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    
    if not all_dataframes:
        raise ValueError("No CSV files were successfully loaded")
    
    # Concatenate all dataframes vertically
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nTotal rows after concatenation: {len(combined_df)}")
    
    # Renumber the episode column with ordered natural numbers
    if 'episode' in combined_df.columns:
        combined_df['episode'] = range(1, len(combined_df) + 1)
        print("Renumbered 'episode' column with sequential natural numbers")
    else:
        print("Warning: 'episode' column not found in the data")
    
    # Create output Excel file
    output_filename = "consolidated_episodes.xlsx"
    output_path = base_path / output_filename
    
    combined_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\nExcel file created: {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python consolidate_episodes.py <folder_path>")
        print("\nExample:")
        print("  python consolidate_episodes.py ./training_sessions")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    try:
        output_file = consolidate_episodes(folder_path)
        print(f"\n✓ Success! Consolidated data saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
