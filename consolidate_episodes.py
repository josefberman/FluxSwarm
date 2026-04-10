import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def _add_episode_end_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive booleans from RL ``status`` so truncated and terminated episodes are explicit."""
    if "status" not in df.columns:
        return df
    out = df.copy()
    s = out["status"].astype(str).str.strip().str.lower()
    out["ended_terminated"] = s.isin(["terminated", "both"])
    out["ended_truncated"] = s.isin(["truncated", "both"])
    return out


def _summarize_status_counts(dfs) -> dict:
    counts = {}
    for df in dfs:
        if "status" not in df.columns:
            continue
        for k, v in df["status"].value_counts().items():
            counts[k] = counts.get(k, 0) + int(v)
    return counts


def consolidate_episodes(base_folder):
    """
    Consolidate episode summary CSV files from multiple training session folders.
    Creates plots showing mean and standard deviation of rewards across environments.
    Rows include every finished episode: RL writes 'status' as 'terminated',
    'truncated', or 'both'; all are kept in rewards and in the Excel export.
    
    Args:
        base_folder (str): Path to the folder containing env_0_YYYYMMDD_{id} folders
    
    Returns:
        str: Path to the created Excel file
    """
    base_path = Path(base_folder)
    
    if not base_path.exists():
        raise ValueError(f"Folder does not exist: {base_folder}")
    
    # Pattern to match folder names: env_{i}_YYYYMMDD_HHMMSS
    folder_pattern = re.compile(r'env_(\d+)_(\d{8})_(\d{6})')
    
    # Find all matching folders and group by environment number
    env_folders = {}  # {env_num: [(datetime, folder_path), ...]}
    for item in base_path.iterdir():
        if item.is_dir():
            match = folder_pattern.match(item.name)
            if match:
                env_num = int(match.group(1))
                date_str = match.group(2)
                time_str = match.group(3)
                try:
                    # Parse the datetime for sorting
                    datetime_str = f"{date_str}{time_str}"
                    date_obj = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                    if env_num not in env_folders:
                        env_folders[env_num] = []
                    env_folders[env_num].append((date_obj, item))
                except ValueError:
                    print(f"Warning: Could not parse datetime from folder: {item.name}")
    
    if not env_folders:
        raise ValueError(f"No folders matching 'env_{{i}}_YYYYMMDD_HHMMSS' pattern found in {base_folder}")
    
    # Sort folders within each environment chronologically
    for env_num in env_folders:
        env_folders[env_num].sort(key=lambda x: x[0])
    
    print(f"Found {len(env_folders)} different environments:")
    for env_num in sorted(env_folders.keys()):
        folder_count = len(env_folders[env_num])
        print(f"  - env_{env_num}: {folder_count} folder(s)")
    
    # Load and concatenate CSV files for each environment number
    env_dataframes = []  # List of dataframes, one per environment number
    for env_num in sorted(env_folders.keys()):
        env_dfs = []  # Dataframes for this environment number
        
        print(f"\nProcessing env_{env_num}:")
        for date_obj, folder_path in env_folders[env_num]:
            # Look for any episodes_summary_*.csv file in the folder
            csv_files = list(folder_path.glob("episodes_summary_*.csv"))
            
            if not csv_files:
                print(f"  Warning: No episodes_summary_*.csv file found in {folder_path.name}")
                continue
            
            if len(csv_files) > 1:
                print(f"  Warning: Multiple CSV files found in {folder_path.name}, using first one: {csv_files[0].name}")
            
            csv_path = csv_files[0]
            
            try:
                df = pd.read_csv(csv_path)
                print(f"  Loaded {len(df)} episodes from {folder_path.name}")
                env_dfs.append(df)
            except Exception as e:
                print(f"  Error loading {csv_path}: {e}")
        
        # Concatenate all dataframes for this environment chronologically
        if env_dfs:
            env_combined = pd.concat(env_dfs, ignore_index=True)
            # Renumber episodes sequentially
            if 'episode' in env_combined.columns:
                env_combined['episode'] = range(1, len(env_combined) + 1)
            print(f"  Total episodes for env_{env_num}: {len(env_combined)}")
            if "status" in env_combined.columns:
                vc = env_combined["status"].value_counts().to_dict()
                print(f"    Outcomes (terminated / truncated / both): {vc}")
            env_dataframes.append(env_combined)
    
    if not env_dataframes:
        raise ValueError("No CSV files were successfully loaded")
    
    # Identify all reward columns (cumulative rewards start with 'cum_' or contain 'reward')
    sample_df = env_dataframes[0]
    all_columns = sample_df.columns.tolist()
    
    # Find columns that start with 'cum_' or contain 'reward'
    reward_columns = []
    for col in all_columns:
        col_lower = col.lower()
        if col_lower.startswith('cum_') or 'reward' in col_lower:
            if col != 'episode':  # Exclude episode column
                reward_columns.append(col)
    
    if not reward_columns:
        print("\nWarning: No reward columns found. Looking for numeric columns...")
        reward_columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'episode' in reward_columns:
            reward_columns.remove('episode')
    
    print(f"\nFound {len(reward_columns)} reward types: {reward_columns}")
    
    # Determine the maximum number of episodes across all environments
    max_episodes = max(len(df) for df in env_dataframes)
    num_envs = len(env_dataframes)
    print(f"Maximum episodes: {max_episodes}")
    print(f"Number of environments: {num_envs}")
    status_counts = _summarize_status_counts(env_dataframes)
    if status_counts:
        print(
            "Episode outcomes across all envs (truncated and terminated rows all enter rewards): "
            f"{status_counts}"
        )
    
    # Create plots for each reward type with publication-quality formatting
    # Set matplotlib parameters for publication quality
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.5
    
    reward_plot_data = {}
    for reward_col in reward_columns:
        # Create a matrix to store rolling means: rows=episodes, cols=environments
        window_size = 5
        reward_matrix = np.full((max_episodes, len(env_dataframes)), np.nan)
        
        for env_idx, df in enumerate(env_dataframes):
            if reward_col in df.columns:
                # Compute rolling mean for this environment
                rewards = df[reward_col].values
                rolling_mean = pd.Series(rewards).rolling(window=window_size, min_periods=1, center=False).mean().values
                num_episodes = len(rolling_mean)
                reward_matrix[:num_episodes, env_idx] = rolling_mean
        
        # Compute mean and std of rolling means across environments for each episode
        mean_rewards = np.nanmean(reward_matrix, axis=1)
        std_rewards = np.nanstd(reward_matrix, axis=1)
        
        # Create global episode index so x-axis reflects parallel environment count.
        # Example: 70 episodes per env with 8 envs -> x-axis reaches 560.
        episodes = np.arange(1, max_episodes + 1) * num_envs
        reward_plot_data[reward_col] = {
            'episodes': episodes,
            'mean': mean_rewards,
            'std': std_rewards,
        }
        
        # Create the plot with single-column figure size (3.5 inches width for most journals)
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Use grayscale-friendly colors suitable for publication
        line_color = '#000000'  # Black for mean line
        fill_color = '#808080'  # Gray for std dev
        
        # Plot shaded standard deviation first (so it appears behind the line)
        ax.fill_between(episodes, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards,
                        alpha=0.25, 
                        color=fill_color,
                        edgecolor='none',
                        label='±1 SD')
        
        # Plot mean line on top
        ax.plot(episodes, mean_rewards, 
                linewidth=1.5, 
                color=line_color, 
                label='Rolling Mean (w=5)',
                solid_capstyle='round')
        
        # Format axes
        ax.set_xlabel('Episode (all environments)', fontsize=11)
        # Expand "cum" to "cumulative" for better readability
        ylabel = reward_col.replace('cum_', 'cumulative_').replace('_', ' ').title()
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel}', 
                    fontsize=12, pad=10)
        
        # Grid with subtle styling
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='--')
        ax.set_axisbelow(True)  # Grid behind data
        
        # Legend with frame
        ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fancybox=False)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Tighten layout
        plt.tight_layout()
        
        # Save in both PNG (high-res) and PDF (vector) formats
        plot_filename_base = f"{reward_col}_mean_std"
        
        # Save PNG at 300 DPI
        png_path = base_path / f"{plot_filename_base}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Saved PNG plot: {png_path}")
        
        # Save PDF (vector format, preferred for publications)
        pdf_path = base_path / f"{plot_filename_base}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        print(f"Saved PDF plot: {pdf_path}")
        
        plt.close()

    # Combined multi-panel plot with same consolidated logic (rolling mean across envs)
    if len(reward_plot_data) > 0:
        def _pretty_name(col_name: str) -> str:
            return col_name.replace('cum_', 'cumulative_').replace('_', ' ').title()

        color_map = {
            'progress': 'tab:purple',
            'location_progress': 'tab:purple',  # legacy column name
            'energy_efficiency': 'tab:green',
            'cohesion': 'tab:green',  # legacy column name
            'smoothness': 'tab:orange',
        }
        # ref_lines = {
        #     'progress': [400.0, 0.0],
        #     'energy_efficiency': [400.0, 0.0],
        #     'smoothness': [400.0, -400.0],
        # }

        combined_reward_columns = [
            col for col in reward_columns
            if col.lower() not in {'cum_total_reward', 'total_reward', 'reward'}
        ]
        if len(combined_reward_columns) == 0:
            combined_reward_columns = reward_columns

        fig, axes = plt.subplots(
            nrows=len(combined_reward_columns),
            ncols=1,
            figsize=(11, 3.2 * len(combined_reward_columns))
        )
        if len(combined_reward_columns) == 1:
            axes = [axes]

        for idx, reward_col in enumerate(combined_reward_columns):
            ax = axes[idx]
            plot_data = reward_plot_data[reward_col]
            episodes = plot_data['episodes']
            mean_rewards = plot_data['mean']
            std_rewards = plot_data['std']

            key = reward_col.lower().replace('cum_', '')
            line_color = color_map.get(key, 'black')

            ax.plot(episodes, mean_rewards, color=line_color, linewidth=0.8)
            ax.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.18,
                color=line_color,
                edgecolor='none'
            )

            # for y in ref_lines.get(key, []):
            #     ax.axhline(y, linestyle='dashed', color=line_color, linewidth=0.8, alpha=0.35)

            ax.set_title(f'{_pretty_name(reward_col)} over episodes', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel(_pretty_name(reward_col))
            ax.grid(True, alpha=0.25, linewidth=0.5, linestyle='--')
            ax.set_axisbelow(True)

        plt.tight_layout()

        combined_png = base_path / "rewards_combined_mean_std.png"
        plt.savefig(combined_png, dpi=300, bbox_inches='tight', format='png')
        print(f"Saved combined PNG plot: {combined_png}")

        combined_pdf = base_path / "rewards_combined_mean_std.pdf"
        plt.savefig(combined_pdf, bbox_inches='tight', format='pdf')
        print(f"Saved combined PDF plot: {combined_pdf}")
        plt.close()
    
    # Also create consolidated Excel file with all data (for reference)
    combined_df = pd.concat(env_dataframes, ignore_index=True)
    print(f"\nTotal rows after concatenation: {len(combined_df)}")
    
    # Renumber the episode column with ordered natural numbers
    if 'episode' in combined_df.columns:
        combined_df['episode'] = range(1, len(combined_df) + 1)
        print("Renumbered 'episode' column with sequential natural numbers")
    else:
        print("Warning: 'episode' column not found in the data")

    combined_df = _add_episode_end_columns(combined_df)
    if "ended_terminated" in combined_df.columns:
        nt = int(combined_df["ended_terminated"].sum())
        nu = int(combined_df["ended_truncated"].sum())
        print(
            f"Episode ends: {nt} with terminated=True, {nu} with truncated=True "
            "(episodes can be both if status is 'both')"
        )
    
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
