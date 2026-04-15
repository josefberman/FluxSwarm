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
    Consolidate episode summary CSV files from all env_0_* folders in chronological order.
    Creates plots and an Excel file for env_0 only.
    Rows include every finished episode: RL writes 'status' as 'terminated',
    'truncated', or 'both'; all are kept in rewards and in the Excel export.

    Args:
        base_folder (str): Path to the folder containing env_0_YYYYMMDD_HHMMSS subfolders

    Returns:
        str: Path to the created Excel file
    """
    base_path = Path(base_folder)

    if not base_path.exists():
        raise ValueError(f"Folder does not exist: {base_folder}")

    # Pattern to match env_0 folders only
    folder_pattern = re.compile(r'env_0_(\d{8})_(\d{6})')

    folders = []
    for item in base_path.iterdir():
        if item.is_dir():
            match = folder_pattern.match(item.name)
            if match:
                date_str, time_str = match.group(1), match.group(2)
                try:
                    date_obj = datetime.strptime(f"{date_str}{time_str}", '%Y%m%d%H%M%S')
                    folders.append((date_obj, item))
                except ValueError:
                    print(f"Warning: Could not parse datetime from folder: {item.name}")

    if not folders:
        raise ValueError(f"No env_0_YYYYMMDD_HHMMSS folders found in {base_folder}")

    folders.sort(key=lambda x: x[0])
    print(f"Found {len(folders)} env_0 folder(s) (chronological order):")
    for date_obj, folder_path in folders:
        print(f"  {folder_path.name}")

    dfs = []
    for date_obj, folder_path in folders:
        csv_files = list(folder_path.glob("episodes_summary_*.csv"))
        if not csv_files:
            print(f"  Warning: No episodes_summary_*.csv found in {folder_path.name}")
            continue
        if len(csv_files) > 1:
            print(f"  Warning: Multiple CSV files in {folder_path.name}, using first: {csv_files[0].name}")
        try:
            df = pd.read_csv(csv_files[0])
            print(f"  Loaded {len(df)} episodes from {folder_path.name}")
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {csv_files[0]}: {e}")

    if not dfs:
        raise ValueError("No CSV files were successfully loaded")

    combined = pd.concat(dfs, ignore_index=True)
    if 'episode' in combined.columns:
        combined['episode'] = range(1, len(combined) + 1)
    if "status" in combined.columns:
        vc = combined["status"].value_counts().to_dict()
        print(f"  Outcomes (terminated / truncated / both): {vc}")
    print(f"  Total episodes: {len(combined)}")

    # Wrap in a list so the rest of the function works with its env_dataframes loop
    env_dataframes = [combined]

    if not env_dataframes:
        raise ValueError("No CSV files were successfully loaded")
    
    # Identify all reward columns (cumulative rewards start with 'cum_' or contain 'reward')
    sample_df = env_dataframes[0]
    all_columns = sample_df.columns.tolist()
    
    # Find columns that start with 'cum_' or contain 'reward'
    _non_reward = {'episode', 'steps', 'ended_terminated', 'ended_truncated'}
    reward_columns = []
    for col in all_columns:
        col_lower = col.lower()
        if col_lower.startswith('cum_') or 'reward' in col_lower:
            if col not in _non_reward:
                reward_columns.append(col)

    if not reward_columns:
        print("\nWarning: No reward columns found. Looking for numeric columns...")
        reward_columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        for _exc in _non_reward:
            if _exc in reward_columns:
                reward_columns.remove(_exc)

    has_steps = all('steps' in df.columns for df in env_dataframes)
    
    print(f"\nFound {len(reward_columns)} reward types: {reward_columns}")
    
    max_episodes = len(env_dataframes[0])
    print(f"Total episodes: {max_episodes}")
    status_counts = _summarize_status_counts(env_dataframes)
    if status_counts:
        print(f"Episode outcomes: {status_counts}")
    
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
    norm_plot_data = {}   # reward / steps (only populated when steps column is present)
    df0 = env_dataframes[0]
    window_size = 5
    episodes = np.arange(1, max_episodes + 1)

    for reward_col in reward_columns:
        if reward_col not in df0.columns:
            continue

        series = pd.Series(df0[reward_col].values)
        mean_rewards = series.rolling(window=window_size, min_periods=1).mean().values
        std_rewards  = series.rolling(window=window_size, min_periods=1).std(ddof=0).fillna(0).values

        reward_plot_data[reward_col] = {
            'episodes': episodes,
            'mean': mean_rewards,
            'std': std_rewards,
        }

        if has_steps and 'steps' in df0.columns:
            steps = df0['steps'].replace(0, np.nan).values.astype(float)
            norm_series = pd.Series(df0[reward_col].values / steps)
            norm_plot_data[reward_col] = {
                'episodes': episodes,
                'mean': norm_series.rolling(window=window_size, min_periods=1).mean().values,
                'std':  norm_series.rolling(window=window_size, min_periods=1).std(ddof=0).fillna(0).values,
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
        ax.set_xlabel('Episode', fontsize=11)
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

            ax.set_title(f'{_pretty_name(reward_col)}', fontweight='bold')
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

    # Step-normalized combined plot (only when steps column is available)
    if has_steps and len(norm_plot_data) > 0:
        norm_cols = [col for col in combined_reward_columns if col in norm_plot_data]
        if norm_cols:
            fig, axes = plt.subplots(
                nrows=len(norm_cols),
                ncols=1,
                figsize=(11, 3.2 * len(norm_cols))
            )
            if len(norm_cols) == 1:
                axes = [axes]

            for idx, reward_col in enumerate(norm_cols):
                ax = axes[idx]
                pd_entry = norm_plot_data[reward_col]
                episodes = pd_entry['episodes']
                mean_n = pd_entry['mean']
                std_n = pd_entry['std']

                key = reward_col.lower().replace('cum_', '')
                line_color = color_map.get(key, 'black')

                ax.plot(episodes, mean_n, color=line_color, linewidth=0.8)
                ax.fill_between(
                    episodes,
                    mean_n - std_n,
                    mean_n + std_n,
                    alpha=0.18,
                    color=line_color,
                    edgecolor='none'
                )

                pretty = _pretty_name(reward_col)
                ax.set_title(f'{pretty} per episode duration', fontweight='bold')
                ax.set_xlabel('Episode')
                ax.set_ylabel(f'{pretty} / duration')
                ax.grid(True, alpha=0.25, linewidth=0.5, linestyle='--')
                ax.set_axisbelow(True)

            plt.tight_layout()

            norm_png = base_path / "rewards_combined_per_step.png"
            plt.savefig(norm_png, dpi=300, bbox_inches='tight', format='png')
            print(f"Saved per-step PNG plot: {norm_png}")

            norm_pdf = base_path / "rewards_combined_per_step.pdf"
            plt.savefig(norm_pdf, bbox_inches='tight', format='pdf')
            print(f"Saved per-step PDF plot: {norm_pdf}")
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
