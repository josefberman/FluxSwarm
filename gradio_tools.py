import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gradio as gr
from argparse import ArgumentParser


def process_csv(csv_file):
    # Read the uploaded CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Take the top 5 rows for display
    top5 = df.head()

    # Identify all paired location_i_x / location_i_y columns
    pairs = []
    for col in df.columns:
        m = re.match(r'location_(\d+)_x$', col)
        if m:
            idx = m.group(1)
            y_col = f'location_{idx}_y'
            if y_col in df.columns:
                pairs.append(idx)
    # Sort numerically
    pairs = sorted(pairs, key=lambda x: int(x))

    # First 3000-row scatter
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    subset1 = df.iloc[:1000]
    for idx in pairs:
        ax1.scatter(
            subset1[f'location_{idx}_x'],
            subset1[f'location_{idx}_y'],
            label=f'Loc {idx}', s=1
        )
    ax1.set_title('First 1000 Rows')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.legend(fontsize='small', ncol=2)

    # Last 3000-row scatter
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    subset2 = df.iloc[-1000:]
    for idx in pairs:
        ax2.scatter(
            subset2[f'location_{idx}_x'],
            subset2[f'location_{idx}_y'],
            label=f'Loc {idx}', s=1
        )
    ax2.set_title('Last 1000 Rows')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.legend(fontsize='small', ncol=2)

    return top5, fig1, fig2


def visualize_rewards(csv_file, window_size):
    # If no file yet, return empty plots
    if csv_file is None:
        return None, None

    # 1. Load and sort
    df = pd.read_csv(csv_file.name if hasattr(csv_file, "name") else csv_file)
    df = df.sort_values("timestep")

    # 2. First plot: raw reward
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df["timestep"], df["reward"], marker=None, linestyle="-")
    ax1.set_title("Reward vs. Timestep")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Reward")
    ax1.grid(True)

    # 3. Compute moving average with the user-selected window
    df["reward_avg"] = df["reward"].rolling(window=window_size, min_periods=1).mean()

    # 4. Second plot: averaged reward
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df["timestep"], df["reward_avg"], marker=None, linestyle="-")
    ax2.set_title(f"Reward vs. Timestep ({window_size}-Step Moving Average)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Avg. Reward")
    ax2.grid(True)

    return fig1, fig2


def location_plot():
    # Build the Gradio interface with side-by-side plots
    with gr.Blocks() as demo:
        gr.Markdown("## CSV Scatter-Plot Explorer")
        file_input = gr.File(label="Upload your CSV file")
        table_out = gr.Dataframe(label="Top 5 Rows", datatype="pandas")
        plot1 = gr.Plot(label="Scatter: First 3000 Rows")
        plot2 = gr.Plot(label="Scatter: Last 3000 Rows")
        # Trigger processing on upload
        file_input.upload(
            fn=process_csv,
            inputs=file_input,
            outputs=[table_out, plot1, plot2]
        )
    demo.launch()


def reward_plot():
    with gr.Blocks() as demo:
        gr.Markdown("## 📈 Reward Over Time Explorer")
        with gr.Row():
            uploader = gr.File(label="Upload CSV (`timestep`,`reward`)")

            avg_slider = gr.Slider(
                minimum=1, maximum=50, step=1, value=5,
                label="Moving-Average Window Size"
            )

        raw_plot = gr.Plot(label="Raw Reward")
        smooth_plot = gr.Plot(label="Smoothed Reward")

        # Whenever the file is uploaded or the slider moves, re-run
        uploader.upload(
            fn=visualize_rewards,
            inputs=[uploader, avg_slider],
            outputs=[raw_plot, smooth_plot]
        )
        avg_slider.change(
            fn=visualize_rewards,
            inputs=[uploader, avg_slider],
            outputs=[raw_plot, smooth_plot]
        )
    demo.launch()


def visualize_locations_over_time(csv_file, window_size):
    # If no file yet, return empty plots
    if csv_file is None:
        return None, None, None, None

    # Load and (if present) sort by timestep
    df = pd.read_csv(csv_file.name if hasattr(csv_file, "name") else csv_file)
    if "timestep" in df.columns:
        df = df.sort_values("timestep")

    # Coerce window to a valid integer >= 1
    try:
        window_size = int(window_size)
    except Exception:
        window_size = 1
    if window_size < 1:
        window_size = 1

    # Time axis: prefer explicit 'timestep' if present, otherwise use the row index
    t = df["timestep"] if "timestep" in df.columns else pd.RangeIndex(start=0, stop=len(df))

    # Identify all paired location_i_x / location_i_y columns
    pairs = []
    for col in df.columns:
        m = re.match(r'location_(\d+)_x$', col)
        if m:
            idx = m.group(1)
            y_col = f'location_{idx}_y'
            if y_col in df.columns:
                pairs.append(idx)
    pairs = sorted(pairs, key=lambda x: int(x))

    # Plot X for each member over time (smoothed)
    fig_x = Figure(figsize=(12, 4))
    ax_x = fig_x.add_subplot(111)
    for idx in pairs:
        series_x = df[f'location_{idx}_x'].rolling(window=window_size, min_periods=1).mean()
        ax_x.plot(t, series_x, linewidth=1, label=f'Loc {idx}')
    ax_x.set_title(f'X over Timesteps (per Member) — {window_size}-Step MA')
    ax_x.set_xlabel('Timestep')
    ax_x.set_ylabel('X')
    ax_x.grid(True)
    if len(pairs) <= 15:
        ax_x.legend(fontsize='small', ncol=2)

    # Plot Y for each member over time (smoothed)
    fig_y = Figure(figsize=(12, 4))
    ax_y = fig_y.add_subplot(111)
    for idx in pairs:
        series_y = df[f'location_{idx}_y'].rolling(window=window_size, min_periods=1).mean()
        ax_y.plot(t, series_y, linewidth=1, label=f'Loc {idx}')
    ax_y.set_title(f'Y over Timesteps (per Member) — {window_size}-Step MA')
    ax_y.set_xlabel('Timestep')
    ax_y.set_ylabel('Y')
    ax_y.grid(True)
    if len(pairs) <= 15:
        ax_y.legend(fontsize='small', ncol=2)

    # Average X and Y over members, then smooth over time
    x_cols = [f'location_{idx}_x' for idx in pairs]
    y_cols = [f'location_{idx}_y' for idx in pairs]
    avg_x = df[x_cols].mean(axis=1) if x_cols else pd.Series([], dtype=float)
    avg_y = df[y_cols].mean(axis=1) if y_cols else pd.Series([], dtype=float)
    avg_x = avg_x.rolling(window=window_size, min_periods=1).mean()
    avg_y = avg_y.rolling(window=window_size, min_periods=1).mean()

    fig_avg_x = Figure(figsize=(12, 4))
    ax_avg_x = fig_avg_x.add_subplot(111)
    ax_avg_x.plot(t, avg_x, linewidth=1.5)
    ax_avg_x.set_title(f'Average X over Timesteps — {window_size}-Step MA')
    ax_avg_x.set_xlabel('Timestep')
    ax_avg_x.set_ylabel('Avg X')
    ax_avg_x.grid(True)

    fig_avg_y = Figure(figsize=(12, 4))
    ax_avg_y = fig_avg_y.add_subplot(111)
    ax_avg_y.plot(t, avg_y, linewidth=1.5)
    ax_avg_y.set_title(f'Average Y over Timesteps — {window_size}-Step MA')
    ax_avg_y.set_xlabel('Timestep')
    ax_avg_y.set_ylabel('Avg Y')
    ax_avg_y.grid(True)

    return fig_x, fig_y, fig_avg_x, fig_avg_y


def location_timeseries_plot():
    with gr.Blocks() as demo:
        gr.Markdown("## 📊 Location Time-Series Explorer")
        with gr.Row():
            uploader = gr.File(label="Upload CSV with location_i_x / location_i_y columns")
            avg_slider = gr.Slider(
                minimum=1, maximum=300, step=1, value=5,
                label="Moving-Average Window Size"
            )

        plot_x = gr.Plot(label="X over Timesteps (per Member)")
        plot_y = gr.Plot(label="Y over Timesteps (per Member)")
        plot_avg_x = gr.Plot(label="Average X over Timesteps")
        plot_avg_y = gr.Plot(label="Average Y over Timesteps")

        uploader.upload(
            fn=visualize_locations_over_time,
            inputs=[uploader, avg_slider],
            outputs=[plot_x, plot_y, plot_avg_x, plot_avg_y]
        )
        avg_slider.change(
            fn=visualize_locations_over_time,
            inputs=[uploader, avg_slider],
            outputs=[plot_x, plot_y, plot_avg_x, plot_avg_y]
        )
    demo.launch()


def _make_overview_plots(actions: pd.DataFrame, rewards: pd.DataFrame,
                         locations: pd.DataFrame, velocities: pd.DataFrame,
                         t_min: int, t_max: int):
    # Determine time range and slice
    def clip_df(df: pd.DataFrame):
        if df is None or 'timestep' not in df.columns:
            return None
        mask = (df['timestep'] >= t_min) & (df['timestep'] <= t_max)
        return df.loc[mask]

    a = clip_df(actions)
    r = clip_df(rewards)
    l = clip_df(locations)
    v = clip_df(velocities)

    figs = []
    # Rewards
    if r is not None:
        fig_r = Figure(figsize=(12, 2.5))
        ax_r = fig_r.add_subplot(111)
        ax_r.plot(r['timestep'], r['reward'], lw=1)
        ax_r.set_title('Reward')
        ax_r.set_xlabel('timestep')
        ax_r.set_ylabel('reward')
        ax_r.grid(True)
        figs.append(fig_r)
    else:
        figs.append(None)

    # Locations X and Y (stacked as one figure with two subplots)
    if l is not None:
        fig_loc = Figure(figsize=(12, 4))
        axx = fig_loc.add_subplot(211)
        axy = fig_loc.add_subplot(212)
        member_ids = sorted({int(re.findall(r"location_(\d+)_x", c)[0]) for c in l.columns if re.match(r"location_\d+_x", c)})
        for mid in member_ids:
            axx.plot(l['timestep'], l[f'location_{mid}_x'], lw=0.8)
            axy.plot(l['timestep'], l[f'location_{mid}_y'], lw=0.8)
        axx.set_title('Locations X')
        axy.set_title('Locations Y')
        for ax in (axx, axy):
            ax.set_xlabel('timestep')
            ax.grid(True)
        axx.set_ylabel('x')
        axy.set_ylabel('y')
        figs.append(fig_loc)
    else:
        figs.append(None)

    # Velocities X and Y
    if v is not None:
        fig_vel = Figure(figsize=(12, 4))
        axvx = fig_vel.add_subplot(211)
        axvy = fig_vel.add_subplot(212)
        member_ids = sorted({int(re.findall(r"velocity_(\d+)_x", c)[0]) for c in v.columns if re.match(r"velocity_\d+_x", c)})
        for mid in member_ids:
            axvx.plot(v['timestep'], v[f'velocity_{mid}_x'], lw=0.8)
            axvy.plot(v['timestep'], v[f'velocity_{mid}_y'], lw=0.8)
        axvx.set_title('Velocities X')
        axvy.set_title('Velocities Y')
        for ax in (axvx, axvy):
            ax.set_xlabel('timestep')
            ax.grid(True)
        axvx.set_ylabel('vx')
        axvy.set_ylabel('vy')
        figs.append(fig_vel)
    else:
        figs.append(None)

    # Actions X and Y
    if a is not None:
        fig_act = Figure(figsize=(12, 4))
        axax = fig_act.add_subplot(211)
        axay = fig_act.add_subplot(212)
        member_ids = sorted({int(re.findall(r"action_(\d+)_x", c)[0]) for c in a.columns if re.match(r"action_\d+_x", c)})
        for mid in member_ids:
            axax.plot(a['timestep'], a[f'action_{mid}_x'], lw=0.8)
            axay.plot(a['timestep'], a[f'action_{mid}_y'], lw=0.8)
        axax.set_title('Actions X')
        axay.set_title('Actions Y')
        for ax in (axax, axay):
            ax.set_xlabel('timestep')
            ax.grid(True)
        axax.set_ylabel('ax')
        axay.set_ylabel('ay')
        figs.append(fig_act)
    else:
        figs.append(None)

    return tuple(figs)


def _make_trajectory_plot(locations: pd.DataFrame, t_min: int, t_max: int):
    if locations is None or 'timestep' not in locations.columns:
        return None
    mask = (locations['timestep'] >= t_min) & (locations['timestep'] <= t_max)
    l = locations.loc[mask]
    fig = Figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    member_ids = sorted({int(re.findall(r"location_(\d+)_x", c)[0]) for c in l.columns if re.match(r"location_\d+_x", c)})
    for mid in member_ids:
        ax.scatter(l[f'location_{mid}_x'], l[f'location_{mid}_y'], s=1)
        ax.scatter(l[f'location_{mid}_x'].iloc[-1], l[f'location_{mid}_y'].iloc[-1], s=8, c='k')
    ax.set_title('Trajectories (Y vs X)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    # ax.set_aspect('equal', adjustable='box')
    return fig


def run_overview_tool():
    with gr.Blocks() as demo:
        gr.Markdown('## Run Overview and Trajectory Explorer')
        with gr.Row():
            files = gr.Files(label='Select run folder (upload the 4 CSVs)', file_count='multiple', type='filepath', file_types=['.csv'])
        with gr.Row():
            t_min = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label='Timestep min')
            t_max = gr.Slider(minimum=0, maximum=1000, value=100, step=1, label='Timestep max')

        plot_rewards = gr.Plot(label='Rewards')
        plot_locations = gr.Plot(label='Locations')
        plot_velocities = gr.Plot(label='Velocities')
        plot_actions = gr.Plot(label='Actions')
        traj_plot = gr.Plot(label='Trajectories (Y vs X)')

        def load_and_plot(filepaths, tmin_val, tmax_val):
            actions = rewards = locations = velocities = None
            if filepaths is None:
                filepaths = []
            # Map basenames to paths
            lower_map = {os.path.basename(p).lower(): p for p in filepaths}
            if 'forces.csv' in lower_map:
                try:
                    actions = pd.read_csv(lower_map['forces.csv'])
                except Exception:
                    actions = None
            elif 'actions.csv' in lower_map:
                try:
                    actions = pd.read_csv(lower_map['actions.csv'])
                except Exception:
                    actions = None
            if 'rewards.csv' in lower_map:
                try:
                    rewards = pd.read_csv(lower_map['rewards.csv'])
                except Exception:
                    rewards = None
            if 'locations.csv' in lower_map:
                try:
                    locations = pd.read_csv(lower_map['locations.csv'])
                except Exception:
                    locations = None
            if 'velocities.csv' in lower_map:
                try:
                    velocities = pd.read_csv(lower_map['velocities.csv'])
                except Exception:
                    velocities = None

            # Determine global timestep range from available files
            mins = []
            maxs = []
            for df in (actions, rewards, locations, velocities):
                if df is not None and 'timestep' in df.columns:
                    mins.append(df['timestep'].min())
                    maxs.append(df['timestep'].max())
            if mins and maxs:
                global_min = int(np.floor(min(mins)))
                global_max = int(np.ceil(max(maxs)))
            else:
                global_min, global_max = 0, 100

            # Current selected range (clamped to available)
            sel_min = int(np.clip(int(tmin_val), global_min, global_max))
            sel_max = int(np.clip(int(tmax_val), global_min, global_max))
            if sel_min > sel_max:
                sel_min, sel_max = sel_max, sel_min

            figs = _make_overview_plots(actions, rewards, locations, velocities, sel_min, sel_max)
            traj = _make_trajectory_plot(locations, sel_min, sel_max)

            # Update the slider bounds and current selection
            tmin_update = gr.update(minimum=global_min, maximum=global_max, value=sel_min)
            tmax_update = gr.update(minimum=global_min, maximum=global_max, value=sel_max)
            return (*figs, traj, tmin_update, tmax_update)

        def plot_only(filepaths, tmin_val, tmax_val):
            actions = rewards = locations = velocities = None
            if filepaths is None:
                filepaths = []
            lower_map = {os.path.basename(p).lower(): p for p in filepaths}
            if 'forces.csv' in lower_map:
                try:
                    actions = pd.read_csv(lower_map['forces.csv'])
                except Exception:
                    actions = None
            elif 'actions.csv' in lower_map:
                try:
                    actions = pd.read_csv(lower_map['actions.csv'])
                except Exception:
                    actions = None
            if 'rewards.csv' in lower_map:
                try:
                    rewards = pd.read_csv(lower_map['rewards.csv'])
                except Exception:
                    rewards = None
            if 'locations.csv' in lower_map:
                try:
                    locations = pd.read_csv(lower_map['locations.csv'])
                except Exception:
                    locations = None
            if 'velocities.csv' in lower_map:
                try:
                    velocities = pd.read_csv(lower_map['velocities.csv'])
                except Exception:
                    velocities = None

            sel_min = int(tmin_val)
            sel_max = int(tmax_val)
            if sel_min > sel_max:
                sel_min, sel_max = sel_max, sel_min

            figs = _make_overview_plots(actions, rewards, locations, velocities, sel_min, sel_max)
            traj = _make_trajectory_plot(locations, sel_min, sel_max)
            return (*figs, traj)

        files.change(
            fn=load_and_plot,
            inputs=[files, t_min, t_max],
            outputs=[plot_rewards, plot_locations, plot_velocities, plot_actions, traj_plot, t_min, t_max]
        )
        t_min.change(
            fn=plot_only,
            inputs=[files, t_min, t_max],
            outputs=[plot_rewards, plot_locations, plot_velocities, plot_actions, traj_plot]
        )
        t_max.change(
            fn=plot_only,
            inputs=[files, t_min, t_max],
            outputs=[plot_rewards, plot_locations, plot_velocities, plot_actions, traj_plot]
        )
    demo.launch()


TOOLS = {
    'location_plot': location_plot,
    'reward_plot': reward_plot,
    'location_timeseries_plot': location_timeseries_plot,
    'run_overview_tool': run_overview_tool
}

parser = ArgumentParser()
parser.add_argument('--tool', '-t', default='location_plot')
args = parser.parse_args()

TOOLS[args.tool]()
