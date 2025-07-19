import re
import pandas as pd
import matplotlib.pyplot as plt
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
    subset1 = df.iloc[:3000]
    for idx in pairs:
        ax1.scatter(
            subset1[f'location_{idx}_x'],
            subset1[f'location_{idx}_y'],
            label=f'Loc {idx}', s=1
        )
    ax1.set_title('First 3000 Rows')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.legend(fontsize='small', ncol=2)

    # Last 3000-row scatter
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    subset2 = df.iloc[-3000:]
    for idx in pairs:
        ax2.scatter(
            subset2[f'location_{idx}_x'],
            subset2[f'location_{idx}_y'],
            label=f'Loc {idx}', s=1
        )
    ax2.set_title('Last 3000 Rows')
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


TOOLS = {
    'location_plot': location_plot,
    'reward_plot': reward_plot
}

parser = ArgumentParser()
parser.add_argument('--tool', '-t', default='location_plot')
args = parser.parse_args()

TOOLS[args.tool]()
