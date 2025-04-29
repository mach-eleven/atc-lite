import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

# JC8D style: ocean palette, bold lines, grid, large fonts, clean look
sns.set(style="whitegrid", context="notebook", palette="ocean", font_scale=1.3)
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.facecolor"] = "#f7fbfd"

def find_log_files(root_dir):
    return list(Path(root_dir).rglob("evaluation_log.csv"))

def load_log(log_path):
    df = pd.read_csv(log_path)
    df['run'] = log_path.parent.name
    return df

def plot_single_run(df, run_name, output_dir):
    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    # Total Reward
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['episode'], y=df['total_reward'], label='Total Reward (raw)', alpha=0.3, linewidth=1.5)
    sns.lineplot(x=df['episode'], y=df['total_reward'].rolling(20).mean(), label='Total Reward (smoothed)', linewidth=2.5)
    plt.title(f"Total Reward per Episode: {run_name}")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(csv_dir / f"{run_name}_reward.png")
    plt.close()

    # Success Rate
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['episode'], y=df['success_rate'], label='Success Rate (raw)', alpha=0.3, linewidth=1.5)
    sns.lineplot(x=df['episode'], y=df['success_rate'].rolling(20).mean(), label='Success Rate (smoothed)', linewidth=2.5)
    plt.title(f"Success Rate per Episode: {run_name}")
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(csv_dir / f"{run_name}_success_rate.png")
    plt.close()

    # Reward Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_reward'], bins=30, kde=True, color=sns.color_palette("ocean")[2])
    plt.title(f"Reward Distribution: {run_name}")
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(csv_dir / f"{run_name}_reward_hist.png")
    plt.close()

    # Rolling mean/median
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['episode'], y=df['total_reward'].rolling(20).mean(), label='Rolling Mean (20)', linewidth=2.5)
    sns.lineplot(x=df['episode'], y=df['total_reward'].rolling(20).median(), label='Rolling Median (20)', linewidth=2.5, linestyle='--')
    plt.title(f"Rolling Mean/Median Reward: {run_name}")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(csv_dir / f"{run_name}_rolling_mean_median.png")
    plt.close()

    # Success/Failure bar
    if 'success_rate' in df.columns:
        num_success = int(df['success_rate'].sum())
        num_failure = len(df) - num_success
        plt.figure(figsize=(6, 6))
        sns.barplot(x=['Success', 'Failure'], y=[num_success, num_failure], palette='ocean')
        plt.title(f'Success vs Failure: {run_name}')
        plt.ylabel('Number of Episodes')
        plt.tight_layout()
        plt.savefig(csv_dir / f"{run_name}_success_failure.png")
        plt.close()

    # Pairplot for reward components
    reward_components = [col for col in df.columns if col not in ['episode', 'total_reward', 'success_rate', 'run']]
    if len(reward_components) > 1:
        sns.pairplot(df[reward_components], corner=True, kind='reg', plot_kws={'line_kws':{'color':'#005377'}})
        plt.suptitle(f"Reward Components Correlation: {run_name}", y=1.02)
        plt.savefig(csv_dir / f"{run_name}_components_pairplot.png")
        plt.close()

    # Each reward component as a separate image
    for comp in reward_components:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=df['episode'], y=df[comp], label=comp)
        plt.title(f"{comp} per Episode: {run_name}")
        plt.xlabel('Episode')
        plt.ylabel(comp)
        plt.legend()
        plt.tight_layout()
        plt.savefig(csv_dir / f"{run_name}_{comp}.png")
        plt.close()

    # Heatmap of reward components (episodes x components)
    if len(reward_components) > 1:
        plt.figure(figsize=(14, 6))
        data = df[reward_components].T
        sns.heatmap(data, cmap='ocean', cbar=True)
        plt.title(f"Reward Components Heatmap: {run_name}")
        plt.xlabel('Episode')
        plt.ylabel('Component')
        plt.tight_layout()
        plt.savefig(csv_dir / f"{run_name}_components_heatmap.png")
        plt.close()

def plot_aggregate(runs, output_dir):
    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.concat(runs)
    df_grouped = df_all.groupby('episode').mean().reset_index()
    # Average Total Reward
    plt.figure(figsize=(12, 6))
    plt.title("Average Total Reward per Episode (All Runs)")
    plt.plot(df_grouped['episode'], df_grouped['total_reward'], label='Avg Total Reward', color='royalblue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(csv_dir / "aggregate_reward.png")
    plt.close()

    # Average Success Rate
    plt.figure(figsize=(12, 6))
    plt.title("Average Success Rate per Episode (All Runs)")
    plt.plot(df_grouped['episode'], df_grouped['success_rate'], label='Avg Success Rate', color='seagreen')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(csv_dir / "aggregate_success_rate.png")
    plt.close()

    # Each reward component as a separate image
    reward_components = [col for col in df_grouped.columns if col not in ['episode', 'total_reward', 'success_rate', 'run']]
    for comp in reward_components:
        plt.figure(figsize=(12, 6))
        plt.title(f"Average {comp} per Episode (All Runs)")
        plt.plot(df_grouped['episode'], df_grouped[comp], label=f'Avg {comp}')
        plt.xlabel('Episode')
        plt.ylabel(comp)
        plt.legend()
        plt.tight_layout()
        plt.savefig(csv_dir / f"aggregate_{comp}.png")
        plt.close()

def plot_tensorboard_scalars(tb_dir, run_name, output_dir, smoothing=10):
    tb_dir = Path(tb_dir)
    tb_out = output_dir / 'tensorboard'
    tb_out.mkdir(parents=True, exist_ok=True)
    if not tb_dir.exists() or not tb_dir.is_dir():
        print(f"No tensorboard directory found for {run_name}.")
        return
    event_files = list(tb_dir.glob("events.out*"))
    if not event_files:
        print(f"No TensorBoard event files found in {tb_dir}.")
        return
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"No scalar tags found in TensorBoard logs for {run_name}.")
        return
    # Collect all scalars for a dataframe
    tb_scalars = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        tb_scalars[tag] = pd.Series(values, index=steps)
        # Plot each scalar
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=steps, y=values, label=f"{tag} (raw)", alpha=0.3, linewidth=1.5)
        if len(values) > smoothing:
            from scipy.ndimage import gaussian_filter1d
            values_smooth = gaussian_filter1d(values, sigma=smoothing)
            sns.lineplot(x=steps, y=values_smooth, label=f"{tag} (smoothed)", linewidth=2.5)
        plt.title(f"TensorBoard: {tag} ({run_name})")
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.legend()
        plt.tight_layout()
        fname = tag.replace('/', '_')
        plt.savefig(tb_out / f"{run_name}_tensorboard_{fname}.png")
        plt.close()
    print(f"Saved TensorBoard plots for {run_name} to {tb_out}")
    # Correlation heatmap for all scalars
    if len(tb_scalars) > 1:
        tb_df = pd.DataFrame(tb_scalars)
        tb_df = tb_df.interpolate().fillna(method='bfill').fillna(method='ffill')
        corr = tb_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='ocean', fmt='.2f', linewidths=0.5)
        plt.title(f"TensorBoard Scalar Correlation: {run_name}")
        plt.tight_layout()
        plt.savefig(tb_out / f"{run_name}_tensorboard_correlation.png")
        plt.close()
    # Pairplot for all scalars
    if len(tb_scalars) > 1:
        tb_df = pd.DataFrame(tb_scalars)
        tb_df = tb_df.interpolate().fillna(method='bfill').fillna(method='ffill')
        sns.pairplot(tb_df, corner=True, kind='reg', plot_kws={'line_kws':{'color':'#005377'}})
        plt.suptitle(f"TensorBoard Scalars Pairplot: {run_name}", y=1.02)
        plt.savefig(tb_out / f"{run_name}_tensorboard_pairplot.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize ATC-lite training logs.")
    parser.add_argument("viz_dir", type=str, help="Path to directory containing evaluation_log.csv files")
    parser.add_argument('--smoothing', type=int, default=10, help='Smoothing window size')
    args = parser.parse_args()

    log_files = list(Path(args.viz_dir).rglob("evaluation_log.csv"))
    if not log_files:
        print("No evaluation_log.csv files found.")
        return

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    runs = []
    for log_file in log_files:
        df = load_log(log_file)
        run_name = log_file.parent.name
        plot_single_run(df, run_name, output_dir)
        runs.append(df)
        # --- TensorBoard plotting ---
        tb_dir = log_file.parent / "tensorboard"
        if tb_dir.exists():
            plot_tensorboard_scalars(tb_dir, run_name, output_dir, smoothing=args.smoothing)

    if len(runs) > 1:
        plot_aggregate(runs, output_dir)

    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
