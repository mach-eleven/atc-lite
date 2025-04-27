"""
Generate rich visualizations of training performance for ATC-Lite model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

# Set up nice looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate visualizations for ATC training data')
    parser.add_argument('--log_dir', type=str, default='logs/ppo/AtcGym-v0', help='Directory containing training logs')
    parser.add_argument('--run', type=str, default=None, help='Specific run to visualize (e.g., "run_6")')
    parser.add_argument('--compare', action='store_true', help='Compare multiple runs')
    parser.add_argument('--smoothing', type=int, default=10, help='Smoothing window size')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    return parser.parse_args()

def load_run_data(log_path):
    """Load training data from a log file, skipping bad rows (e.g., repeated headers)."""
    try:
        df = pd.read_csv(log_path)
        # Remove rows where 'episode' is not a number
        df = df[pd.to_numeric(df['episode'], errors='coerce').notnull()].copy()
        df['episode'] = df['episode'].astype(int)
        if 'timestep' in df.columns:
            df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce').fillna(0).astype(int)
        if 'reward' in df.columns:
            df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
        if 'total_reward' in df.columns:
            df['total_reward'] = pd.to_numeric(df['total_reward'], errors='coerce')
        # Also convert all reward component columns to float if present
        for col in df.columns:
            if col not in ['episode', 'timestep', 'reward', 'total_reward']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {log_path}: {e}")
        return None

def plot_reward_components(df, run_name, output_dir='visualizations', smoothing=10):
    """Plot all individual reward components over episodes if present in the DataFrame."""
    reward_cols = [col for col in df.columns if col not in ['episode', 'timestep', 'reward', 'total_reward']]
    if not reward_cols:
        print("No individual reward components found in log file.")
        return
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 8))
    for col in reward_cols:
        smooth = gaussian_filter1d(df[col].values, sigma=smoothing)
        plt.plot(df['episode'], smooth, label=col)
    plt.title(f'Reward Components vs Episodes ({run_name})')
    plt.xlabel('Episode')
    plt.ylabel('Component Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'{run_name}_reward_components.png', dpi=200)
    print(f"Saved reward component plot to {output_path / f'{run_name}_reward_components.png'}")

def create_single_run_visualizations(df, run_name, smoothing=10, output_dir='visualizations'):
    """Create comprehensive visualizations for a single training run."""
    # Make sure output directory exists
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    total_episodes = df['episode'].max()
    total_timesteps = df['timestep'].max()
    mean_reward = df['reward'].mean()
    std_reward = df['reward'].std()
    max_reward = df['reward'].max()
    min_reward = df['reward'].min()
    last_10pct_mean = df['reward'].iloc[-int(len(df)*0.1):].mean()
    
    # Apply smoothing for plotting
    df['reward_smooth'] = gaussian_filter1d(df['reward'].values, sigma=smoothing)
    
    # Create a multi-panel figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. Reward over episodes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['episode'], df['reward'], alpha=0.3, label='Raw')
    ax1.plot(df['episode'], df['reward_smooth'], linewidth=2, label='Smoothed')
    ax1.set_title(f'Reward vs Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward over timesteps
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['timestep'], df['reward'], alpha=0.3, label='Raw')
    ax2.plot(df['timestep'], df['reward_smooth'], linewidth=2, label='Smoothed')
    ax2.set_title(f'Reward vs Timesteps')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward histogram
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(df['reward'], bins=30, kde=True, ax=ax3)
    ax3.axvline(mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    ax3.set_title('Reward Distribution')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward moving average
    ax4 = fig.add_subplot(gs[1, 1])
    window_sizes = [5, 20, 50]
    for window in window_sizes:
        df[f'ma_{window}'] = df['reward'].rolling(window=window).mean()
        ax4.plot(df['episode'], df[f'ma_{window}'], label=f'MA-{window}')
    ax4.set_title('Moving Averages of Reward')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward (Moving Average)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    
    # 6. Stats display
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    stats_text = f"""
    Training Statistics for {run_name}:
    
    Total Episodes: {total_episodes}
    Total Timesteps: {total_timesteps:,}
    
    Reward Statistics:
        Mean: {mean_reward:.2f}
        Std Dev: {std_reward:.2f}
        Min: {min_reward:.2f}
        Max: {max_reward:.2f}
        
    Final Performance:
        Last 10% Mean: {last_10pct_mean:.2f}
        % of Maximum: {(last_10pct_mean/max_reward*100):.1f}%
    
    Training Progress:
        Initial vs Final: {(df['reward_smooth'].iloc[-1]/df['reward_smooth'].iloc[min(20, len(df)-1)])*100-100:.1f}% improvement
    """
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.suptitle(f'Training Analysis - {run_name}', fontsize=20, y=1.02)
    fig.savefig(output_path / f'{run_name}_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {output_path / f'{run_name}_analysis.png'}")
    
    # Save statistics to JSON
    stats = {
        "run_name": run_name,
        "total_episodes": int(total_episodes),
        "total_timesteps": int(total_timesteps),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "min_reward": float(min_reward),
        "max_reward": float(max_reward),
        "last_10pct_mean": float(last_10pct_mean),
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_path / f'{run_name}_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Create additional plots
    
    # Learning curve with confidence intervals
    plt.figure(figsize=(12, 8))
    # Group by episodes for aggregation
    episode_groups = df.groupby(pd.cut(df['episode'], 50))
    means = episode_groups['reward'].mean()
    std = episode_groups['reward'].std()
    x = [(interval.left + interval.right)/2 for interval in means.index]
    
    plt.plot(x, means.values, label='Mean reward')
    plt.fill_between(x, means - std, means + std, alpha=0.3, label='±1 std dev')
    plt.title(f'Learning Curve with Confidence Intervals - {run_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / f'{run_name}_learning_curve.png', dpi=200, bbox_inches='tight')
    
    plot_reward_components(df, run_name, output_dir=output_dir, smoothing=smoothing)
    
    return stats

def compare_runs(run_data, smoothing=10, output_dir='visualizations'):
    """Compare multiple training runs."""
    # Make sure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create plots comparing runs
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Run Comparison', fontsize=20)
    
    # 1. Reward over episodes
    for run_name, df in run_data.items():
        df['reward_smooth'] = gaussian_filter1d(df['reward'].values, sigma=smoothing)
        axes[0, 0].plot(df['episode'], df['reward_smooth'], label=run_name)
    
    axes[0, 0].set_title('Reward vs Episodes')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (Smoothed)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Reward over timesteps
    for run_name, df in run_data.items():
        axes[0, 1].plot(df['timestep'], df['reward_smooth'], label=run_name)
    
    axes[0, 1].set_title('Reward vs Timesteps')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Reward (Smoothed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    run_rewards = [df['reward'] for df in run_data.values()]
    axes[1, 0].boxplot(run_rewards, labels=run_data.keys())
    axes[1, 0].set_title('Reward Distribution Comparison')
    axes[1, 0].set_xlabel('Run')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Final performance comparison
    last_10pct_means = []
    run_names = []
    
    for run_name, df in run_data.items():
        last_10pct_mean = df['reward'].iloc[-int(len(df)*0.1):].mean()
        last_10pct_means.append(last_10pct_mean)
        run_names.append(run_name)
    
    axes[1, 1].bar(run_names, last_10pct_means)
    axes[1, 1].set_title('Final Performance (Last 10% Mean)')
    axes[1, 1].set_xlabel('Run')
    axes[1, 1].set_ylabel('Mean Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'run_comparison.png', dpi=200, bbox_inches='tight')
    print(f"Saved run comparison to {output_path / 'run_comparison.png'}")
    
    # Create a learning curve comparison
    plt.figure(figsize=(14, 10))
    
    for run_name, df in run_data.items():
        # Group by episodes for smoother plot
        episode_groups = df.groupby(pd.cut(df['episode'], 30))
        means = episode_groups['reward'].mean()
        x = [(interval.left + interval.right)/2 for interval in means.index]
        plt.plot(x, means.values, label=f'{run_name} (mean: {df["reward"].mean():.2f})')
    
    plt.title('Learning Curve Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'learning_curve_comparison.png', dpi=200, bbox_inches='tight')
    
    # Create a 3D visualization of reward landscapes
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, (run_name, df) in enumerate(run_data.items()):
            x = df['episode'].values
            y = np.ones_like(x) * i  # Different y-value for each run
            z = df['reward_smooth'].values
            
            ax.plot(x, y, z, label=run_name)
            # Add a filled curtain below for better visibility
            ax.plot_surface(
                x.reshape(-1, 1), 
                np.ones((len(x), 1)) * i, 
                z.reshape(-1, 1),
                alpha=0.3,
                color=plt.cm.tab10(i % 10)
            )
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Run')
        ax.set_zlabel('Reward')
        ax.set_yticks(range(len(run_data)))
        ax.set_yticklabels(run_data.keys())
        ax.set_title('3D Reward Landscape Comparison')
        
        plt.savefig(output_path / 'reward_landscape_3d.png', dpi=200, bbox_inches='tight')
        
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")
    
    return run_names, last_10pct_means

def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all runs
    all_runs = {}
    
    if args.run:
        # Process a specific run
        run_path = log_dir / args.run / 'log.txt'
        if run_path.exists():
            df = load_run_data(run_path)
            if df is not None:
                all_runs[args.run] = df
                stats = create_single_run_visualizations(df, args.run, smoothing=args.smoothing, output_dir=str(output_dir))
                print(f"Processed run: {args.run}")
        else:
            print(f"Run not found: {args.run}")
    else:
        # Process all runs
        run_dirs = [d for d in log_dir.glob('run_*') if d.is_dir()]
        
        for run_dir in run_dirs:
            run_name = run_dir.name
            log_path = run_dir / 'log.txt'
            
            if log_path.exists():
                df = load_run_data(log_path)
                if df is not None:
                    all_runs[run_name] = df
                    stats = create_single_run_visualizations(df, run_name, smoothing=args.smoothing, output_dir=str(output_dir))
                    print(f"Processed run: {run_name}")
    
    # Compare runs if requested
    if args.compare and len(all_runs) > 1:
        run_names, final_rewards = compare_runs(all_runs, smoothing=args.smoothing, output_dir=str(output_dir))
        
        # Identify best run
        best_run_idx = np.argmax(final_rewards)
        best_run = run_names[best_run_idx]
        print(f"\nBest run based on final performance: {best_run} with reward {final_rewards[best_run_idx]:.2f}")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()