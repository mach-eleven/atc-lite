import csv
from logging import Logger
import os
from pathlib import Path
import time
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from torch.utils.tensorboard import SummaryWriter

def set_log_paths(args, reward_keys):
    if args.log_csv:
        eval_log_path_csv = Path(args.outdir) / "evaluation_log.csv"
    else:
        eval_log_path_csv = None

    if args.log_file:
        flog_path = Path(args.outdir) / "log.txt"
    else:
        flog_path = None

    if args.log_tensorboard:
        tensorboard_logd = Path(args.outdir) / "tensorboard"
        tensorboard_logd.mkdir(parents=True, exist_ok=True)
        # Create TensorBoard logger
        tb_logger = SummaryWriter(log_dir=str(tensorboard_logd))
    else:
        tensorboard_logd = None
        tb_logger = None

    if args.live_plot:
        plotter = RewardPlotter(reward_keys)
    else:
        plotter = None
    
    return eval_log_path_csv, flog_path, tensorboard_logd, tb_logger, plotter

def log_info(model_name, args, logger):
    logger.info(f"=" * 80)
    logger.info(f"Training Information for {model_name}:")
    logger.info(f"Logging -> CSV: {args.log_csv}, TensorBoard: {args.log_tensorboard}, File: {args.log_file}")
    logger.info(f"Output directory: '{args.outdir}'")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Max episodes: {args.max_episodes}")
    logger.info(f"Max steps per episode: {args.max_steps_per_episode}")
    logger.info(f"Save frequency: {args.save_freq}")
    logger.info(f"Evaluation frequency: {args.eval_freq}. Evaluating for {args.eval_episodes} episodes.")
    logger.info(f"Threads: {args.threads}")
    logger.info(f"Live plotting: {args.live_plot}")
    logger.info(f"=" * 80)

def log_model_stuff(log_tensorboard, log_csv, log_file, ep, total_reward, total_rewards, success_rate, log_path, flog_path, reward_keys, tblogger = None):
    if log_csv:
        file_exists = log_path.exists()
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode', 'total_reward', 'success_rate'] + list(reward_keys))
            writer.writerow([ep, total_reward, success_rate] + [total_rewards[k] for k in reward_keys])
    if log_tensorboard:
        # Log to TensorBoard
        # Assuming you have a TensorBoard logger set up
        tblogger.add_scalar('Total Reward', total_reward, ep)
        tblogger.add_scalar('Success Rate', success_rate, ep)
        for k in reward_keys:
            tblogger.add_scalar(f'Reward/{k}', total_rewards[k], ep)
        pass

    if log_file:
        with open(flog_path, 'a') as f:
            f.write(f"Episode {ep}: Total Reward = {total_reward}, Success Rate = {success_rate}, Rewards = {total_rewards}\n")

class DebugLogCallback(BaseCallback):
    """
    Custom callback to log progress every 100 timesteps.
    """
    
    def __init__(self, logger, log_freq, verbose=0):
        super().__init__(verbose)
        self._logger = logger  # Use underscore to avoid property conflicts
        self._log_freq = log_freq
        self.timesteps = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        self.timesteps += self.training_env.num_envs
        if self.timesteps % self._log_freq == 0:
            elapsed_time = time.time() - self.start_time
            self.locals['remaining_time'] = (self.locals['total_timesteps'] - self.timesteps) * (elapsed_time / self.timesteps)

            self._logger.debug(f"Progress: {self.timesteps}/{self.locals['total_timesteps']} timesteps completed. -{self.locals['remaining_time']:.2f} seconds.")
        return True

    def _on_training_end(self) -> None:
        self._logger.debug(f"Finished training {self.timesteps} timesteps")

def human_readable_time(seconds):
    """
    Convert seconds to a human-readable format
    
    Xh Ym Zs
    """
    if seconds < 0:
        return "Negative time"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    # return exactly 00:00:00.MS
    return f"{hours:02}:{minutes:02}:{seconds:02}.{int((seconds - int(seconds)) * 1000):03}"


class RewardPlotter:
    def __init__(self, reward_keys):
        self.reward_keys = reward_keys
        self.history = {k: [] for k in reward_keys}
        self.episodes = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.lines = {k: self.ax.plot([], [], label=k)[0] for k in reward_keys}
        self.ax.legend()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward Component')
        self.fig.canvas.manager.set_window_title('Live Reward Shaping Visualization')

    def update(self, episode, reward_components):
        self.episodes.append(episode)
        for k in self.reward_keys:
            self.history[k].append(reward_components.get(k, 0))
        for k in self.reward_keys:
            self.lines[k].set_data(self.episodes, self.history[k])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, outdir):
        # Save reward curves as PNG and CSV
        png_path = os.path.join(outdir, 'reward_components.png')
        self.fig.savefig(png_path)

    def close(self):
        plt.ioff()
        plt.close(self.fig)