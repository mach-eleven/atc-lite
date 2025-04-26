import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.atc.atc_gym import AtcGym
from envs.atc import scenarios
import envs.atc.model as model

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'
sys.path.append('.')

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
        csv_path = os.path.join(outdir, 'reward_components.csv')
        self.fig.savefig(png_path)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode'] + self.reward_keys)
            for i, ep in enumerate(self.episodes):
                row = [ep] + [self.history[k][i] for k in self.reward_keys]
                writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to SB3 PPO checkpoint to resume training from')
    args = parser.parse_args()

    outdir = 'sb3_logs'
    os.makedirs(outdir, exist_ok=True)

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO

    # Curriculum entry points: from easy (close) to hard (far), with varied headings and positions
    curriculum_entry_points = [
        ((15, 15), 45),   # Close, NE
        ((14, 14), 90),   # East
        ((13, 13), 135),  # SE
        ((12, 12), 180),  # South
        ((12, 12), 160),  # South
        ((11, 11), 180),  # SW
        ((10, 10), 180),  # West
        ((9, 9), 180),    # NW
        ((8, 8), 180),      # North
        ((7, 7), 180),     # NE, off-axis
        ((6, 6), 180),    # SE, off-axis
        ((5, 5), 180),    # NW, off-axis
        ((0, 0), 180),     # Farthest corner, NE
        ((0, 0), 160),     # Farthest corner, NE
    ]

    reward_keys = [
        'success_rewards', 'airspace_penalties', 'mva_penalties',
        'time_penalty', 'approach_position_rewards', 'approach_angle_rewards',
        'glideslope_rewards', 'fuel_efficiency_rewards', 'fuel_penalties'
    ]

    n_episodes_per_stage = 500
    steps_per_episode = 500
    log_path = os.path.join(outdir, 'training_log.csv')

    success_threshold = 0.99  # 99% success required to move to next stage
    window_size = 50         # Check over last 50 episodes
    max_episodes_per_stage = 5000  # Safety cap to avoid infinite loops

    for stage, (entry_xy, entry_heading) in enumerate(curriculum_entry_points):
        print(f"\n=== Curriculum Stage {stage+1}: Entry {entry_xy}, Heading {entry_heading} ===")
        stage_name = f"stage{stage+1}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}"
        stage_dir = os.path.join(outdir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        env = AtcGym(
            airplane_count=1,
            sim_parameters=model.SimParameters(1.0, discrete_action_space=False, normalize_state=True),
            scenario=scenarios.SupaSupa(),
            render_mode='headless'
        )
        check_env(env, warn=True)
        vec_env = make_vec_env(lambda: env, n_envs=1)

        model_path = os.path.join(stage_dir, f'sb3_ppo_model_{stage_name}.zip')
        # Load from checkpoint if provided, else create new model
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model_ = PPO.load(args.checkpoint, env=vec_env)
        elif stage > 0 and os.path.exists(prev_model_path):
            print(f"Loading model from previous curriculum stage: {prev_model_path}")
            model_ = PPO.load(prev_model_path, env=vec_env)
        else:
            print("Starting new model from scratch.")
            model_ = PPO('MlpPolicy', vec_env, verbose=1)

        plotter = RewardPlotter(reward_keys)
        recent_successes = []
        ep = 0
        while True:
            obs = env.reset()[0]
            done = False
            total_rewards = {k: 0 for k in reward_keys}
            total_reward = 0
            for step in range(steps_per_episode):
                action, _ = model_.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if 'reward_components' in info:
                    for k in reward_keys:
                        total_rewards[k] += info['reward_components'].get(k, 0)
                if done:
                    break
            model_.learn(total_timesteps=steps_per_episode, reset_num_timesteps=False)
            plotter.update(ep, total_rewards)
            episode_success = total_rewards['success_rewards'] > 0
            recent_successes.append(episode_success)
            if len(recent_successes) > window_size:
                recent_successes.pop(0)
            if ep % 10 == 0:
                print(f"{stage_name} Ep {ep}: {total_rewards}, Total: {total_reward}")
            if ep % 50 == 0 and ep > 0:
                model_.save(os.path.join(stage_dir, f'sb3_ppo_model_{stage_name}_ep{ep}.zip'))
                with open(os.path.join(stage_dir, f'training_log_{stage_name}.csv'), 'a') as f:
                    writer = csv.writer(f)
                    if ep == 0:
                        writer.writerow(['episode', 'total_reward'] + reward_keys)
                    writer.writerow([ep, total_reward] + [total_rewards[k] for k in reward_keys])
                print(f"[LOG] {stage_name} Ep {ep}: Total Reward: {total_reward} | " + ", ".join([f"{k}: {total_rewards[k]}" for k in reward_keys]))
            # Check if we can promote to next stage
            if ep >= window_size and sum(recent_successes) / window_size >= success_threshold:
                print(f"[CURRICULUM] Success threshold reached at episode {ep}, moving to next stage.")
                break
            if ep >= max_episodes_per_stage:
                print(f"[CURRICULUM] Max episodes ({max_episodes_per_stage}) reached for {stage_name}, moving to next stage.")
                break
            ep += 1
        # Save model and log at end of stage
        model_.save(model_path)
        plotter.save(stage_dir)
        with open(os.path.join(stage_dir, f'training_log_{stage_name}.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['final', total_reward] + [total_rewards[k] for k in reward_keys])
        prev_model_path = model_path
