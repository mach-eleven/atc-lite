""" FILE IGNORED BY EVERYTHING """

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
        png_path = os.path.join(outdir, 'reward_components.png')
        csv_path = os.path.join(outdir, 'reward_components.csv')
        self.fig.savefig(png_path)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode'] + self.reward_keys)
            for i, ep in enumerate(self.episodes):
                row = [ep] + [self.history[k][i] for k in self.reward_keys]
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to SB3 PPO checkpoint to resume training from')
    args = parser.parse_args()

    outdir = 'sb3_logs_two_planes'
    os.makedirs(outdir, exist_ok=True)

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO

    # Use a scenario with at least two entrypoints (LOWW or custom SimpleScenario)
    scenario = scenarios.SimpleScenario(random_entrypoints=True)
    airplane_count = 2
    sim_params = model.SimParameters(1.0, discrete_action_space=False, normalize_state=True)
    env = AtcGym(
        airplane_count=airplane_count,
        sim_parameters=sim_params,
        scenario=scenario,
        render_mode='headless'
    )
    check_env(env, warn=True)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    reward_keys = [
        'success_rewards', 'airspace_penalties', 'mva_penalties',
        'time_penalty', 'approach_position_rewards', 'approach_angle_rewards',
        'glideslope_rewards', 'fuel_efficiency_rewards', 'fuel_penalties',
        'collision_penalty'
    ]

    model_path = os.path.join(outdir, 'sb3_ppo_model_two_planes.zip')
    log_path = os.path.join('logs/two_planes', 'training_log.csv')

    # Custom reward shaping: add collision penalty
    def custom_reward_shaping(info):
        reward_components = info.get('reward_components', {})
        # Check for collisions (if any two planes are too close)
        collision_penalty = 0
        if hasattr(env, '_airplanes'):
            for i in range(airplane_count):
                for j in range(i+1, airplane_count):
                    dx = env._airplanes[i].x - env._airplanes[j].x
                    dy = env._airplanes[i].y - env._airplanes[j].y
                    dh = env._airplanes[i].h - env._airplanes[j].h
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 2.0 and abs(dh) < 1000:  # 2nm and 1000ft separation
                        collision_penalty -= 1000
        reward_components['collision_penalty'] = collision_penalty
        return reward_components

    # Load or create model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model_ = PPO.load(args.checkpoint, env=vec_env)
    else:
        print("Starting new model from scratch.")
        model_ = PPO('MlpPolicy', vec_env, verbose=1)

    plotter = RewardPlotter(reward_keys)
    n_episodes = 1000
    steps_per_episode = 500
    window_size = 50
    success_threshold = 0.95
    recent_successes = []

    for ep in range(n_episodes):
        obs = env.reset()[0]
        done = False
        total_rewards = {k: 0 for k in reward_keys}
        total_reward = 0
        for step in range(steps_per_episode):
            action, _ = model_.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            # Custom reward shaping
            reward_components = custom_reward_shaping(info)
            for k in reward_keys:
                total_rewards[k] += reward_components.get(k, 0)
            total_reward += reward
            if done:
                break
        model_.learn(total_timesteps=steps_per_episode, reset_num_timesteps=False)
        plotter.update(ep, total_rewards)
        # Success: both planes reached FAF (success_rewards > 0 for both)
        episode_success = total_rewards['success_rewards'] > 1.5  # Each plane gets 1 for success
        recent_successes.append(episode_success)
        if len(recent_successes) > window_size:
            recent_successes.pop(0)
        if ep % 10 == 0:
            print(f"Ep {ep}: {total_rewards}, Total: {total_reward}")
        if ep % 50 == 0 and ep > 0:
            model_.save(model_path.replace('.zip', f'_ep{ep}.zip'))
            with open(log_path, 'a') as f:
                writer = csv.writer(f)
                if ep == 0:
                    writer.writerow(['episode', 'total_reward'] + reward_keys)
                writer.writerow([ep, total_reward] + [total_rewards[k] for k in reward_keys])
        if ep >= window_size and sum(recent_successes) / window_size >= success_threshold:
            print(f"[CURRICULUM] Success threshold reached at episode {ep}, stopping early.")
            break
    model_.save(model_path)
    plotter.save(outdir)
    with open(log_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['final', total_reward] + [total_rewards[k] for k in reward_keys])

if __name__ == '__main__':
    main()
