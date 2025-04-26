import os
import sys
import time
import argparse
from stable_baselines3 import PPO
from envs.atc.atc_gym import AtcGym
from envs.atc import scenarios
import envs.atc.model as model

os.environ['PYGLET_SHADOW_WINDOW'] = '0'
sys.path.append('.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SB3 PPO model zip file to run')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to visualize')
    args = parser.parse_args()

    scenario = scenarios.SimpleScenario(random_entrypoints=True)
    airplane_count = 2
    sim_params = model.SimParameters(1.0, discrete_action_space=False, normalize_state=True)
    env = AtcGym(
        airplane_count=airplane_count,
        sim_parameters=sim_params,
        scenario=scenario,
        render_mode='human'
    )
    model_ = PPO.load(args.model)

    for ep in range(args.episodes):
        obs = env.reset()[0]
        done = False
        print(f"\n=== Visualizing Episode {ep+1} ===")
        while not done:
            action, _ = model_.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            time.sleep(0.05)
    env.close()

if __name__ == "__main__":
    main()
