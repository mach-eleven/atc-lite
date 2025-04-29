import argparse
from stable_baselines3 import PPO
from envs.atc.atc_gym import AtcGym
from envs.atc.scenarios.mva_goaround import MvaGoAroundScenario
from envs.atc import model
import time

def main(model_path, episodes=5, render=True, sleep=0.05):
    scenario = MvaGoAroundScenario()
    env = AtcGym(
        airplane_count=1,
        sim_parameters=model.SimParameters(1.0, discrete_action_space=False, normalize_state=True),
        scenario=scenario,
        render_mode="human" if render else "headless",
    )
    model_ = PPO.load(model_path, env=env)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model_.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()
                time.sleep(sleep)
        print(f"Episode {ep+1}: Total Reward = {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained PPO model zip file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to replay')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--sleep', type=float, default=0.05, help='Sleep time between frames (for rendering)')
    args = parser.parse_args()
    main(args.model, episodes=args.episodes, render=not args.no_render, sleep=args.sleep)
