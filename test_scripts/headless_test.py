# headless_test.py
import sys
import time
import os
import argparse

from envs.atc.scenarios import LOWW

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'

# Add the parent directory to sys.path
sys.path.append('.')

# Import dependencies
import gymnasium as gym
import numpy as np
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
import math
import envs  # Required to register the environment

def parse_args():
    parser = argparse.ArgumentParser(description='Test the ATC environment in headless or visual mode')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no visualization)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=600, help='Maximum steps per episode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine render mode based on arguments
    render_mode = 'headless' if args.headless else 'human'
    
    print(f"Running ATC simulation in {render_mode} mode for {args.episodes} episodes...")
    
    # Create environment with continuous action space
    sim_params = model.SimParameters(2.0, discrete_action_space=False)
    
    # Create environment with the specified render mode
    env = AtcGym(airplane_count=3, sim_parameters=sim_params, 
                 scenario=LOWW(random_entrypoints=True), 
                 render_mode=render_mode)

    # Reset the environment
    state, info = env.reset()
    
    # Run specified number of episodes
    for episode in range(args.episodes):
        print(f"\n*** EPISODE {episode+1}/{args.episodes} ***")
        state, info = env.reset()

        # Get number of airplanes
        airplane_count = len(env._airplanes)
        
        total_reward = 0
        for step in range(args.steps):
            # Random actions for testing
            action = np.random.uniform(-1, 1, size=3*airplane_count)

            # Step the environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Only render in human mode, and only every few steps to speed up simulation
            if not args.headless and step % 5 == 0:
                env.render()
                time.sleep(0.05)  # Small delay for visualization
            
            # Print status periodically
            if step % 30 == 0 or step < 5:
                print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                for i, airplane in enumerate(env._airplanes):
                    print(f"{airplane.name}: Pos({airplane.x:.1f}, {airplane.y:.1f}) Alt:{airplane.h:.0f}ft "
                          f"Speed:{airplane.v:.1f}kts Fuel:{airplane.fuel_remaining_pct:.1f}%")
            
            if done:
                print(f"Episode {episode+1} ended after {step} steps with reward: {reward:.2f}")
                break
        
        # Small delay between episodes
        time.sleep(1.0)
    
    env.close()
    print("Testing complete")

if __name__ == "__main__":
    main()