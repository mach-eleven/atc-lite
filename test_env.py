# minimal-atc-rl/test_env.py
import sys
import time
import os

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'

# Add the parent directory to sys.path
sys.path.append('.')

# Import dependencies
import gymnasium as gym
import numpy as np
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
import envs  # Required to register the environment

def main():
    print("Creating ATC environment...")
    
    # Create environment with continuous action space and slower simulation speed
    sim_params = model.SimParameters(0.5, discrete_action_space=False)
    env = AtcGym(airplane_count=8, sim_parameters=sim_params)
    
    # Reset the environment
    state, info = env.reset()
    
    # Run a few episodes
    num_episodes = 3
    max_steps_per_episode = 200  # Increased to allow more time to see the visualization
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        state, info = env.reset()

        airplane_count = len(env._airplanes)
        
        total_reward = 0
        for step in range(max_steps_per_episode):
            # Instead of purely random actions, let's use actions that tend to keep the plane 
            # moving toward the center of the airspace
            action = np.array([0.0, 0.0, 0.0]*airplane_count)  # Default action - maintain course
            
            # Add small random perturbations
            action += np.random.uniform(-0.2, 0.2, size=action.shape)
            
            # Step the environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the environment with the correct mode
            env.render(mode='human')
            time.sleep(0.1)  # Longer pause to better see the visualization
            
            if step % 10 == 0:
                print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                for i, airplane in enumerate(env._airplanes):
                    print(f"Aircraft {i}: Alt: {airplane.h:.0f}, Heading: {airplane.phi:.0f}°, Speed: {airplane.v:.1f}, Position: ({airplane.x:.1f}, {airplane.y:.1f})")
            
            if done:
                if reward > 1000:
                    print("Episode won!")
                else:
                    print(f"Episode ended with reward: {reward:.2f}")
                break
        
        # Small delay between episodes
        time.sleep(1.0)
    
    env.close()
    print("Testing complete")

if __name__ == "__main__":
    main()