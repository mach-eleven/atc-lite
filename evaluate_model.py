"""
Script to evaluate and visualize a trained PPO model on the ATC environment.
"""

import argparse
import os
import time
import numpy as np
import torch
from pathlib import Path

import gymnasium as gym
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
from envs.atc.scenarios import LOWW
from models.PPO import PPO

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model on ATC environment')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--render_mode', type=str, default='human', choices=['human', 'rgb_array'], 
                       help='Rendering mode (human for visualization, rgb_array for recording)')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--save_video', action='store_true', help='Save rendered frames as a video')
    parser.add_argument('--aircraft_count', type=int, default=2, help='Number of aircraft in the environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sim_speed', type=float, default=1.0, 
                       help='Simulation speed multiplier (higher = faster planes)')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Number of simulation steps to take between renders (higher = faster simulation)')
    return parser.parse_args()

def evaluate(args):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize environment with faster timestep
    sim_params = model.SimParameters(
        # Increase the timestep to make planes move faster
        args.sim_speed * 1.0,  # Use positional argument instead of keyword 
        discrete_action_space=False
    )
    
    env = AtcGym(
        airplane_count=args.aircraft_count,
        sim_parameters=sim_params,
        scenario=LOWW(),
        render_mode=args.render_mode
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim, 
        action_dim,
        lr_actor=0.0001,
        lr_critic=0.0001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.1  # Use a small std for evaluation
    )
    
    # Load trained weights
    print(f"Loading model from: {args.model_path}")
    ppo_agent.load(args.model_path)
    
    # Set up video recording if enabled
    frames = []
    
    # For faster execution, disable gradient calculation
    with torch.no_grad():
        # Run evaluation episodes
        rewards = []
        for ep in range(args.num_episodes):
            state, _ = env.reset(seed=args.seed + ep)
            ep_reward = 0
            done = False
            truncated = False
            step = 0
            
            print(f"\nEpisode {ep+1}/{args.num_episodes}")
            
            # Show initial aircraft state
            print("Initial aircraft state:")
            for i, airplane in enumerate(env._airplanes):
                compass_heading = (90 - airplane.phi) % 360
                print(f"Aircraft {i}: Position ({airplane.x:.1f}, {airplane.y:.1f}, {airplane.h:.1f}), "
                    f"Heading {airplane.phi:.1f}° (Compass: {compass_heading:.1f}°), Speed {airplane.v:.1f}")
            
            while not (done or truncated):
                # Select action with the policy network
                action = ppo_agent.select_action(state)
                
                # Use a loop here to skip frames but only accumulate reward once
                for _ in range(args.skip_frames):
                    # Take a step in the environment
                    state, reward, done, truncated, info = env.step(action)
                    
                    # Only accumulate reward and increment step for the first action
                    # to maintain comparable rewards across different skip_frames values
                    if _ == 0:
                        ep_reward += reward
                        step += 1
                    
                    # Break if episode ended
                    if done or truncated:
                        break
                
                # Render environment after skipping frames
                if args.render_mode == 'human':
                    env.render()
                    # No sleep time for maximum speed
                elif args.render_mode == 'rgb_array' and args.save_video:
                    frame = env.render()
                    frames.append(frame)
            
            # Print occasional status updates
            if step % 50 == 0:
                print(f"Step {step}, Current reward: {ep_reward:.2f}")
                
                # Show current aircraft state every 50 steps
                print("Current aircraft state:")
                for i, airplane in enumerate(env._airplanes):
                    status = getattr(airplane, 'status', 'UNKNOWN')
                    compass_heading = (90 - airplane.phi) % 360
                    print(f"Aircraft {i} ({status}): Position ({airplane.x:.1f}, {airplane.y:.1f}, {airplane.h:.1f}), "
                          f"Heading {airplane.phi:.1f}° (Compass: {compass_heading:.1f}°), Speed {airplane.v:.1f}")
            
            # Check for info messages (landing, fuel warnings, etc.)
            if info and 'message' in info and info['message']:
                print(f"Info: {info['message']}")
        
        # Episode summary
        print(f"Episode {ep+1} finished after {step} steps")
        print(f"Total reward: {ep_reward:.2f}")
        rewards.append(ep_reward)
        
        if info:
            print(f"Final status: {info.get('episode_status', 'Unknown')}")
            
            # Display landing stats if available
            if 'landings' in info:
                print("Landing statistics:")
                for aircraft_id, landing_stats in info['landings'].items():
                    print(f"Aircraft {aircraft_id}:")
                    for stat, value in landing_stats.items():
                        print(f"  {stat}: {value}")
    
    # Save video if requested
    if args.save_video and args.render_mode == 'rgb_array' and frames:
        try:
            import cv2
            height, width, _ = frames[0].shape
            video_path = f"eval_video_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for frame in frames:
                # OpenCV uses BGR instead of RGB
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            video.release()
            print(f"Video saved to {video_path}")
        except ImportError:
            print("Could not save video: cv2 module not found")
            np.save(f"eval_frames_{int(time.time())}.npy", np.array(frames))
            print(f"Saved raw frames to eval_frames_{int(time.time())}.npy")
    
    # Final results
    print("\nEvaluation complete")
    print(f"Average reward over {args.num_episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Reward standard deviation: {np.std(rewards):.2f}")
    print(f"Min/Max rewards: {np.min(rewards):.2f}/{np.max(rewards):.2f}")
    
    env.close()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)