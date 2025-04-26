"""
Runs the PPO algorithm on the code with checkpoint saving.
"""

from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import sys
import time
import os

from envs.atc import scenarios

# Set Pyglet configurations for macOS
os.environ["PYGLET_SHADOW_WINDOW"] = "0"

# Add the parent directory to sys.path
sys.path.append(".")
from rich import print

import gymnasium as gym
import numpy as np
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
import math
import envs

import torch
from torch import multiprocessing

from collections import defaultdict
from tqdm import tqdm

from models.PPO import PPO


def train(
    name,
    is_continuous_action_space,
    log_dir,
    max_ep_len,
    max_training_timesteps,
    action_std,
    action_std_decay_rate,
    min_action_std,
    action_std_decay_freq,
    print_freq,
    log_freq,
    save_model_freq,
    save_model,
    ppo_params,
):
    """
    Train the PPO agent on the specified environment.
    """

    print("Training PPO agent on environment:", name)
    print("Continuous action space:", is_continuous_action_space)

    # Adjust initial action standard deviation to be more conservative
    action_std = 0.3  # Start with smaller actions to keep aircraft in bounds

<<<<<<< HEAD
    # TODO: DOESNT LISTEN TO HEADLESS

=======
    # Use a longer max episode length to give more time for learning
    max_ep_len = 2000

    # Create environment with proper parameters
    sim_params = model.SimParameters(
        1.0,  # Use a smaller timestep for more gradual state changes
        discrete_action_space=not is_continuous_action_space
    )
    
    # Use the simple scenario for training
    env = AtcGym(
        airplane_count=1,  # Single aircraft for easier learning
        sim_parameters=sim_params,
        scenario=scenarios.SimpleTrainingScenario(),
    )

    # Reset environment to get initial state
    state, _ = env.reset()  # Handle the newer Gym API that returns (state, info)

    state_dim = env.observation_space.shape[0]
    if is_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    print("State dimension:", state_dim)
    print("Action dimension:", action_dim)

    log_dir = log_dir / name
    log_dir.mkdir(parents=True, exist_ok=True)

    run_number = len(list(log_dir.glob("run_*")))
    run_number = run_number + 1
    run_dir = log_dir / f"run_{run_number}"

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Run directory:", run_dir)

    checkpoint_directory = run_dir / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    ############# print all hyperparameters #############
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print(
        "printing average reward over episodes in last : "
        + str(print_freq)
        + " timesteps"
    )
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print(
        "--------------------------------------------------------------------------------------------"
    )

    if is_continuous_action_space:
        print("Initializing a continuous action space policy")
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print(
            "decay frequency of std of action distribution : "
            + str(action_std_decay_freq)
            + " timesteps"
        )
    else:
        print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("PPO update frequency : " + str(ppo_params["update_timestep"]) + " timesteps")
    print("PPO K epochs : ", ppo_params["K_epochs"])
    print("PPO epsilon clip : ", ppo_params["eps_clip"])
    print("discount factor (gamma) : ", ppo_params["gamma"])
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("optimizer learning rate : ", ppo_params["lr"])
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("random seed : ", ppo_params["seed"])
    print(
        "--------------------------------------------------------------------------------------------"
    )

    if ppo_params["seed"]:
        torch.manual_seed(ppo_params["seed"])
        np.random.seed(ppo_params["seed"])
        # Use the correct method for setting seed
        env.reset(seed=ppo_params["seed"])

    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor=0.0003,  # Increased learning rate
        lr_critic=0.0003,  # Increased learning rate
        gamma=ppo_params["gamma"],
        K_epochs=120,  # More PPO epochs
        eps_clip=ppo_params["eps_clip"],
        has_continuous_action_space=is_continuous_action_space,
        action_std_init=0.2,  # Lower action std for more precise control
        use_gae=True,  # Enable GAE
        gae_lambda=0.97,  # Slightly higher lambda for bias-variance tradeoff
        obs_norm=True,  # Enable observation normalization
        entropy_coef=0.02,  # Higher entropy for more exploration
    )

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at : ", start_time)

    log_f = open(run_dir / "log.txt", "w")
    header_written = False
    
    print_running_reward = 0
    print_running_episodes = 0

    save_running_reward = 0
    save_running_episodes = 0

    time_step = 0
    i_episode = 0

    # Add a debug flag to monitor aircraft positions
    debug_mode = True

    # Prepare to accumulate reward components for each episode
    episode_reward_components = None
    reward_component_keys = None

    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0
        done = False
        truncated = False
        # Reset reward components accumulator
        episode_reward_components = None
        
        # Debug initial state
        if debug_mode and i_episode < 3:
            print(f"\nEpisode {i_episode} - Initial state:")
            # Use _airplanes instead of airplanes and correct attribute names
            for i, airplane in enumerate(env._airplanes):
                print(f"Aircraft {i}: Position ({airplane.x:.1f}, {airplane.y:.1f}, {airplane.h:.1f}), "
                      f"Heading {airplane.phi:.1f}°, Speed {airplane.v:.1f}")
        
        for t in range(1, max_ep_len+1):
            # Mask actions for per-aircraft done logic
            if isinstance(state, tuple):
                state = state[0]
            action = ppo_agent.select_action(state)

            # Track which aircraft are still active (not done)
            if t == 1:
                airplane_count = len(env._airplanes)
                active_aircraft = [True] * airplane_count

            # Mask actions for aircraft that are done (set to zeros)
            action = action.copy() if isinstance(action, np.ndarray) else np.array(action)
            for i in range(airplane_count):
                # Heuristic: If aircraft is in approach corridor or out of fuel, mask its action
                airplane = env._airplanes[i]
                # Check if aircraft is in approach corridor (landed)
                in_corridor = env._runway.inside_corridor(airplane.x, airplane.y, airplane.h, airplane.phi)
                # Check if aircraft is out of fuel
                out_of_fuel = getattr(airplane, 'fuel_remaining_pct', 100) <= 0.1
                if in_corridor or out_of_fuel:
                    active_aircraft[i] = False
                    action[i*3:(i+1)*3] = 0.0

            # Perform action in the environment
            state, reward, done, truncated, info = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # Accumulate reward components
            if 'reward_components' in info:
                if episode_reward_components is None:
                    episode_reward_components = {k: 0.0 for k in info['reward_components']}
                    reward_component_keys = list(episode_reward_components.keys())
                for k, v in info['reward_components'].items():
                    episode_reward_components[k] += v

            # update the PPO agent
            if time_step % ppo_params["update_timestep"] == 0:
                ppo_agent.update()

            if is_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_freq == 0:
                # Avoid division by zero
                if save_running_episodes > 0:
                    log_avg_reward = save_running_reward / save_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                save_running_reward = 0
                save_running_episodes = 0
            
            if time_step % print_freq == 0:
                # Avoid division by zero
                if print_running_episodes > 0:
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0 and save_model:
                checkpoint_path = checkpoint_directory / f"ppo_{name}_{ppo_params['seed']}_{time_step}.pth"
                print("--------------------------------------------------------------------------------------------")
                print(f"saving model at : {checkpoint_path}")
                
                # Only print average reward if we have episodes to average
                if print_running_episodes > 0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                else:
                    print("Episode : {} \t\t Timestep : {}".format(i_episode, time_step))
                    
                print("--------------------------------------------------------------------------------------------")
                ppo_agent.save(str(checkpoint_path))
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # Debug after step if aircraft left airspace
            if debug_mode and info and "message" in info and "left the airspace" in info.get("message", ""):
                # Use _airplanes instead of airplanes and correct attribute names
                for i, airplane in enumerate(env._airplanes):
                    if hasattr(airplane, 'status') and airplane.status == model.AircraftStatus.OUT_OF_BOUNDS:
                        print(f"Aircraft {i} left airspace: Position ({airplane.x:.1f}, {airplane.y:.1f}, {airplane.h:.1f}), "
                              f"Heading {airplane.phi:.1f}°, Speed {airplane.v:.1f}")
                        if t < 10:  # Only for early departures
                            print(f"Action taken: {action}")
                            print(f"Initial position was approximately: {getattr(env, 'initial_positions', {}).get(i, 'Unknown')}")

            if done or truncated:
                break

        # update running reward
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        save_running_reward += current_ep_reward
        save_running_episodes += 1

        # Write episode reward components to log ONLY if header and keys are available
        if not header_written and reward_component_keys is not None:
            log_f.write('episode,timestep,reward,' + ','.join(reward_component_keys) + '\n')
            header_written = True
        if header_written and reward_component_keys is not None:
            row = [str(int(i_episode)), str(int(time_step)), f"{current_ep_reward:.4f}"]
            if episode_reward_components is not None:
                row += [f"{episode_reward_components[k]:.4f}" for k in reward_component_keys]
            else:
                row += ["0.0000" for _ in reward_component_keys]
            log_f.write(",".join(row) + "\n")
            log_f.flush()

        i_episode += 1
    
    # close the environment
    env.close()
    log_f.close()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at : ", start_time)
    print("Finished training at : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    train(
        name="AtcGym-v0",
        is_continuous_action_space=True,
        max_ep_len=2000,  # Longer episodes
        max_training_timesteps=500_000,  # More training time
        action_std=0.3,  # Reduced initial action variance
        action_std_decay_rate=0.03,  # Slower decay
        min_action_std=0.1,
        action_std_decay_freq=250_000,
        print_freq=5_000,
        log_freq=2_000,
        save_model_freq=10_000,
        save_model=True,
        ppo_params={
            "K_epochs": 80,
            "update_timestep": 4_000,  # Update less frequently for more stable learning
            "eps_clip": 0.2,
            "gamma": 0.99,
            "lr": 0.0001,  # Lower learning rate for more stable learning
            "seed": 42,
        },
        log_dir=Path("logs/ppo"),
    )
