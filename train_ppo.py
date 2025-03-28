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

from envs.atc.scenarios import LOWW

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
    Train the PPO agent on the specified environment."
    """

    print("Training PPO agent on environment:", name)
    print("Continuous action space:", is_continuous_action_space)

    sim_params = model.SimParameters(2.0, discrete_action_space=False)
    # Create environment with 3 aircraft for different scenarios

    env = AtcGym(
        airplane_count=3,
        sim_parameters=sim_params,
        scenario=LOWW(),
    )
    env.reset()

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
        env.seed(ppo_params["seed"])

    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor=ppo_params["lr"],
        lr_critic=ppo_params["lr"],
        gamma=ppo_params["gamma"],
        K_epochs=ppo_params["K_epochs"],
        eps_clip=ppo_params["eps_clip"],
        has_continuous_action_space=is_continuous_action_space,
        action_std_init=action_std,
        # action_std_decay_rate=action_std_decay_rate,
        # min_action_std=min_action_std,
        # action_std_decay_freq=action_std_decay_freq,
    )

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at : ", start_time)

    log_f = open(run_dir / "log.txt", "w")
    log_f.write('episode,timestep,reward\n')
    log_f.flush()
    
    print_running_reward = 0
    print_running_episodes = 0

    save_running_reward = 0
    save_running_episodes = 0

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0
        done = False
        
        for t in range(1, max_ep_len+1):
            if isinstance(state, tuple):
                state = state[0]
            action = ppo_agent.select_action(state)

            # Perform action in the environment
            state, reward, done, truncated, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update the PPO agent
            if time_step % ppo_params["update_timestep"] == 0:
                ppo_agent.update()

            if is_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_freq == 0:
                log_avg_reward = save_running_reward / save_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                save_running_reward = 0
                save_running_episodes = 0
            
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                checkpoint_path = checkpoint_directory / f"ppo_{name}_{ppo_params['seed']}_{time_step}.pth"
                print("--------------------------------------------------------------------------------------------")
                print(f"saving model at : {checkpoint_path}")
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                print("--------------------------------------------------------------------------------------------")
                ppo_agent.save(str(checkpoint_path))
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done or truncated:
                break

        # update running reward
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        save_running_reward += current_ep_reward
        save_running_episodes += 1
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
        max_ep_len=1000,
        max_training_timesteps=300_000,
        action_std=0.6,
        action_std_decay_rate=0.05,  # linear decay implemented currently
        min_action_std=0.1,
        action_std_decay_freq=250_000,  # decay every 250k steps
        print_freq=5_000,
        log_freq=2_000,
        save_model_freq=10_000,
        save_model=True,
        ppo_params={
            "K_epochs": 80,
            "update_timestep": 2_000,  # update policy every 2000 steps
            "eps_clip": 0.2,  # clip parameter for PPO
            "gamma": 0.99,  # discount factor
            "lr": 0.0003,  # learning rate
            "seed": 42,  # random seed
        },
        log_dir=Path("logs/ppo"),
    )
