"""
NOT YET IMPLEMENTED
"""


def train_dqn(args, reward_keys):

    raise NotImplementedError("DQN training is not implemented yet.")

def train(
    name,
    log_dir,
    max_ep_len,
    max_training_timesteps,
    print_freq,
    log_freq,
    save_model_freq,
    save_model,
    dqn_params,
):
    from datetime import datetime
    from pathlib import Path
    import os
    import sys
    import numpy as np
    import torch
    import random
    import warnings

    warnings.filterwarnings("ignore")

    sys.path.append(".")

    from envs.atc.scenarios import LOWW
    from envs.atc.atc_gym import AtcGym
    import envs.atc.model as model
    from models.DQN import DQNAgen
    print("Training DQN agent on environment:", name)

    sim_params = model.SimParameters(2.0, discrete_action_space=True)
    env = AtcGym(
        airplane_count=1,
        sim_parameters=sim_params,
        scenario=LOWW(),
        render_mode="headless"
    )

    state_dim = env.observation_space.shape[0]
    action_shape = env.action_space.nvec
    action_dim = int(np.prod(action_shape))

    log_dir = log_dir / name
    log_dir.mkdir(parents=True, exist_ok=True)
    run_number = len(list(log_dir.glob("run_*"))) + 1
    run_dir = log_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = run_dir / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "log.txt", "w")
    log_f.write("episode,timestep,reward\n")
    log_f.flush()

    agent = DQNAgent(
        state_dim=state_dim,
        action_shape=action_shape,
        buffer_size=dqn_params["buffer_size"],
        batch_size=dqn_params["batch_size"],
        gamma=dqn_params["gamma"],
        lr=dqn_params["lr"],
        epsilon_start=dqn_params["epsilon_start"],
        epsilon_end=dqn_params["epsilon_end"],
        epsilon_decay=dqn_params["epsilon_decay"],
        target_update_freq=dqn_params["target_update_freq"]
    )

    print("State dimension:", state_dim)
    print("Action shape:", action_shape, ", Flattened dim:", action_dim)

    timestep = 0
    episode = 0
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at:", start_time)

    print_running_reward = 0
    print_running_episodes = 0

    while timestep <= max_training_timesteps:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        ep_reward = 0

        for t in range(max_ep_len):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % log_freq == 0:
                avg_reward = print_running_reward / max(print_running_episodes, 1)
                log_f.write(f"{episode},{timestep},{avg_reward:.2f}\n")
                log_f.flush()
                print_running_reward = 0
                print_running_episodes = 0

            if timestep % print_freq == 0:
                print(f"Episode: {episode}, Timestep: {timestep}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

            if timestep % save_model_freq == 0 and save_model:
                checkpoint_path = checkpoint_directory / f"dqn_{name}_{timestep}.pth"
                agent.save(str(checkpoint_path))
                print("Model saved at", checkpoint_path)

            if done:
                break

        print_running_reward += ep_reward
        print_running_episodes += 1
        episode += 1

    env.close()
    log_f.close()
    print("Training complete. Total time:", datetime.now().replace(microsecond=0) - start_time)


if __name__ == "__main__":
    train(
        name="AtcGym-v0",
        max_ep_len=1000,
        max_training_timesteps=300_000,
        print_freq=5000,
        log_freq=2000,
        save_model_freq=10000,
        save_model=True,
        dqn_params={
            "buffer_size": 100000,
            "batch_size": 64,
            "gamma": 0.99,
            "lr": 1e-3,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "target_update_freq": 1000,
        },
        log_dir=Path("logs/dqn")
    )