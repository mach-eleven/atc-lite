# ATC-lite

Gymnasium environment and Training code for the "Air Traffic Control (ATC) problem".

## Overview

The problem is defined as follows:

> The agent must learn to navigate the aircraft in a given airspace, avoiding collisions and ensuring safe distances between aircraft. More than an ATC planning problem, it is a navigation problem, where the agent must learn to navigate the aircraft in a given airspace, avoiding collisions and ensuring safe distances between aircraft.

The problem has been trained for 2 different scenarios:

* A real world scenario, where the agent must learn to navigate a singular aircraft in the airspace of an actual airport. We have selected New Orleans Lakefront Airport (KNEW) as our real world scenario. The agent must learn to navigate the aircraft in the airspace of the airport, avoiding collisions and ensuring safe distances between aircraft. Using ADS-B data, we have created a simulation of an actual aircraft's trajectory, which is used to compare the agent's performance against the actual aircraft's trajectory.
* A simulated scenario, where the agent must learn how to work with 2 aircraft at the same time. The agent must learn to land them after one another.

We have used the Proximal Policy Optimization (PPO) algorithm to train the agent, along with Curriculum Learning. The agent is trained using a reward function that is based on the distance between the aircraft and the runway, as well as the distance between the aircraft and other aircraft in the airspace. The agent is trained to maximize the reward function, while minimizing the distance to the runway and avoiding collisions with other aircraft.

## Submission

* Video is present at this link: [ATC-lite Video](https://youtu.be/J-xRird-wb8)
* PDF report is present in the repository with the file name `report.pdf`.
* The code is present in the repository, and the environment is implemented in the `envs` folder. The training code is present in the `train.py` file. The code is compatible with Python 3.12 and above.
* You can use `replay.py` to visualize the trained agent's performance in the environment.
* Best models are present in `final_models` folder.

## Training

`train.py` is used to train the reinforcement learning agent.

**Usage:**

```bash
python train.py [OPTIONS]
```

**Key Options:**

* `--model`: Select the RL algorithm (`ppo_sb3`, `curr`, `ppo`, `dqn`). Default: `ppo_sb3`.
* `--checkpoint`: Path to a checkpoint file to resume training.
* `--outdir`: Directory to save logs and models. Default: `new_logs/ppo_sb3`.
* `--max-episodes`: Number of training episodes. Default: 500.
* `--max-steps-per-episode`: Maximum steps per episode. Default: 50,000.
* `--save-freq`: Save model checkpoint frequency (in episodes). Default: 20.
* `--eval-freq`: Evaluation frequency (in episodes). Default: 2.
* `--eval-episodes`: Number of episodes for evaluation. Default: 5.
* `--log-where`: Logging destinations (`csv`, `tensorboard`, `file`). Default: `csv, tensorboard`.
* `--threads`: Number of parallel threads for training. Default: 8.
* `--debug`: Enable detailed logging.
* `--live-plot`: Enable live plotting of evaluation progress.
* `--curr-*`: Options specific to curriculum learning (`--curr-window-size`, `--curr-success-threshold`, `--curr-stages`).

## Replay

`replay.py` is used to visualize a trained agent's performance in the environment.

**Usage:**

```bash
python replay.py --checkpoint <path_to_checkpoint> [OPTIONS]
```

**Key Options:**

* `--model`: Select the RL algorithm used for the checkpoint (`ppo_sb3`, `ppo`, `dqn`). Default: `ppo_sb3`.
* `--checkpoint`: **Required.** Path to the trained model checkpoint file.
* `--curr-stage-entry-point`: Curriculum stage entry point to use (1 to N, or "max"). Default: "max".
* `--curr-stages`: Total number of curriculum stages defined during training. Default: 15.
* `--entry`: Override entry point coordinates (e.g., `5,5`). Requires `--heading` and `--level`.
* `--heading`: Override entry heading.
* `--level`: Override entry flight level.
* `--skip-frames`: Render every Nth frame. Default: 100.
* `--debug`: Enable detailed logging.

## Test Env

`test_env.py` runs a simulation with specific scenarios, often for testing environment behavior or specific flight patterns, potentially with modified physics like wind or fuel consumption.

**Usage:**

```bash
python test_env.py [OPTIONS]
```

**Key Options:**

* `--headless`: Run without visualization.
* `--episodes`: Number of episodes to run. Default: 5.
* `--steps`: Maximum steps per episode. Default: 600.
* `--render-interval`: Render every N steps (non-headless mode). Default: 1.
* `--wind-scale`: Adjust wind strength. Default: 2.0.
* `--autopilot`/`--no-autopilot`: Enable/disable autopilot heading correction. Default: Enabled.
* `--curr-stages`: Number of curriculum stages available. Default: 15.
* `--curr-stage-entry-point`: Select curriculum stage entry point. Default: 15.
* `--pause-frame`: Pause simulation indefinitely after the first frame.
