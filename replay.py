"""
Train the RL agent using the selected algorithm.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

from stable_baselines3 import PPO
import torch

from envs.atc import model, scenarios
from envs.atc.atc_gym import AtcGym
from utils.type_checkers import (
    curr_stage_type,
    entry_point_type,
    gt_0,
    checkpoint_type,
    gt_0_float,
    outdir_type,
    log_list_type,
    parse_log_config,
)
import logging
from rich.logging import RichHandler

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)


def validate_and_get_entry_point(entry, heading, level, curr_stage_entry_point):
    if entry is not None and heading is not None and level is not None:
        entry_point = model.EntryPoint(entry[0], entry[1], heading, level)
    elif entry is not None:
        raise ValueError(
            "If entry is provided, heading and level must also be provided."
        )
    elif heading is not None or level is not None:
        raise ValueError(
            "If heading or level is provided, entry must also be provided."
        )
    elif curr_stage_entry_point == "max":
        entry_point = scenarios.SupaSupa().entrypoints[0]
    else:
        if curr_stage_entry_point < 1 or curr_stage_entry_point > args.curr_stages:
            raise ValueError(
                f"Curriculum stage {curr_stage_entry_point} is out of range. Must be between 1 and {args.curr_stages}."
            )
        entry_point = scenarios.SupaSupa().generate_curriculum_entrypoints(
            num_entrypoints=args.curr_stages
        )[curr_stage_entry_point - 1]

    return entry_point


def add_arguments(parser):
    """
    Add command line arguments for training.
    """
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_sb3",
        choices=["ppo_sb3", "ppo", "dqn"],
        help="RL algorithm to replay",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to run"
    )
    parser.add_argument(
        "--curr-stage-entry-point",
        type=curr_stage_type,
        default="max",
        help="Curriculum stage to use (1 = closest, ... N = farthest). If none, it will use default entry point.",
    )
    parser.add_argument(
        "--curr-stages",
        type=gt_0,
        default=100,
        help="Number of entry points to generate for curriculum training",
    )
    parser.add_argument(
        "--entry",
        type=entry_point_type,
        default=None,
        help="Override entry point as x,y (e.g. 5,5)",
    )
    parser.add_argument("--heading", type=gt_0, default=None, help="Override heading.")
    parser.add_argument(
        "--level", type=gt_0, default=None, help="Override flight level."
    )
    parser.add_argument(
        "--skip-frames",
        type=gt_0,
        default=100,
        help="Render every Nth frame (default: 1, i.e., no skipping)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, i.e. stdout logging.",
        default=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode ENABLED.")
    else:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode DISABLED.")

    match args.model:
        case "ppo_sb3":
            logger.info("Running PPO SB3 replay.")
        case "ppo":
            raise NotImplementedError("PPO replay is not implemented yet.")
        case "dqn":
            raise NotImplementedError("DQN replay is not implemented yet.")

    entry_point = validate_and_get_entry_point(
        args.entry, args.heading, args.level, args.curr_stage_entry_point
    )

    env = AtcGym(
        airplane_count=1,
        sim_parameters=model.SimParameters(
            1.0, discrete_action_space=False, normalize_state=True
        ),
        scenario=scenarios.SupaSupa(entry_point=entry_point),
        render_mode="human",
    )
    model_ = PPO.load(
        args.checkpoint,
        env)
    
    # model_state_dict = torch.load(args.checkpoint, map_location="cpu")
    # fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict['policy'].items()}
    # model_.policy.load_state_dict(fixed_state_dict)
    
    obs = env.reset()[0]
    done = False
    frame_count = 0
    while not done:
        action, _ = model_.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        frame_count += 1
        if frame_count % args.skip_frames == 0:
            env.render()
            time.sleep(0.05)
    env.close()
