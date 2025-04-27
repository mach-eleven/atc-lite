"""
Train the RL agent using the selected algorithm.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from utils.type_checkers import gt_0, checkpoint_type, gt_0_float, outdir_type, log_list_type, parse_log_config
import logging
from rich.logging import RichHandler

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("train")

def add_arguments(parser):
    """
    Add command line arguments for training.
    """
    parser.add_argument('--model', type=str, default='ppo_sb3', choices=['curr', 'ppo_sb3', 'ppo', 'dqn'], help='RL algorithm to use')
    parser.add_argument('--checkpoint', type=checkpoint_type, default=None, help='Path to SB3 PPO checkpoint to resume training from.')
    parser.add_argument('--outdir', type=outdir_type, default='new_logs/ppo_sb3', help='Output directory for logs and models')
    parser.add_argument('--max-episodes', type=gt_0, default=500, help='Number of episodes to train the agent')
    parser.add_argument('--max-steps-per-episode', type=gt_0, default=50_000, help='Maximum steps per episode')
    parser.add_argument('--save-freq', type=gt_0, default=20, help='Frequency of saving the model checkpoint, in episodes') # 10 episodes rn
    parser.add_argument('--eval-freq', type=gt_0, default=2, help='Frequency of evaluating the agent, in episodes')
    parser.add_argument('--eval-episodes', type=gt_0, default=5, help='Number of episodes to evaluate the agent')
    parser.add_argument('--log-where', type=log_list_type, default=['csv', 'tensorboard'], help='Where to log the training progress (csv, tensorboard, file)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, i.e. stdout logging.', default=False)
    parser.add_argument('--live-plot', action='store_true', help='Enable live plotting of evaluation progress, updated every eval freqeuncy.', default=True)
    parser.add_argument('--threads', type=gt_0, default=8, help='Number of threads to use for training')
    parser.add_argument('--curr-window-size', type=gt_0, default=50, help='Window size for success rate check in curriculum training')
    parser.add_argument('--curr-success-threshold', type=gt_0_float, default=0.90, help='Success rate threshold for curriculum training')
    parser.add_argument('--curr-stages', type=gt_0, default=100, help='Number of stages for curriculum training')

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

    args.log_csv, args.log_tensorboard, args.log_file = parse_log_config(args.log_where)

    reward_keys = [
        'success_rewards', 'airspace_penalties', 'mva_penalties',
        'time_penalty', 'approach_position_rewards', 'approach_angle_rewards',
        'glideslope_rewards', 'fuel_efficiency_rewards', 'fuel_penalties'
    ]

    match args.model:
        case 'curr':
            from train_src.functions import train_curriculum
            train_curriculum(args, reward_keys)
        case 'ppo_sb3':
            from train_src.functions import train_sb3_ppo
            train_sb3_ppo(args, reward_keys)
        case 'ppo':
            from train_src.functions import train_ppo
            train_ppo(args, reward_keys)
        case 'dqn':
            from train_src.functions import train_dqn
            train_dqn(args, reward_keys)
        case _:
            raise ValueError(f"Unknown model type: {args.model}")
