from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

import logging

import torch

from envs.atc import model, scenarios
from envs.atc.atc_gym import AtcGym

from utils.log_stuff import log_info, set_log_paths
from utils.train_test_functions import train_model

logger = logging.getLogger("train.sb3_ppo")
logger.setLevel(logging.INFO)

def train_sb3_ppo(args, reward_keys, scenario=None, num_airplanes=1):

    eval_log_path_csv, flog_path, tensorboard_logd, tb_logger, plotter = set_log_paths(args, reward_keys)
    log_info("SB3 PPO", args, logger)

    # Create the environment, using the provided scenario if given
    def my_env():
        return AtcGym(
            airplane_count=num_airplanes,
            sim_parameters=model.SimParameters(
                1.0, discrete_action_space=False, normalize_state=True
            ),
            scenario=scenario if scenario else scenarios.SupaSupa(),
            render_mode="headless",
            wind_badness=args.wind_badness,
            wind_dirn=args.wind_dirn if hasattr(args, 'wind_dirn') else 270,  # Default to 270 if not specified
            starting_fuel=args.starting_fuel
        )
    
    env = my_env()
    vec_env = make_vec_env(my_env, n_envs=args.threads, vec_env_cls=SubprocVecEnv)
    logger.info(f"Training with {num_airplanes} airplanes in scenario: {scenario.__class__.__name__ if scenario else 'SupaSupa'}")
    logger.info(f"Wind settings - Badness: {args.wind_badness}, Direction: {args.wind_dirn if hasattr(args, 'wind_dirn') else 270}")
    logger.info(f"="*80)

    # Load from checkpoint if provided, else create new model
    if args.checkpoint:
        model_ = PPO.load(
            args.checkpoint,
            env=vec_env,
            verbose=1 if args.debug else 0,
            tensorboard_log=tensorboard_logd,
            n_steps=2048,
            batch_size=2048,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
        )
    else:
        model_ = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1 if args.debug else 0,
            tensorboard_log=tensorboard_logd,
            n_steps=2048,
            batch_size=2048,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
        )

    train_model(model_, args, logger, env, reward_keys, eval_log_path_csv, flog_path, tb_logger, plotter)
