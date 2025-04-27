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

def train_sb3_ppo(args, reward_keys):

    eval_log_path_csv, flog_path, tensorboard_logd, tb_logger, plotter = set_log_paths(args, reward_keys)
    log_info("SB3 PPO", args, logger)

    """ Create the environment """
    def my_env():
        return AtcGym(
            airplane_count=1,
            sim_parameters=model.SimParameters(
                1.0, discrete_action_space=False, normalize_state=True
            ),
            scenario=scenarios.SupaSupa(),
            render_mode="headless",
        )
        
    env = my_env()
    vec_env = make_vec_env(my_env, n_envs=args.threads, vec_env_cls=SubprocVecEnv)
    logger.info(f"="*80)

    # Load from checkpoint if provided, else create new model
    if args.checkpoint:  # os path has been checked in the main function
        model_ = PPO.load(
            args.checkpoint,
            env=vec_env,
            verbose=1 if args.debug else 0,
            tensorboard_log=tensorboard_logd,
            n_steps=2048,
            batch_size=2048,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,  # GAE parameter
            clip_range=0.2,  # PPO clipping parameter
            ent_coef=0.01,  # Entropy coefficient
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
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
            gae_lambda=0.95,  # GAE parameter
            clip_range=0.2,  # PPO clipping parameter
            ent_coef=0.01,  # Entropy coefficient
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            n_epochs=10,
        )

    model_.policy = torch.compile(model_.policy)

    """ Main training loop """
    train_model(model_, args, logger, env, reward_keys, eval_log_path_csv, flog_path, tb_logger, plotter)
