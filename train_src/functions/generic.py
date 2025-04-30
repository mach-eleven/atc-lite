import csv
from pathlib import Path
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

import logging

from envs.atc import model, scenarios
from envs.atc.atc_gym import AtcGym

from utils.log_stuff import (
    RewardPlotter,
    human_readable_time,
    log_info,
    log_model_stuff,
)
from utils.train_test_functions import evaluate
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("train.gen")
logger.setLevel(logging.INFO)

def train_generic_simple(args, reward_keys):

    log_info("GENERIC_PPO_NO_CURRICULUM", args, logger)

    """ Create the environment """

    def my_env(entry_point=None):
        # Create a scenario with the specified entry point
        if entry_point is None:
            scenario_obj = scenarios.LOWW()
        else:
            scenario_obj = scenarios.LOWW(entry_point=entry_point)

        return AtcGym(
            airplane_count=1,
            sim_parameters=model.SimParameters(
                1.0, discrete_action_space=False, normalize_state=True
            ),
            scenario=scenario_obj,
            render_mode="headless",
            wind_badness=5,
            wind_dirn=270,
            reduced_time_penalty=args.reduced_time_penalty,
        )
    
    logger.info(f"Training with {1} airplane without curriculum learning")
    logger.info(f"Using scenario: LOWW with 5 fixed entry points")
    logger.info(f"="*80)

    
    # Get the last stage of entry points (assuming 1 stage)
    entry_points = scenarios.LOWW().generate_last_bound_entry_points()
    logger.info(f"Entry points: {entry_points}")
    logger.info(f"Selected {len(entry_points)} entry points for training")
    
    # Setup the output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Set up logging paths
    if args.log_csv:
        eval_log_path_csv = Path(outdir) / "evaluation_log.csv"
    else:
        eval_log_path_csv = None

    if args.log_file:
        flog_path = Path(outdir) / "log.txt"
    else:
        flog_path = None

    if args.log_tensorboard:
        tensorboard_logd = Path(outdir) / "tensorboard"
        tensorboard_logd.mkdir(parents=True, exist_ok=True)
        tb_logger = SummaryWriter(log_dir=str(tensorboard_logd))
    else:
        tensorboard_logd = None
        tb_logger = None

    if args.live_plot:
        plotter = RewardPlotter(reward_keys)
    else:
        plotter = None
    
    # Model checkpoint path
    model_path = outdir / "generic_model.zip"

    # Create a temporary environment for model initialization with the first entry point
    temp_env = make_vec_env(
        my_env, 
        n_envs=args.threads, 
        env_kwargs={"entry_point": entry_points[0]}, 
    )
    
    # Initialize model
    if args.checkpoint and Path(args.checkpoint).exists():
        model_ = PPO.load(
            args.checkpoint,
            env=temp_env,
            verbose=1 if args.debug else 0,
            tensorboard_log=tensorboard_logd,
            n_steps=2048,
            batch_size=4096,
        )
        logger.info(f"Loaded model from checkpoint: {args.checkpoint}")
    else:
        model_ = PPO(
            "MlpPolicy",
            temp_env,
            verbose=1 if args.debug else 0,
            tensorboard_log=tensorboard_logd,
            n_steps=2048,
            batch_size=2048,
        )
        logger.info("Created new model")
    
    # For tracking entry point cycling
    entry_point_index = 0
    episodes_on_current_entry_point = 0
    entry_point_cycle_size = 10  # Number of episodes to train on each entry point
    
    # Keep track of environments for proper cleanup
    prev_env = None
    prev_vec_env = None
    
    try:
        for ep in range(args.max_episodes):
            # Change entry point after entry_point_cycle_size episodes
            if episodes_on_current_entry_point >= entry_point_cycle_size:
                entry_point_index = (entry_point_index + 1) % len(entry_points)
                episodes_on_current_entry_point = 0
                logger.info(f"Switching to next entry point after {entry_point_cycle_size} episodes")
            
            # Use the current entry point
            entry_point = entry_points[entry_point_index]
            episodes_on_current_entry_point += 1
            
            # Log the current entry point
            logger.info(f"Using entry point: {entry_point} (episode {episodes_on_current_entry_point}/{entry_point_cycle_size} on this entry point)")
            
            # Close previous environments if they exist
            if prev_env is not None:
                prev_env.close()
            if prev_vec_env is not None:
                prev_vec_env.close()
            
            # Create environment with the current entry point
            env = my_env(entry_point)
            vec_env = make_vec_env(
                my_env, 
                n_envs=args.threads, 
                env_kwargs={"entry_point": entry_point}, 
            )
            
            # Store references to current environments for cleanup in next iteration
            prev_env = env
            prev_vec_env = vec_env
            
            # Reset environments
            vec_env.reset()
            env.reset()
            
            # Set the environment for the model
            model_.set_env(vec_env)
            
            try:
                logger.info(f"Episode {ep+1}/{args.max_episodes} started.")
                start_time = time.time()
                model_.learn(
                    total_timesteps=args.max_steps_per_episode,
                    reset_num_timesteps=False,
                    progress_bar=True,
                    tb_log_name="generic"
                )
                
                eval_rewards, eval_components, success_rate = evaluate(
                    model_, env, logger, args.eval_episodes, args.max_steps_per_episode, reward_keys
                )
                
                logger.info(f"Logging model at episode {ep+1}...")
                log_model_stuff(
                    args.log_tensorboard,
                    args.log_csv,
                    args.log_file,
                    ep+1,
                    eval_rewards,
                    eval_components,
                    success_rate,
                    eval_log_path_csv,
                    flog_path,
                    reward_keys,
                    tblogger=tb_logger,
                )

                if args.live_plot:
                    plotter.update(ep, eval_components)
                    plotter.save(outdir)

                end_time = time.time()
                logger.info(f"Episode {ep+1}/{args.max_episodes} completed. {(args.max_steps_per_episode / (end_time - start_time)):.2f} FPS. Est. Remaining time: {human_readable_time((args.max_episodes - ep - 1) * (end_time - start_time))}.")
                
                # Save model checkpoint periodically
                if (ep+1) % args.save_freq == 0:
                    checkpoint_path = outdir / f"generic_model_ep{ep+1}.zip"
                    model_.save(checkpoint_path)
                    logger.info("=" * 80)
                    logger.info(f"[SAVED] Model saved to '{checkpoint_path}'")
                    logger.info("=" * 80)
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user.")
                break
                
    finally:
        # Final cleanup
        if 'prev_env' in locals() and prev_env is not None:
            prev_env.close()
        if 'prev_vec_env' in locals() and prev_vec_env is not None:
            prev_vec_env.close()
        if temp_env is not None:
            temp_env.close()
    
    # Save final model
    model_.save(model_path)
    
    # Final evaluation
    env = my_env(entry_points[0])
    eval_rewards, eval_components, success_rate = evaluate(
        model_, env, logger, args.eval_episodes, args.max_steps_per_episode, reward_keys, no_progress_bar=True
    )
    log_model_stuff(
        args.log_tensorboard,
        args.log_csv,
        args.log_file,
        ep+1,
        eval_rewards,
        eval_components,
        success_rate,
        eval_log_path_csv,
        flog_path,
        reward_keys,
        tblogger=tb_logger,
    )
    
    if args.live_plot:
        plotter.update(ep, eval_components)
        plotter.save(outdir)
        plotter.close()

    if args.log_tensorboard:
        tb_logger.close()

    # Final cleanup
    if env is not None:
        env.close()
    
    logger.info(f"Training completed. Final model saved to {model_path}.")