import csv
import platform
from pathlib import Path
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO

import logging

import torch

from envs.atc import model, scenarios
from envs.atc.atc_gym import AtcGym

from utils.log_stuff import (
    RewardPlotter,
    human_readable_time,
    log_info,
    log_model_stuff,
)
from utils.train_test_functions import evaluate, train_model
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("train.curriculum")
logger.setLevel(logging.INFO)

def train_curriculum(args, reward_keys, scenario=None, num_airplanes=1):

    log_info("CURR with PPO", args, logger)

    """ Create the environment """

    def my_env(curriculum_entry_point=None):
        # Using LOWW scenario for two-airplane training with curriculum
        if args.scenario == 'LOWW' and num_airplanes == 2:
            # For LOWW with multiple airplanes, we receive a list of entry points (one per plane)
            if curriculum_entry_point is not None:
                # Create LOWW scenario with entry points for multiple planes
                scenario_obj = scenarios.LOWW(entry_point=curriculum_entry_point)
            else:
                # Use default LOWW scenario if no entry points are provided
                scenario_obj = scenarios.LOWW()
        else:
            # For other scenarios or single airplane, use the original approach
            if curriculum_entry_point is not None:
                scenario_obj = scenarios.SupaSupa(entry_point=curriculum_entry_point)
            else:
                scenario_obj = scenarios.SupaSupa()
        
        return AtcGym(
            airplane_count=num_airplanes,
            sim_parameters=model.SimParameters(
                1.0, discrete_action_space=False, normalize_state=True
            ),
            scenario=scenario_obj,
            render_mode="headless",
            wind_badness=args.wind_badness,
            starting_fuel=args.starting_fuel if hasattr(args, 'starting_fuel') else None
        )
    
    logger.info(f"Training with {num_airplanes} airplanes in curriculum learning")
    logger.info(f"Using scenario: {args.scenario}")
    logger.info(f"="*80)

    # Determine which vector environment to use based on OS
    # Windows has issues with SubprocVecEnv, so use DummyVecEnv on Windows
    if platform.system() == 'Windows':
        vec_env_cls = DummyVecEnv
        logger.info("Using DummyVecEnv for Windows compatibility")
        # Adjust threads for Windows - DummyVecEnv doesn't benefit from many threads
        actual_threads = min(4, args.threads)
        if actual_threads < args.threads:
            logger.info(f"Reducing threads from {args.threads} to {actual_threads} for Windows DummyVecEnv")
    else:
        vec_env_cls = SubprocVecEnv
        actual_threads = args.threads
        logger.info(f"Using SubprocVecEnv with {actual_threads} threads")

    # Generate curriculum entry points based on scenario
    if args.scenario == 'LOWW' and num_airplanes == 2:
        # Use LOWW's curriculum entry points generator for 2 airplanes
        curriculum_entry_points = scenarios.LOWW().generate_curriculum_entrypoints(num_entrypoints=args.curr_stages)
    else:
        # Use default SupaSupa curriculum for other scenarios
        curriculum_entry_points = scenarios.SupaSupa().generate_curriculum_entrypoints(num_entrypoints=args.curr_stages)

    for stage, entry_point in enumerate(curriculum_entry_points, start=0):

        if stage < args.checkpoint_stage - 1:
            logger.info(f"Skipping stage {stage} as per checkpoint stage.")
            continue

        # Create environment with the current curriculum entry point
        env = my_env(entry_point)
        vec_env = make_vec_env(my_env, n_envs=actual_threads, env_kwargs={"curriculum_entry_point": entry_point}, vec_env_cls=vec_env_cls)

        # For LOWW with 2 planes, each entry_point is actually a list of two entry points
        if args.scenario == 'LOWW' and num_airplanes == 2:
            # Use first plane's coordinates for stage naming
            entry_xy = (entry_point[0].x, entry_point[0].y)
            entry_heading = entry_point[0].phi
            stage_name = f"stage{stage+1}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}"
        else:
            # For single plane scenarios, use the standard approach
            entry_xy = (entry_point.x, entry_point.y)
            entry_heading = entry_point.phi
            stage_name = f"stage{stage+1}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}"
        
        stage_dir = Path(args.outdir) / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging paths
        if args.log_csv:
            eval_log_path_csv = Path(stage_dir) / "evaluation_log.csv"
        else:
            eval_log_path_csv = None

        if args.log_file:
            flog_path = Path(stage_dir) / "log.txt"
        else:
            flog_path = None

        if args.log_tensorboard:
            tensorboard_logd = Path(stage_dir) / "tensorboard"
            tensorboard_logd.mkdir(parents=True, exist_ok=True)
            # Create TensorBoard logger
            tb_logger = SummaryWriter(log_dir=str(tensorboard_logd))
        else:
            tensorboard_logd = None
            tb_logger = None

        if args.live_plot:
            plotter = RewardPlotter(reward_keys)
        else:
            plotter = None
        
        # Save the model path for the current stage
        model_path = stage_dir / f"curr_model_{stage_name}.zip"

        # Load from checkpoint if provided, else create new model
        if args.checkpoint and Path(args.checkpoint).exists():
            model_ = PPO.load(
                args.checkpoint,
                env=vec_env,
                verbose=1 if args.debug else 0,
                tensorboard_log=tensorboard_logd,
                n_steps=2048,
                batch_size=4096,
            )
        elif stage > 0 and (prev_model_path := stage_dir.parent / f"curr_model_stage{stage}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}.zip").exists():
            model_ = PPO.load(
                prev_model_path,
                env=vec_env,
                verbose=1 if args.debug else 0,
                tensorboard_log=tensorboard_logd,
                n_steps=2048,
                batch_size=2048,
            )
        else:
            model_ = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1 if args.debug else 0,
                tensorboard_log=tensorboard_logd,
                n_steps=2048,
                batch_size=2048,
            )

        try:
            ep = 0
            recent_successes = []
            for ep in range(args.max_episodes):

                try:
                    logger.info(f"Stage {stage} (Entry point at stage {stage+1}/{len(curriculum_entry_points)}). Episode {ep+1}/{args.max_episodes} started.")
                    start_time = time.time()
                    model_.learn(
                        total_timesteps=args.max_steps_per_episode,
                        reset_num_timesteps=False,
                        progress_bar=True,
                        tb_log_name="ppo_sb3"
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
                        plotter.save(stage_dir)

                    episode_success = eval_components["success_rewards"] > (500 * num_airplanes) 
                    recent_successes.append(episode_success)
                    if len(recent_successes) > args.curr_window_size:
                        recent_successes.pop(0)

                    end_time = time.time()
                    logger.info(f"Episode {ep+1}/{args.max_episodes} completed. {(args.max_steps_per_episode / (end_time - start_time)):.2f} FPS. Est. Remaining time: {human_readable_time((args.max_episodes - ep - 1) * (end_time - start_time))}.")
                    
                    # Save model checkpoint
                    if (ep+1) % args.save_freq == 0:
                        model_path = stage_dir / f"curr_model_{stage_name}_ep{ep+1}.zip"
                        model_.save(model_path)
                        logger.info("=" * 80)
                        logger.info(f"[SAVED] Model saved to '{model_path}'")
                        logger.info("=" * 80)

                    if ep >= args.curr_window_size:
                        logger.info(f"Recent successes: {recent_successes}. Attempting promotion to the next stage.")
                        if sum(recent_successes) / args.curr_window_size >= args.curr_success_threshold:
                            logger.info(f"Promoting to next stage: {stage_name} with value {sum(recent_successes) / args.curr_window_size:.2f} >= {args.curr_success_threshold}")
                            break
                        else:
                            logger.info(f"Not promoting to next stage: {stage_name} with value {sum(recent_successes) / args.curr_window_size:.2f} < {args.curr_success_threshold}")
       
                    if ep >= args.max_episodes:
                        logger.info(f"Reached max episodes {args.max_episodes} for stage: {stage_name}")
                        break

                except KeyboardInterrupt:
                    logger.info("Training interrupted by user.")
                    raise KeyboardInterrupt

            # Save model and log at end of stage
            model_.save(model_path)
            if args.live_plot:
                plotter.save(stage_dir)
                plotter.close()

            with open(stage_dir / f"training_log_{stage_name}.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(["final", eval_rewards] + [eval_components[k] for k in reward_keys])

        except KeyboardInterrupt:
            logger.info("Stage training interrupted!")
            break
    
    model_path = Path(args.outdir) / f"curriculum_stage_{stage}_model_final.zip"
    model_.save(model_path)

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
        plotter.save(args.outdir)

    if args.log_tensorboard:
        tb_logger.close()

    env.close()
    if args.live_plot:
        plotter.close()

    logger.info(f"Training completed. Model saved to {model_path}.")