import csv
from pathlib import Path
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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

logger = logging.getLogger("train.gen")
logger.setLevel(logging.INFO)

def train_generalized_model(args, reward_keys):

    log_info("GENERALIZE_CURR_PPO_ONLY_LOWW", args, logger)

    """ Create the environment """

    def my_env(scenario=1, scenario_obj=None, curriculum_entry_point=None):
        # For other scenarios or single airplane, use the original approach
        if curriculum_entry_point is None:
            raise ValueError("curriculum_entry_point must be provided for this scenario.")
        if scenario_obj is None:
            scenario_obj = scenarios.LOWW(entry_point=curriculum_entry_point)

        return AtcGym(
            airplane_count=1,
            sim_parameters=model.SimParameters(
                1.0, discrete_action_space=False, normalize_state=True
            ),
            scenario=scenario_obj,
            render_mode="headless",
            wind_badness=5 if scenario == 1 else 10,
            wind_dirn=270 if scenario == 1 else 220,
            reduced_time_penalty=args.reduced_time_penalty,
        )
    
    logger.info(f"Training with {1} airplanes in curriculum learning with {args.curr_stages} stages")
    logger.info(f"Using scenario: {args.scenario}")
    logger.info(f"="*80)

    scenario_1 = scenarios.LOWW() # default LOWW
    scenario_2 = scenarios.ModifiedLOWW() # modified LOWW (diff wind, diff mvas)

    curriculum_entry_points = scenarios.LOWW().generate_curriculum_entrypoint_but_many(num_entrypoints=args.curr_stages)
   
    # Keep track of environments for proper cleanup
    temp_env = None
    env = None
    vec_env = None
    
    for stage, entry_point_choices in enumerate(curriculum_entry_points, start=0):
        try:
            if stage < args.checkpoint_stage - 1:
                logger.info(f"Skipping stage {stage} as per checkpoint stage.")
                continue
        
            stage_name = f"stage{stage+1}_many_entry_many_hdg"
            
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
            model_path = stage_dir / f"curr_gen_model_{stage_name}.zip"

            try:
                ep = 0
                recent_successes = []
                
                # Close previous temporary environment if it exists
                if temp_env is not None:
                    temp_env.close()
                    temp_env = None
                
                # Create a temporary env just for model initialization with first entry point
                temp_entry_point = entry_point_choices[0]
                temp_env = make_vec_env(
                    my_env, 
                    n_envs=args.threads, 
                    env_kwargs={
                        "scenario": 1, 
                        "scenario_obj": scenario_1, 
                        "curriculum_entry_point": temp_entry_point
                    }, 
                    # vec_env_cls=DummyVecEnv
                )
                
                # Load from checkpoint if provided, else from previous stage, else create new model
                # Initialize model ONCE per stage, not per episode
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
                elif stage > 0:
                    # Try to load from previous stage
                    prev_stage_name = f"stage{stage}_many_entry_many_hdg"
                    prev_model_path = stage_dir.parent / f"curr_gen_model_{prev_stage_name}.zip"
                    
                    if prev_model_path.exists():
                        model_ = PPO.load(
                            prev_model_path,
                            env=temp_env,
                            verbose=1 if args.debug else 0,
                            tensorboard_log=tensorboard_logd,
                            n_steps=2048,
                            batch_size=2048,
                        )
                        logger.info(f"Loaded model from previous stage: {prev_model_path}")
                    else:
                        # Create new model if no previous stage found
                        model_ = PPO(
                            "MlpPolicy",
                            temp_env,
                            verbose=1 if args.debug else 0,
                            tensorboard_log=tensorboard_logd,
                            n_steps=2048,
                            batch_size=2048,
                        )
                        logger.info("Created new model (previous stage model not found)")
                else:
                    # Create new model for first stage
                    model_ = PPO(
                        "MlpPolicy",
                        temp_env,
                        verbose=1 if args.debug else 0,
                        tensorboard_log=tensorboard_logd,
                        n_steps=2048,
                        batch_size=2048,
                    )
                    logger.info("Created new model for first stage")
                
                # For tracking entry point cycling (train 10 episodes per entry point)
                entry_point_index = 0
                episodes_on_current_entry_point = 0
                entry_point_cycle_size = 10  # Number of episodes to train on each entry point
                
                # Keep track of environments to properly close them
                prev_env = None
                prev_vec_env = None
                
                for ep in range(args.max_episodes):
                    if ep % 100 == 0:
                        # Every 100 episodes, we change the scenario between one of two scenario set ups
                        chosen_scenario = scenario_1 if ep % 200 == 0 else scenario_2
                        scenario_code = 1 if ep % 200 == 0 else 2
                    else:
                        chosen_scenario = scenario_1
                        scenario_code = 1
                    
                    # Change entry point after entry_point_cycle_size episodes
                    if episodes_on_current_entry_point >= entry_point_cycle_size:
                        entry_point_index = (entry_point_index + 1) % len(entry_point_choices)
                        episodes_on_current_entry_point = 0
                        logger.info(f"Switching to next entry point after {entry_point_cycle_size} episodes")
                    
                    # Use the current entry point
                    entry_point = entry_point_choices[entry_point_index]
                    episodes_on_current_entry_point += 1
                    
                    # Log the chosen scenario and entry point
                    logger.info(f"Chosen scenario: {chosen_scenario}, Entry point: {entry_point} (episode {episodes_on_current_entry_point}/{entry_point_cycle_size} on this entry point)")
                    
                    # Close previous environments if they exist to prevent "Too many open files" error
                    if prev_env is not None:
                        prev_env.close()
                    if prev_vec_env is not None:
                        prev_vec_env.close()
                    
                    # Create environment with the current curriculum entry point
                    env = my_env(scenario_code, chosen_scenario, entry_point)
                    vec_env = make_vec_env(
                        my_env, 
                        n_envs=args.threads, 
                        env_kwargs={
                            "scenario": scenario_code, 
                            "scenario_obj": chosen_scenario, 
                            "curriculum_entry_point": entry_point
                        }, 
                        # vec_env_cls=SubprocVecEnv
                    )
                    
                    # Store references to current environments for cleanup in next iteration
                    prev_env = env
                    prev_vec_env = vec_env
                    
                    # Reset environments
                    vec_env.reset()
                    env.reset()

                    # Update the environment of the existing model without recreating it
                    model_.set_env(vec_env)
                    
                    try:
                        logger.info(f"Stage {stage} (Entry point at stage {stage+1}/{len(curriculum_entry_points)}). Episode {ep+1}/{args.max_episodes} started.")
                        start_time = time.time()
                        model_.learn(
                            total_timesteps=args.max_steps_per_episode,
                            reset_num_timesteps=False,
                            progress_bar=True,
                            tb_log_name="curr"
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

                        episode_success = eval_components["success_rewards"] > (500 * 1) 
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

                # Clean up the last environments from the episode loop
                if prev_env is not None:
                    prev_env.close()
                if prev_vec_env is not None:
                    prev_vec_env.close()
                    
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
                # Clean up environments before breaking
                if prev_env is not None:
                    prev_env.close()
                if prev_vec_env is not None:
                    prev_vec_env.close()
                if temp_env is not None:
                    temp_env.close()
                break
        
        finally:
            # Ensure environments are closed even if an exception occurs
            if 'prev_env' in locals() and prev_env is not None:
                prev_env.close()
            if 'prev_vec_env' in locals() and prev_vec_env is not None:
                prev_vec_env.close()
            if temp_env is not None:
                temp_env.close()
    
    # Final cleanup and model saving
    model_path = Path(args.outdir) / f"gen_curriculum_stage_{stage}_model_final.zip"
    model_.save(model_path)

    # Make sure we have valid environments for final evaluation
    if env is None or vec_env is None:
        # Create a simple environment for evaluation if needed
        env = my_env(1, scenario_1, entry_point_choices[0])
        
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

    # Final cleanup
    if env is not None:
        env.close()
    if vec_env is not None:
        vec_env.close()
    if temp_env is not None:
        temp_env.close()
        
    if args.live_plot:
        plotter.close()

    logger.info(f"Training completed. Model saved to {model_path}.")