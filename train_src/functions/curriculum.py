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
from utils.train_test_functions import evaluate, train_model
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("train.curriculum")


def train_curriculum(args, reward_keys):

    log_info("CURR with PPO", args, logger)

    """ Create the environment """

    def my_env():
        return AtcGym(
            airplane_count=2,
            sim_parameters=model.SimParameters(
                1.0, discrete_action_space=False, normalize_state=True
            ),
            scenario=scenarios.SupaSupa(),
            render_mode="headless",
        )

    env = my_env()
    vec_env = make_vec_env(my_env, n_envs=args.threads)

    curriculum_entry_points = [
        ((15, 15), 45),   # Close, NE
        ((14, 14), 90),   # East
        ((13, 13), 135),  # SE
        ((12, 12), 180),  # South
        ((12, 12), 160),  # South
        ((11, 11), 180),  # SW
        ((10, 10), 180),  # West
        ((9, 9), 180),    # NW
        ((8, 8), 180),      # North
        ((7, 7), 180),     # NE, off-axis
        ((6, 6), 180),    # SE, off-axis
        ((5, 5), 180),    # NW, off-axis
        ((0, 0), 180),     # Farthest corner, NE
        ((0, 0), 160),     # Farthest corner, NE
    ]

    for stage, (entry_xy, entry_heading) in enumerate(curriculum_entry_points):
        
        # set the avriables: eval_log_path_csv, flog_path, tensorboard_logd, tb_logger, plotter 
        stage_name = f"stage{stage+1}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}"
        stage_dir = Path(args.outdir) / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

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
                n_steps=4096,
                batch_size=4096,
            )
        elif stage > 0 and (prev_model_path := stage_dir.parent / f"curr_model_stage{stage}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}.zip").exists():
            model_ = PPO.load(
                prev_model_path,
                env=vec_env,
                verbose=1 if args.debug else 0,
                tensorboard_log=tensorboard_logd,
                n_steps=4096,
                batch_size=4096,
            )
        else:
            model_ = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1 if args.debug else 0,
                tensorboard_log=tensorboard_logd,
                n_steps=4096,
                batch_size=4096,
            )

        try:
            ep = 0
            recent_successes = []
            for ep in range(args.max_episodes):

                try:
                    logger.info(f"Stage {stage} ({entry_xy}, {entry_heading}). Episode {ep+1}/{args.max_episodes} started.")
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

                    episode_success = eval_components["success_rewards"] > 0
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

                    if ep >= args.curr_window_size and sum(recent_successes) / args.curr_window_size >= args.curr_success_threshold:
                        logger.info(f"Promoting to next stage: {stage_name}")
                        break
                    if ep >= args.max_episodes:
                        logger.info(f"Reached max episodes {args.curr_ep_per_stage} for stage: {stage_name}")
                        break

                except KeyboardInterrupt:
                    logger.info("Training interrupted by user.")
                    raise KeyboardInterrupt

            # Save model and log at end of stage
            model_.save(model_path)
            plotter.save(stage_dir)
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
    plotter.close()

    logger.info(f"Training completed. Model saved to {model_path}.")