from pathlib import Path
from rich.progress import track 
import numpy as np
import time

from utils.log_stuff import human_readable_time, log_model_stuff

def evaluate(model_, env, logger, n_episodes, max_steps, reward_keys, deterministic=True, no_progress_bar=False):
    """Custom evaluation to also track component rewards."""
    total_rewards = []
    total_components = {k: 0.0 for k in reward_keys}
    
    for ep_idx in track(range(n_episodes), description="Evaluating", total=n_episodes) if not no_progress_bar else range(n_episodes):
        obs = env.reset()[0]
        done = False
        ep_reward = 0
        ep_components = {k: 0.0 for k in reward_keys}

        ts = 0
        while not done:
            action, _ = model_.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if "reward_components" in info:
                for k in reward_keys:
                    ep_components[k] += info["reward_components"].get(k, 0)
            if ts >= max_steps:
                break
            ts += 1

        total_rewards.append(ep_reward)
        for k in reward_keys:
            total_components[k] += ep_components[k]
        
        logger.debug(f"Evaluation episode {ep_idx+1}/{n_episodes} completed with reward: {ep_reward:.2f}")

    avg_reward = sum(total_rewards) / n_episodes
    avg_components = {k: float(v / n_episodes) for k, v in total_components.items()}
    success_rate = np.mean(np.array(total_rewards) > 0)
    logger.info(f"Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
    logger.info(f"Avg Components: {avg_components}")
    
    return avg_reward, avg_components, success_rate


def train_model(model_, args, logger, env, reward_keys, eval_log_path_csv, flog_path, tb_logger, plotter):
    ep = 0
    for ep in range(args.max_episodes):

        try:
            logger.info(f"Episode {ep+1}/{args.max_episodes} started.")
            start_time = time.time()
            model_.learn(
                total_timesteps=args.max_steps_per_episode,
                reset_num_timesteps=False,
                progress_bar=True,
                tb_log_name="ppo_sb3"
            )

            end_time = time.time()
            logger.info(f"Episode {ep+1}/{args.max_episodes} completed. {(args.max_steps_per_episode / (end_time - start_time)):.2f} FPS. Est. Remaining time: {human_readable_time((args.max_episodes - ep - 1) * (end_time - start_time))}.")
            
            # Save model checkpoint
            if (ep+1) % args.save_freq == 0:
                model_path = Path(args.outdir) / f"sb3_ppo_model_{ep+1}.zip"
                model_.save(model_path)
                logger.info("=" * 80)
                logger.info(f"[SAVED] Model saved to '{model_path}'")
                logger.info("=" * 80)

            # Log training progress
            if (ep+1) % args.eval_freq == 0:
                logger.info(f"Evaluating model at episode {ep+1}...")
                eval_rewards, eval_components, success_rate = evaluate(
                    model_, env, logger, args.eval_episodes, args.max_steps_per_episode, reward_keys
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

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            break
        
    model_path = Path(args.outdir) / "sb3_ppo_model_final.zip"
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