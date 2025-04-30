"""
Train the RL agent using the selected algorithm.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import cv2  # Use OpenCV for MP4 saving
import random  # Add import for random selection

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

# Store current entry point index for sequential selection
current_entry_point_index = 0

def validate_and_get_entry_point(entry, heading, level, curr_stage_entry_point, scenario_name, num_airplanes, episode_num=None):
    """
    Validates and returns the appropriate entry point(s) based on command line arguments.
    For LOWW with 2 airplanes, returns a list of entry points, one for each airplane.
    For other scenarios, returns a single entry point.
    
    Args:
        episode_num: if provided, will use this to determine which entry point to use sequentially
    """
    global current_entry_point_index
    
    # Special handling for LOWW scenario with 2 airplanes
    if scenario_name == "LOWW" and num_airplanes == 2:
        # If specific entry point parameters are provided, use them
        if entry is not None and heading is not None and level is not None:
            # Create two entry points with slightly different positions to avoid collisions
            entry_point1 = model.EntryPoint(entry[0], entry[1], heading, level)
            # Second plane starts from a different approach path
            entry_point2 = model.EntryPoint(entry[0] - 10, entry[1] - 10, heading - 45, level)
            return [entry_point1, entry_point2]
        elif entry is not None:
            raise ValueError("If entry is provided, heading and level must also be provided.")
        elif heading is not None or level is not None:
            raise ValueError("If heading or level is provided, entry must also be provided.")
        elif curr_stage_entry_point == "max":
            # Use default LOWW entry points for 2 airplanes
            return scenarios.LOWW().entrypoints
        else:
            # Get curriculum entry points for the specified stage
            if curr_stage_entry_point < 1 or curr_stage_entry_point > args.curr_stages:
                raise ValueError(
                    f"Curriculum stage {curr_stage_entry_point} is out of range. Must be between 1 and {args.curr_stages}."
                )
            return scenarios.LOWW().generate_curriculum_entrypoints(
                num_entrypoints=args.curr_stages
            )[curr_stage_entry_point - 1]
    
    # Special handling for LOWW or ModifiedLOWW with 1 airplane using the special curriculum system
    elif scenario_name in ["LOWW", "ModifiedLOWW"] and num_airplanes == 1:
        if entry is not None and heading is not None and level is not None:
            entry_point = model.EntryPoint(entry[0], entry[1], heading, level)
        elif entry is not None:
            raise ValueError("If entry is provided, heading and level must also be provided.")
        elif heading is not None or level is not None:
            raise ValueError("If heading or level is provided, entry must also be provided.")
        elif curr_stage_entry_point == "max":
            # Use the first entry point of the LOWW scenario
            scenario_instance = getattr(scenarios, scenario_name)()
            entry_point = random.choice(scenario_instance.entrypoints)
        else:
            # Get curriculum entry points using the special many-entry-point system
            if curr_stage_entry_point < 1 or curr_stage_entry_point > args.curr_stages:
                raise ValueError(
                    f"Curriculum stage {curr_stage_entry_point} is out of range. Must be between 1 and {args.curr_stages}."
                )
            
            # Get the list of entry points from the specified stage
            scenario_instance = getattr(scenarios, scenario_name)()
            entry_points_list = scenario_instance.generate_curriculum_entrypoint_but_many(
                num_entrypoints=args.curr_stages
            )[curr_stage_entry_point - 1]
            
            # Get alternate entry points (5 entry points)
            alternates_only = [x[1] for x in enumerate(entry_points_list) if x[0] % 2 != 0 and x[0] != 0]
            
            logger.info(f"Alternates only: {alternates_only}")
            
            # Instead of randomly selecting, use sequential selection based on episode number
            if episode_num is not None:
                # Use modulo to cycle through the 5 entry points
                index = episode_num % len(alternates_only)
                entry_point = alternates_only[index]
                logger.info(f"Using entry point {index+1}/{len(alternates_only)} for episode {episode_num+1}")
            else:
                # Use and increment the global index for sequential selection
                index = current_entry_point_index % len(alternates_only)
                entry_point = alternates_only[index]
                logger.info(f"Using entry point {index+1}/{len(alternates_only)} sequentially")
                current_entry_point_index += 1
                
        return entry_point
    
    # Regular handling for other scenarios or LOWW with 1 airplane
    else:
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
            if scenario_name == "SupaSupa":
                entry_point = scenarios.SupaSupa().entrypoints[0]
            else:
                # Use the first entry point of the specified scenario
                scenario_instance = getattr(scenarios, scenario_name)()
                entry_point = scenario_instance.entrypoints[0]
        else:
            # Get curriculum entry points for the specified stage
            if curr_stage_entry_point < 1 or curr_stage_entry_point > args.curr_stages:
                raise ValueError(
                    f"Curriculum stage {curr_stage_entry_point} is out of range. Must be between 1 and {args.curr_stages}."
                )
                
            if scenario_name == "SupaSupa":
                entry_point = scenarios.SupaSupa().generate_curriculum_entrypoints(
                    num_entrypoints=args.curr_stages
                )[curr_stage_entry_point - 1]
            elif scenario_name == "LOWW":
                entry_point = scenarios.LOWW().generate_curriculum_entrypoints(
                    num_entrypoints=args.curr_stages
                )[curr_stage_entry_point - 1][0]  # Use first airplane's entry point
            else:
                # Try to call generate_curriculum_entrypoints if it exists
                scenario_instance = getattr(scenarios, scenario_name)()
                if hasattr(scenario_instance, "generate_curriculum_entrypoints"):
                    entry_point = scenario_instance.generate_curriculum_entrypoints(
                        num_entrypoints=args.curr_stages
                    )[curr_stage_entry_point - 1]
                else:
                    # Fall back to the first entry point if curriculum not supported
                    entry_point = scenario_instance.entrypoints[0]
                
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
        default=50,
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
    parser.add_argument(
        "--episodes",
        type=gt_0,
        default=5,
        help="Number of episodes to run the replay",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1,
        help="Sleep time between frames (for rendering)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="SupaSupa",
        help="Scenario to use for demonstration",
    )
    parser.add_argument(
        "--num-airplanes", type=gt_0, default=1, help="Number of airplanes to demo with"
    )
    parser.add_argument('--pause-frame', action='store_true', default=True)
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Save the last episode replay as an MP4 file.",
        default=False,
    )
    parser.add_argument(
        "--random-entry",   
        action="store_true", help="Use random entry points for airplanes", default=False
    )
    parser.add_argument(
        "--savetraj",
        action="store_true",
        help="Save the trajectory of all airplanes in a Python file after replay.",
        default=False,
    )
    parser.add_argument(
        "--starting-fuel",
        type=gt_0,
        default=10000,
        help="Amount of fuel (kg) the airplane starts with during replay",
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

    from envs.atc import scenarios

    # Load the specified scenario
    logger.info(f"Using scenario: {args.scenario} with {args.num_airplanes} airplanes")

    # Create scenario instance based on scenario name
    try:
        scenario_class = getattr(scenarios, args.scenario)
    except AttributeError:
        logger.error(f"Scenario {args.scenario} not found in envs.atc.scenarios.")
        sys.exit(1)
        
    # Get the initial entry point(s) for the scenario
    entry_point = validate_and_get_entry_point(
        args.entry, args.heading, args.level, args.curr_stage_entry_point, args.scenario, args.num_airplanes
    )
    if type(entry_point.levels) != list:
        entry_point.levels = [entry_point.levels]

    logger.info(f"Initial entry point(s): {entry_point}")
    
    # Create the scenario with appropriate parameters
    if args.scenario in ["SimpleScenario", "SuperSimple"]:
        scenario = scenario_class(random_entrypoints=args.random_entry)
    elif args.scenario == "LOWW":
        # For LOWW with 2 airplanes, we need to set the entry points
        if args.num_airplanes == 2:
            # Pass the entry points to the scenario
            scenario = scenario_class(random_entrypoints=args.random_entry, entry_point=entry_point)
        else:
            scenario = scenario_class(entry_point=entry_point)
    elif args.scenario == "ModifiedLOWW":
        scenario = scenario_class(entry_point=entry_point)
    elif args.scenario == "SupaSupa":
        scenario = scenario_class(entry_point=entry_point)
    elif args.scenario == "CurriculumTrainingScenario":
        scenario = scenario_class()  # Uses default entry points
    else:
        scenario = scenario_class()

    render_mode = "rgb_array" if args.mp4 else "human"  # Change render mode if mp4 is requested

    env = AtcGym(
        airplane_count=args.num_airplanes,
        sim_parameters=model.SimParameters(
            1.0, discrete_action_space=False, normalize_state=True
        ),
        scenario=scenario,
        render_mode=render_mode,  # Use the determined render mode
        starting_fuel=args.starting_fuel,
    )
    model_ = PPO.load(
        args.checkpoint,
        env,
    )

    frames = []  # Initialize list to store frames for mp4

    # --- Trajectory storage ---
    all_trajectories = []  # List of list of trajectories for each episode if needed

    for ep in range(args.episodes):

        # Update entry points for LOWW and ModifiedLOWW scenarios
        if args.scenario in ["LOWW", "ModifiedLOWW"]:
            entry_point = validate_and_get_entry_point(
                args.entry, args.heading, args.level, args.curr_stage_entry_point, 
                args.scenario, args.num_airplanes, episode_num=ep
            )
            logger.info(f"Updated entry point(s) for episode {ep+1}: {entry_point}")
            scenario = scenario_class(entry_point=entry_point)
            env = AtcGym(
                airplane_count=args.num_airplanes,
                sim_parameters=model.SimParameters(
                    1.0, discrete_action_space=False, normalize_state=True
                ),
                scenario=scenario,
                render_mode=render_mode,
            )
            model_.set_env(env)

        obs, _ = env.reset()
        done = False
        frame_count = 0
        total_reward = 0
        episode_frames = []  # Store frames for the current episode if needed
        # Initialize trajectory: one list per airplane
        traj = [[] for _ in range(args.num_airplanes)]
        while not done:
            action, _ = model_.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            frame_count += 1
            total_reward += reward

            # --- Collect positions for each airplane ---
            for i, airplane in enumerate(env._airplanes):
                traj[i].append((float(airplane.x), float(airplane.y)))

            # Render and store frame if it's the last episode and mp4 is requested
            is_last_episode = (ep == args.episodes - 1)
            if is_last_episode and args.mp4:
                if frame_count % args.skip_frames == 0:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                    time.sleep(args.sleep)  # Keep sleep for pacing if needed, even when saving
            elif frame_count % args.skip_frames == 0:  # Regular rendering for other episodes or if not saving mp4
                env.render()
                time.sleep(args.sleep)

            if done:
                # Render final frame if in human mode or if saving mp4 and it wasn't skipped
                if render_mode == "human":
                    env.render()
                elif is_last_episode and args.mp4:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                break

            logger.info(f"Episode {ep+1}, Frame {frame_count}: Action: {action}, Reward: {reward}")
        logger.info(f"Episode {ep+1}: Total Reward = {total_reward}")

        # Store frames from the last episode
        if is_last_episode and args.mp4:
            frames = episode_frames

        # Save trajectory for this episode
        all_trajectories.append(traj)

        if args.pause_frame and not (is_last_episode and args.mp4):
            input("Press Enter to continue...")

    # Save the MP4 file if requested
    if args.mp4 and frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp4_filename = f"replay_{timestamp}.mp4"
        logger.info(f"Saving last episode replay to {mp4_filename}...")
        
        try:
            # Use OpenCV for MP4 writing
            height, width, layers = frames[0].shape
            fps = int(1 / args.sleep) if args.sleep > 0 else 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(mp4_filename, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                
            video_writer.release()
            logger.info(f"MP4 saved successfully to {mp4_filename}")
        except Exception as e:
            logger.error(f"Error saving MP4: {e}")

    # --- Save trajectory if requested ---
    if args.savetraj:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_filename = f"trajectory_{timestamp}.py"
        logger.info(f"Saving trajectory to {traj_filename}...")
        with open(traj_filename, "w") as f:
            f.write("# Trajectories for all airplanes, each as a list of (x, y) tuples per episode\n")
            f.write("trajectories = ")
            f.write(repr(all_trajectories))
            f.write("\n")
        logger.info(f"Trajectory saved successfully to {traj_filename}")

    env.close()