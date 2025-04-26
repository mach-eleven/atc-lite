import os
import sys
import time
import argparse
from stable_baselines3 import PPO
from envs.atc.atc_gym import AtcGym
from envs.atc import scenarios
import envs.atc.model as model

os.environ['PYGLET_SHADOW_WINDOW'] = '0'
sys.path.append('.')

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SB3 PPO model zip file to run')
    parser.add_argument('--stage', type=int, default=None, help='Curriculum stage to use (1 = closest, ... N = farthest). If not set, will try all.')
    # --- NEW: Try to auto-detect stage from model filename if stage not given ---
    parser.add_argument('--entry', type=str, default=None, help='Override entry point as x,y (e.g. 5,5)')
    parser.add_argument('--heading', type=int, default=None, help='Override heading (default: from curriculum)')
    parser.add_argument('--skip-frames', type=int, default=100, help='Render every Nth frame (default: 1, i.e., no skipping)')
    args = parser.parse_args()

    outdir = 'sb3_logs'

    # Try to auto-detect stage from model filename if not given
    if args.stage is None:
        import re
        match = re.search(r'stage(\d+)_entry(\d+)_([\d]+)_hdg(\d+)', args.model)
        if match:
            args.stage = int(match.group(1))
            # print(f"[INFO] Auto-detected stage {args.stage} from model filename.")

    if args.stage is not None:
        stages = [args.stage - 1]
    else:
        stages = range(len(curriculum_entry_points))

    for stage in stages:
        # Allow override from command line
        if args.entry is not None:
            entry_xy = tuple(map(int, args.entry.split(',')))
        else:
            entry_xy = curriculum_entry_points[stage][0]
        if args.heading is not None:
            entry_heading = args.heading
        else:
            entry_heading = curriculum_entry_points[stage][1]
        stage_name = f"stage{stage+1}_entry{entry_xy[0]}_{entry_xy[1]}_hdg{entry_heading}"
        model_path = args.model
        # print(f"\n=== Visualizing {stage_name} with model {model_path} ===")
        env = AtcGym(
            airplane_count=1,
            sim_parameters=model.SimParameters(1.0, discrete_action_space=False, normalize_state=True),
            scenario=scenarios.SupaSupa(),
            render_mode='human'
        )
        model_ = PPO.load(model_path)
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

if __name__ == "__main__":
    main()
