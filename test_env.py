# minimal-atc-rl/test_env.py
import sys
import time
import os
import argparse

from envs.atc.scenarios import LOWW, SupaSupa, SuperSimple

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'

# Add the parent directory to sys.path
sys.path.append('.')

# Import dependencies
import gymnasium as gym
import numpy as np
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
import math
import envs  # Required to register the environment

def generate_approach_actions(steps, target_x, target_y, airplane_x, airplane_y, 
                            target_altitude=3000, approach_speed=180):
    """
    Generate a sequence of actions to guide an aircraft toward a landing approach
    
    Args:
        steps: Number of steps to generate actions for
        target_x, target_y: Coordinates of approach point
        airplane_x, airplane_y: Current aircraft position
        target_altitude: Target altitude for approach in feet
        approach_speed: Target approach speed in knots
        
    Returns:
        Array of actions (speed, altitude, heading)
    """
    actions = []
    
    # Calculate bearing to target
    dx = target_x - airplane_x
    dy = target_y - airplane_y
    target_heading = math.degrees(math.atan2(dx, dy)) % 360
    
    # Calculate distance to target
    distance = math.sqrt(dx**2 + dy**2)
    
    # Generate a sequence of gradually decreasing altitude and speed actions
    for i in range(steps):
        progress = i / steps  # 0 to 1 
        
        # Gradually decrease speed - start reducing when halfway there
        if progress > 0.5:
            speed_factor = 1.0 - (progress - 0.5) * 2  # 1.0 to 0.0
            speed = approach_speed + (250 - approach_speed) * speed_factor
        else:
            speed = 250  # Maintain cruise speed for first half
        
        # Gradually decrease altitude - constant descent rate
        alt = target_altitude + (17000 - target_altitude) * (1 - progress)
        
        # Adjust heading to intercept final approach course
        # Add some realistic S-turns by oscillating heading slightly
        heading_offset = math.sin(progress * 6) * 10 if progress < 0.8 else 0
        heading = target_heading + heading_offset
        
        # Normalize action values to [-1, 1] range for environment
        norm_speed = (speed - 100) / 200 - 1  # 100-300 knots -> [-1, 1]
        norm_alt = (alt / 38000) * 2 - 1      # 0-38000 feet -> [-1, 1]
        norm_heading = (heading / 180) - 1    # 0-360 degrees -> [-1, 1]
        
        actions.append([norm_speed, norm_alt, norm_heading])
    
    return actions

def generate_holding_pattern(steps, center_x, center_y, radius=2, altitude=15000):
    """
    Generate a sequence of actions for an aircraft to fly in a holding pattern
    
    Args:
        steps: Number of steps to generate actions for
        center_x, center_y: Center coordinates of holding pattern
        radius: Radius of holding pattern in nautical miles
        altitude: Target altitude in feet
        
    Returns:
        Array of actions (speed, altitude, heading)
    """
    actions = []
    
    # Speed in holding pattern
    speed = 210  # Slightly reduced speed for holding
    
    # Normalized parameters
    norm_speed = (speed - 100) / 200 - 1
    norm_alt = (altitude / 38000) * 2 - 1
    
    # Generate circular path by varying heading
    for i in range(steps):
        # Calculate angle in the circle (0 to 2π)
        angle = (i / steps) * 2 * math.pi
        
        # Determine target heading (90 degrees offset from angle)
        heading = (math.degrees(angle) + 90) % 360
        norm_heading = (heading / 180) - 1
        
        actions.append([norm_speed, norm_alt, norm_heading])
    
    return actions

def parse_args():
    parser = argparse.ArgumentParser(description='Run realistic ATC simulation with wind and fuel effects')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no visualization)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=600, help='Maximum steps per episode')
    parser.add_argument('--render-interval', type=int, default=1, 
                       help='Render every N steps (only applies in non-headless mode)')
    parser.add_argument('--wind-scale', type=float, default=2.0,
                       help='Wind scale factor (higher = stronger winds)')
    parser.add_argument('--wind-badness', type=int, default=5, choices=range(0, 11),
                       help='How strong and turbulent the wind should be (0-10)')
    parser.add_argument('--autopilot', action='store_true', default=True,
                       help='Enable autopilot heading correction')
    parser.add_argument('--no-autopilot', action='store_false', dest='autopilot',
                       help='Disable autopilot heading correction')
    parser.add_argument('--curr-stages', type=int, default=100,
                          help='Number of stages for curriculum training')
    parser.add_argument('--curr-stage-entry-point', type=int, default=15,
                          help='Curriculum stage to use (1 = closest, ... N = farthest)')
    parser.add_argument('--pause-frame', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Running realistic ATC simulation with wind and fuel effects...")
    print(f"Wind scale: {args.wind_scale}, Autopilot: {'Enabled' if args.autopilot else 'Disabled'}")
    
    # Determine render mode based on arguments
    render_mode = 'headless' if args.headless else 'human'
    print(f"Render mode: {render_mode}")
    
    # Create environment with continuous action space and slower simulation speed
    # Use a longer timestep to increase fuel consumption effects
    sim_params = model.SimParameters(2.0, discrete_action_space=False)
    
    # Store the original Airplane init method
    original_init = model.Airplane.__init__
    
    # Global variable for autopilot setting
    autopilot_enabled = args.autopilot
    
    def modified_init(self, sim_parameters, name, x, y, h, phi, v, h_min=0, h_max=38000, v_min=100, v_max=300):
        # Call the original init with all its original parameters
        original_init(self, sim_parameters, name, x, y, h, phi, v, h_min, h_max, v_min, v_max)
        
        # Modify fuel parameters for more dramatic effects
        self.fuel_mass = 2000    # Reduced fuel quantity
        self.max_fuel = 2000     # Reduced max fuel
        
        # Increased consumption rates
        self.cruise_consumption = 2.0    # Higher base fuel flow (4x normal)
        
        # Set autopilot based on global variable
        self.autopilot_enabled = autopilot_enabled
        
    # Apply the modified initialization
    model.Airplane.__init__ = modified_init
    
    # Create custom wind scale for the scenario
    # class CustomLOWW(LOWW):
    #     def __init__(self, random_entrypoints=True, wind_scale=5.0):
    #         super().__init__(random_entrypoints)
    #         # Override the wind with custom scale
    #         minx, miny, maxx, maxy = self.airspace.get_bounding_box()
    #         self.wind = model.Wind(
    #             (math.ceil(minx), math.ceil(maxx), math.ceil(miny), math.ceil(maxy)),
    #             swirl_scale=wind_scale
    #         )
    
    # Create environment with 3 aircraft for different scenarios with the specified render mode
    
    curr_entry_points = SupaSupa().generate_curriculum_entrypoints(num_entrypoints=args.curr_stages)
    print(curr_entry_points)
    env = AtcGym(
        airplane_count=1, 
        sim_parameters=sim_params, 
        scenario=SupaSupa(curr_entry_points[args.curr_stage_entry_point - 1]),
        render_mode=render_mode,
        wind_badness=args.wind_badness
    )

    # Reset the environment
    state, info = env.reset()
    
    # Get airport runway position for approach planning
    runway_x = env._runway.x
    runway_y = env._runway.y
    runway_phi = env._runway.phi_from_runway
    
    # Calculate final approach fix position (7nm from runway on approach path)
    faf_x = runway_x + 7 * math.sin(math.radians(runway_phi)) 
    faf_y = runway_y + 7 * math.cos(math.radians(runway_phi))
    
    # Run episodes
    num_episodes = args.episodes
    max_steps_per_episode = args.steps
    
    for episode in range(num_episodes):
        print(f"\n*** REALISTIC SCENARIO SIMULATION - EPISODE {episode+1}/{num_episodes} ***")
        state, info = env.reset()

        airplane_count = len(env._airplanes)
        
        # Get initial positions of each aircraft
        airplane_positions = [(airplane.x, airplane.y) for airplane in env._airplanes]
        
        # Create action sequences for each aircraft:
        # 1. First aircraft - landing approach
        approach_actions = generate_approach_actions(
            max_steps_per_episode,
            faf_x, faf_y,
            airplane_positions[0][0], airplane_positions[0][1]
        )
        
        # 2. Second aircraft - holding pattern
        holding_center_x = runway_x + 10  # 10nm east of runway
        holding_center_y = runway_y + 5   # 5nm north of runway
        holding_actions = generate_holding_pattern(
            max_steps_per_episode,
            holding_center_x, holding_center_y
        )
        
        # 3. Third aircraft - high-speed cruise (fuel intensive)
        cruise_actions = [[0.9, 0.3, 0.0]] * max_steps_per_episode  # High speed, medium altitude, constant heading
        
        # Combine the action sequences for all aircraft
        planned_actions = []
        for i in range(max_steps_per_episode):
            if i < len(approach_actions) and i < len(holding_actions):
                # Combine actions for all three aircraft
                action = np.array([
                    *approach_actions[i],  # Aircraft 1 - approach
                    *holding_actions[i],   # Aircraft 2 - holding 
                    *cruise_actions[i]     # Aircraft 3 - high-speed cruise
                ])
                planned_actions.append(action)
            else:
                # Default action if we run out of planned actions
                action = np.array([0.0, 0.0, 0.0] * airplane_count)
                planned_actions.append(action)
        
        total_reward = 0
        # Track which aircraft are still active (not done)
        active_aircraft = [True] * airplane_count
        for step in range(max_steps_per_episode):
            # Use planned actions instead of random actions
            # This gives more realistic flight paths
            action = planned_actions[step].copy()

            # Add small random perturbations to make it more realistic
            action += np.random.uniform(-0.05, 0.05, size=action.shape)

            # Mask actions for aircraft that are done (set to zeros)
            for i in range(airplane_count):
                if not active_aircraft[i]:
                    action[i*3:(i+1)*3] = 0.0

            # Step the environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Only render in non-headless mode, and only at specified intervals
            if not args.headless and step % args.render_interval == 0:
                env.render()
                time.sleep(0.05)  # Slightly faster for longer simulation
            
            if args.pause_frame:
                print("Pausing simulation for observation...")
                # Pause the simulation for a short time to allow for observation
                time.sleep(10000000)


            if step % 30 == 0 or step < 5:
                print(f"\n--- Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f} ---")
                for i, airplane in enumerate(env._airplanes):
                    crab_angle = model.relative_angle(airplane.phi, airplane.track)
                    print(f"{airplane.name}:\t\tPosition: ({airplane.x:.1f}, {airplane.y:.1f}) Altitude: {airplane.h:.0f} ft")
                    print(f"\t\tAirspeed: {airplane.v:.1f} kts, Groundspeed: {airplane.ground_speed:.1f} kts")
                    print(f"\t\tHeading: {airplane.phi:.0f}°, Track: {airplane.track:.0f}° (Crab: {crab_angle:.1f}°)")
                    print(f"\t\tFuel: {airplane.fuel_remaining_pct:.1f}%, Wind: ({airplane.wind_x:.1f}, {airplane.wind_y:.1f}) kts\n")

            # Check which aircraft are done (landed, crashed, or out of fuel)
            # The dones array is tracked internally in the environment, but only the global done is returned.
            # We can infer per-aircraft done by checking if the aircraft is in the approach corridor or out of fuel/airspace.
            # But the environment does not return per-aircraft dones, so we must keep masking actions for landed aircraft.
            # This is handled above by setting their actions to zero after they land.

            if done:
                if reward > 1000:
                    print("Scenario completed successfully!")
                else:
                    print(f"Scenario ended with reward: {reward:.2f}")
                break
        
        # Small delay between episodes
        time.sleep(1.0)
    
    env.close()
    print("Testing complete")

if __name__ == "__main__":
    main()