#!/usr/bin/env python3
# trajectory_tester.py - Script to visualize and verify aircraft trajectories and physics
import sys
import time
import os
import math
import argparse
import numpy as np
from typing import List, Tuple

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'

# Add the parent directory to sys.path
sys.path.append('.')

# Import dependencies
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
from envs.atc.scenarios import LOWW

# Test patterns for aircraft movement
def generate_circle_pattern(steps: int, center_x: float, center_y: float, 
                           radius: float = 5.0, altitude: float = 10000, 
                           speed: float = 250) -> List[np.ndarray]:
    """
    Generate actions for circular flight pattern.
    
    Args:
        steps: Number of steps to generate
        center_x, center_y: Center of the circle
        radius: Radius of the circle in nautical miles
        altitude: Constant altitude in feet
        speed: Constant speed in knots
        
    Returns:
        List of action arrays (speed, altitude, heading)
    """
    actions = []
    
    # Normalized parameters for action space (-1 to 1)
    norm_speed = (speed - 100) / 200 - 1  # 100-300 knots -> [-1, 1]
    norm_alt = (altitude / 38000) * 2 - 1  # 0-38000 feet -> [-1, 1]
    
    # Calculate heading for each point in the circle
    for i in range(steps):
        # Calculate the target position along the circle
        angle = (i / steps) * 2 * math.pi
        target_x = center_x + radius * math.cos(angle)
        target_y = center_y + radius * math.sin(angle)
        
        # Calculate heading to stay tangent to the circle (90 degrees offset from radial)
        heading = (math.degrees(angle) + 90) % 360
        norm_heading = (heading / 180) - 1  # 0-360 degrees -> [-1, 1]
        
        actions.append(np.array([norm_speed, norm_alt, norm_heading]))
    
    return actions

def generate_figure_eight(steps: int, center_x: float, center_y: float,
                         radius: float = 5.0, altitude: float = 10000,
                         speed: float = 250) -> List[np.ndarray]:
    """
    Generate actions for figure-eight flight pattern.
    
    Args:
        steps: Number of steps to generate
        center_x, center_y: Center of the figure eight
        radius: Radius of each circle in nautical miles
        altitude: Constant altitude in feet
        speed: Constant speed in knots
        
    Returns:
        List of action arrays (speed, altitude, heading)
    """
    actions = []
    
    # Normalized parameters
    norm_speed = (speed - 100) / 200 - 1
    norm_alt = (altitude / 38000) * 2 - 1
    
    # Calculate heading for each point in the figure eight
    for i in range(steps):
        t = (i / steps) * 2 * math.pi
        
        # Parametric equation for figure eight using lemniscate
        x = center_x + radius * math.sin(t) / (1 + math.cos(t)**2)
        y = center_y + radius * math.sin(t) * math.cos(t) / (1 + math.cos(t)**2)
        
        # Calculate heading (this gets complex - we use the derivative of the parametric equation)
        if i > 0:
            prev_x = actions[-1][3]  # Store x, y in actions for heading calculation
            prev_y = actions[-1][4]
            dx = x - prev_x
            dy = y - prev_y
            heading = math.degrees(math.atan2(dx, dy)) % 360
        else:
            heading = 0  # Initial heading
            
        norm_heading = (heading / 180) - 1
        
        # Store x, y for heading calculation in the next iteration
        action = np.array([norm_speed, norm_alt, norm_heading, x, y])
        actions.append(action)
    
    # Remove the x, y coordinates from the actions before returning
    return [act[:3] for act in actions]

def generate_s_turn_approach(steps: int, start_x: float, start_y: float, 
                            target_x: float, target_y: float,
                            start_altitude: float = 15000, end_altitude: float = 3000,
                            start_speed: float = 280, end_speed: float = 180) -> List[np.ndarray]:
    """
    Generate S-turn approach pattern with descending altitude and decreasing speed.
    
    Args:
        steps: Number of steps to generate
        start_x, start_y: Starting position
        target_x, target_y: Target position (e.g., final approach fix)
        start_altitude: Starting altitude in feet
        end_altitude: Final altitude in feet
        start_speed: Starting speed in knots
        end_speed: Final speed in knots
        
    Returns:
        List of action arrays (speed, altitude, heading)
    """
    actions = []
    
    # Calculate direct heading to target
    dx = target_x - start_x
    dy = target_y - start_y
    direct_heading = math.degrees(math.atan2(dx, dy)) % 360
    distance = math.sqrt(dx**2 + dy**2)
    
    for i in range(steps):
        # Calculate progress (0 to 1)
        progress = i / steps
        
        # Calculate current speed and altitude (gradual reduction)
        speed = start_speed - progress * (start_speed - end_speed)
        altitude = start_altitude - progress * (start_altitude - end_altitude)
        
        # Add S-curve to heading (stronger in the middle)
        heading_offset = math.sin(progress * 2 * math.pi) * 20  # +/- 20 degrees
        heading = (direct_heading + heading_offset) % 360
        
        # Normalize values for action space
        norm_speed = (speed - 100) / 200 - 1
        norm_alt = (altitude / 38000) * 2 - 1
        norm_heading = (heading / 180) - 1
        
        actions.append(np.array([norm_speed, norm_alt, norm_heading]))
    
    return actions

def generate_wind_test_pattern(steps: int, center_x: float, center_y: float,
                              altitude: float = 15000, speed: float = 250) -> List[np.ndarray]:
    """
    Generate a pattern to test wind effects on aircraft.
    The pattern holds a constant heading for a while, then rotates through all cardinal directions.
    
    Args:
        steps: Number of steps to generate
        center_x, center_y: Center position
        altitude: Constant altitude in feet
        speed: Constant speed in knots
        
    Returns:
        List of action arrays (speed, altitude, heading)
    """
    actions = []
    
    # Normalize constant parameters
    norm_speed = (speed - 100) / 200 - 1
    norm_alt = (altitude / 38000) * 2 - 1
    
    # First third: hold north heading
    first_segment = steps // 3
    north_heading = 0
    norm_north = (north_heading / 180) - 1
    
    for _ in range(first_segment):
        actions.append(np.array([norm_speed, norm_alt, norm_north]))
    
    # Second third: rotate through all headings
    second_segment = steps // 3
    for i in range(second_segment):
        heading = (i / second_segment) * 360  # Rotate through all headings
        norm_heading = (heading / 180) - 1
        actions.append(np.array([norm_speed, norm_alt, norm_heading]))
    
    # Final third: hold east heading
    third_segment = steps - first_segment - second_segment
    east_heading = 90
    norm_east = (east_heading / 180) - 1
    
    for _ in range(third_segment):
        actions.append(np.array([norm_speed, norm_alt, norm_east]))
    
    return actions

def combine_actions_for_multiple_aircraft(airplane_actions: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Combine actions for multiple aircraft into a single action array.
    
    Args:
        airplane_actions: List of action lists for each airplane
        
    Returns:
        Combined action list for all airplanes
    """
    min_steps = min(len(actions) for actions in airplane_actions)
    combined_actions = []
    
    for i in range(min_steps):
        # Concatenate actions for all airplanes at this step
        combined_action = np.concatenate([actions[i] for actions in airplane_actions])
        combined_actions.append(combined_action)
    
    return combined_actions

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Aircraft trajectory visualization and physics testing')
    parser.add_argument('--pattern', type=str, default='all',
                        choices=['circle', 'figure8', 'approach', 'wind', 'all'],
                        help='Flight pattern to test')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of steps to simulate')
    parser.add_argument('--wind-scale', type=float, default=10.0,
                        help='Wind scale factor (higher = stronger winds)')
    parser.add_argument('--autopilot', action='store_true', default=True,
                        help='Enable autopilot heading correction')
    parser.add_argument('--no-autopilot', action='store_false', dest='autopilot',
                        help='Disable autopilot heading correction')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between steps (seconds)')
    
    return parser.parse_args()

def print_pattern_info(pattern_name: str):
    """Print information about the current test pattern"""
    pattern_width = 40
    print("\n" + "=" * pattern_width)
    print(f"{pattern_name.upper()} TEST PATTERN".center(pattern_width))
    print("=" * pattern_width)
    
def print_aircraft_stats(env: AtcGym):
    """Print detailed statistics about each aircraft"""
    print("\nAircraft Status:")
    print("-" * 80)
    
    for i, airplane in enumerate(env._airplanes):
        # Calculate crab angle (difference between heading and track)
        crab_angle = model.relative_angle(airplane.phi, airplane.track)
        wind_speed = math.sqrt(airplane.wind_x**2 + airplane.wind_y**2)
        wind_dir = math.degrees(math.atan2(-airplane.wind_x, -airplane.wind_y)) % 360
        
        print(f"Aircraft {i+1} ({airplane.name}):")
        print(f"  Position: ({airplane.x:.1f}, {airplane.y:.1f}) Altitude: {airplane.h:.0f} ft")
        print(f"  Airspeed: {airplane.v:.1f} kts | Ground speed: {airplane.ground_speed:.1f} kts")
        print(f"  Heading: {airplane.phi:.1f}° | Ground Track: {airplane.track:.1f}° | Crab: {crab_angle:.1f}°")
        print(f"  Wind: {wind_speed:.1f} kts from {wind_dir:.0f}°, Components: ({airplane.wind_x:.1f}, {airplane.wind_y:.1f})")
        print(f"  Fuel: {airplane.fuel_remaining_pct:.1f}%")
        print("-" * 80)

def run_trajectory_test(args):
    """Run trajectory visualization test with the specified parameters"""
    print(f"Running aircraft trajectory tests with:")
    print(f"- Wind scale: {args.wind_scale}")
    print(f"- Autopilot: {'Enabled' if args.autopilot else 'Disabled'}")
    
    # Create simulation environment with slower timestep for better visualization
    sim_params = model.SimParameters(2.0, discrete_action_space=False)
    
    # Store the original Airplane init method
    original_init = model.Airplane.__init__
    
    # Define modified aircraft initialization with reduced fuel for shorter tests
    def modified_init(self, sim_parameters, name, x, y, h, phi, v, h_min=0, h_max=38000, v_min=100, v_max=300):
        # Call the original init with all its original parameters
        original_init(self, sim_parameters, name, x, y, h, phi, v, h_min, h_max, v_min, v_max)
        
        # Set autopilot based on command line argument
        self.autopilot_enabled = args.autopilot
        
        # Increase position history capacity for better visualization
        self.position_history = []  # Reset to make sure we have a clean history
    
    # Apply the modified initialization
    model.Airplane.__init__ = modified_init
    
    # Create custom wind scenario with specified wind scale
    class CustomWindScenario(LOWW):
        def __init__(self, random_entrypoints=False, wind_scale=10.0):
            super().__init__(random_entrypoints)
            # Override wind with custom scale
            minx, miny, maxx, maxy = self.airspace.get_bounding_box()
            self.wind = model.Wind(
                (math.ceil(minx), math.ceil(maxx), math.ceil(miny), math.ceil(maxy)),
                swirl_scale=wind_scale
            )
    
    # Create environment with 3 aircraft and the custom scenario
    env = AtcGym(
        airplane_count=3,
        sim_parameters=sim_params,
        scenario=CustomWindScenario(random_entrypoints=True, wind_scale=args.wind_scale),
        render_mode='human'
    )
    
    # Enable trajectory visualization
    env.show_trajectories = True
    print("Trajectory visualization enabled")
    
    # Get airspace boundaries and runway position
    runway_x = env._runway.x
    runway_y = env._runway.y
    runway_phi = env._runway.phi_from_runway
    
    # Calculate FAF position (7nm from runway on approach path)
    faf_x = runway_x + 7 * math.sin(math.radians(runway_phi))
    faf_y = runway_y + 7 * math.cos(math.radians(runway_phi))
    
    # Center of airspace for pattern generation
    center_x = (env._world_x_min + env._world_x_max) / 2
    center_y = (env._world_y_min + env._world_y_max) / 2
    
    # Run selected pattern tests
    patterns_to_test = []
    
    if args.pattern == 'circle' or args.pattern == 'all':
        patterns_to_test.append('circle')
    if args.pattern == 'figure8' or args.pattern == 'all':
        patterns_to_test.append('figure8')
    if args.pattern == 'approach' or args.pattern == 'all':
        patterns_to_test.append('approach')
    if args.pattern == 'wind' or args.pattern == 'all':
        patterns_to_test.append('wind')
    
    # Run each selected test pattern
    for pattern in patterns_to_test:
        # Reset environment for each test
        state, info = env.reset()
        env.show_trajectories = True  # Make sure trajectories are enabled
        
        # Get initial positions for each aircraft
        airplane_positions = [(airplane.x, airplane.y) for airplane in env._airplanes]
        
        if pattern == 'circle':
            print_pattern_info("Circle")
            print("Testing circular flight pattern - verifying constant radius turns")
            
            # Generate different circles for each aircraft
            actions_1 = generate_circle_pattern(args.steps, center_x - 10, center_y, 8, 15000, 250)
            actions_2 = generate_circle_pattern(args.steps, center_x + 10, center_y, 5, 10000, 220)
            actions_3 = generate_circle_pattern(args.steps, center_x, center_y + 15, 3, 5000, 180)
            
            # Combine actions
            combined_actions = combine_actions_for_multiple_aircraft([actions_1, actions_2, actions_3])
            
        elif pattern == 'figure8':
            print_pattern_info("Figure Eight")
            print("Testing figure-eight pattern - verifying complex maneuvers")
            
            # Generate figure-eight patterns for each aircraft
            actions_1 = generate_figure_eight(args.steps, center_x - 15, center_y, 8, 18000, 270)
            actions_2 = generate_figure_eight(args.steps, center_x, center_y, 6, 12000, 240)
            actions_3 = generate_figure_eight(args.steps, center_x + 15, center_y, 4, 6000, 210)
            
            combined_actions = combine_actions_for_multiple_aircraft([actions_1, actions_2, actions_3])
            
        elif pattern == 'approach':
            print_pattern_info("Approach with S-Turns")
            print("Testing approach pattern with S-turns - verifying descent and deceleration")
            
            # Generate approach patterns from different starting points
            actions_1 = generate_s_turn_approach(
                args.steps,
                airplane_positions[0][0], airplane_positions[0][1],
                faf_x, faf_y, 15000, 3000, 280, 160
            )
            actions_2 = generate_s_turn_approach(
                args.steps,
                airplane_positions[1][0], airplane_positions[1][1],
                faf_x, faf_y, 18000, 3500, 300, 180
            )
            actions_3 = generate_s_turn_approach(
                args.steps,
                airplane_positions[2][0], airplane_positions[2][1],
                faf_x, faf_y, 12000, 4000, 250, 170
            )
            
            combined_actions = combine_actions_for_multiple_aircraft([actions_1, actions_2, actions_3])
            
        elif pattern == 'wind':
            print_pattern_info("Wind Test")
            print("Testing wind effects - verifying drift and crab angle")
            
            # Generate wind test patterns
            actions_1 = generate_wind_test_pattern(args.steps, center_x, center_y - 10, 5000, 200)
            actions_2 = generate_wind_test_pattern(args.steps, center_x - 5, center_y, 10000, 250)
            actions_3 = generate_wind_test_pattern(args.steps, center_x + 5, center_y + 10, 15000, 300)
            
            combined_actions = combine_actions_for_multiple_aircraft([actions_1, actions_2, actions_3])
        
        # Run the simulation with the generated actions
        for step in range(min(args.steps, len(combined_actions))):
            # Add small random perturbations for more realism
            action = combined_actions[step] + np.random.uniform(-0.03, 0.03, size=combined_actions[step].shape)
            
            # Apply action and get new state
            state, reward, done, truncated, info = env.step(action)
            
            # Render and add delay for visualization
            env.render()
            time.sleep(args.delay)
            
            # Print detailed stats periodically
            if step % 30 == 0:
                print(f"\nStep {step}/{args.steps}")
                print_aircraft_stats(env)
            
            if done:
                break
        
        # Pause at the end of each pattern to view final trajectory
        print("\nPattern test completed - press Enter to continue...")
        input()
    
    env.close()
    # Restore the original Airplane init method
    model.Airplane.__init__ = original_init
    print("\nTrajectory testing completed")

if __name__ == "__main__":
    args = parse_args()
    run_trajectory_test(args)