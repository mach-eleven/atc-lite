#!/usr/bin/env python3
# Test script specifically focused on reward shaping for a single aircraft
import sys
import time
import os
import argparse
import numpy as np
import math
import random

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'

# Add the parent directory to sys.path
sys.path.append('.')

# Import dependencies
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
from envs.atc.scenarios import SimpleScenario, LOWW
import envs  # Required to register the environment

def parse_args():
    parser = argparse.ArgumentParser(description='Test enhanced reward functions with a single aircraft')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no visualization)')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--wind-scale', type=float, default=5.0, help='Wind scale factor (0 = no wind)')
    parser.add_argument('--scenario', type=str, choices=['simple', 'loww'], default='simple', 
                        help='Scenario to use (simple or loww)')
    parser.add_argument('--autopilot', action='store_true', default=True, 
                        help='Enable autopilot crab angle correction')
    return parser.parse_args()

def custom_scenario(scenario_name, wind_scale):
    """Create a custom scenario with specified wind scale"""
    if scenario_name.lower() == 'loww':
        class CustomLOWW(LOWW):
            def __init__(self):
                super().__init__(random_entrypoints=False)
                # Set up airspace with custom wind scale
                bbox = self.airspace.get_bounding_box()
                self.wind = model.Wind(
                    (math.ceil(bbox[0]), math.ceil(bbox[2]), 
                     math.ceil(bbox[1]), math.ceil(bbox[3])),
                    swirl_scale=wind_scale
                )
        return CustomLOWW()
    else:  # Simple scenario is default
        class CustomSimple(SimpleScenario):
            def __init__(self):
                super().__init__(random_entrypoints=False)
                # Set up airspace with custom wind scale
                bbox = self.airspace.get_bounding_box()
                self.wind = model.Wind(
                    (math.ceil(bbox[0]), math.ceil(bbox[2]), 
                     math.ceil(bbox[1]), math.ceil(bbox[3])),
                    swirl_scale=wind_scale
                )
        return CustomSimple()

def generate_runway_approach_sequence(steps, runway_x, runway_y, runway_phi, 
                                      airplane_x, airplane_y, airplane_h):
    """
    Generate a sequence of actions to guide the aircraft toward the runway.
    This function creates a more direct approach than the S-pattern approach.
    
    Args:
        steps: Number of steps in the sequence
        runway_x, runway_y: Runway coordinates
        runway_phi: Runway heading (from runway)
        airplane_x, airplane_y: Aircraft starting position
        airplane_h: Aircraft starting altitude
        
    Returns:
        List of actions [speed, altitude, heading] normalized to [-1, 1]
    """
    actions = []
    
    # Calculate final approach fix coordinates (typically ~7nm from runway threshold along approach path)
    runway_phi_rad = math.radians(runway_phi)
    faf_distance = 7  # nautical miles from runway threshold
    faf_x = runway_x + faf_distance * math.sin(runway_phi_rad)
    faf_y = runway_y + faf_distance * math.cos(runway_phi_rad)
    
    # Calculate approach path course (landing direction = runway_phi + 180)
    approach_course = (runway_phi + 180) % 360
    
    # Initial vector from aircraft to FAF
    dx_initial = faf_x - airplane_x
    dy_initial = faf_y - airplane_y
    initial_distance = math.sqrt(dx_initial**2 + dy_initial**2)
    
    # Calculate initial bearing to FAF
    initial_bearing = math.degrees(math.atan2(dx_initial, dy_initial)) % 360
    
    # Calculate if aircraft is on left or right side of approach course
    # We'll use this to determine if we need to fly a left or right pattern
    approach_course_rad = math.radians(approach_course)
    runway_to_aircraft_x = airplane_x - runway_x
    runway_to_aircraft_y = airplane_y - runway_y
    
    # This cross product determines which side of the approach course the aircraft is on
    side_indicator = (runway_to_aircraft_x * math.cos(approach_course_rad) - 
                     runway_to_aircraft_y * math.sin(approach_course_rad))
    
    # Generate the sequence of actions
    for step in range(steps):
        progress = step / steps  # 0 to 1 progress through the sequence
        
        # Speed profile: start fast, gradually slow down
        if progress < 0.7:
            # Maintain higher speed during initial approach
            speed = 250
        else:
            # Slow down during final approach
            slow_down_progress = (progress - 0.7) / 0.3  # 0 to 1 during final 30% of approach
            speed = 250 - slow_down_progress * (250 - 180)  # Gradually slow to 180 knots
        
        # Altitude profile: gradually descend to approach altitude
        if progress < 0.6:
            # Maintain higher altitude during initial part
            target_alt = airplane_h
        else:
            # Calculate distance from FAF at this stage of the approach
            descent_progress = (progress - 0.6) / 0.4  # 0 to 1 during final 40% of approach
            descent_distance = initial_distance * (1 - descent_progress)  # How far from FAF we are
            
            # Standard 3-degree glideslope = ~318 feet per nautical mile
            # FAF is typically at ~2000-3000 feet above runway elevation
            faf_alt = 3000  # feet above runway
            target_alt = max(faf_alt + (descent_distance * 318), faf_alt)
        
        # Heading profile: 
        # Initially fly toward FAF, then transition to align with runway
        if progress < 0.8:
            # Initial vector to intercept
            heading = initial_bearing
        else:
            # Final phase - line up with runway
            lineup_progress = (progress - 0.8) / 0.2  # 0 to 1 during final 20%
            # Smoothly transition from intercept heading to approach course
            heading = initial_bearing + lineup_progress * ((approach_course - initial_bearing) % 360)
            if abs(heading - approach_course) > 180:
                # Handle the case where we're crossing the 0/360 boundary
                if heading > approach_course:
                    heading -= 360
                else:
                    heading += 360
                    
        # Normalize actions to [-1, 1] range for the environment
        norm_speed = (speed - 100) / 200 - 1  # 100-300 knots -> [-1, 1]
        norm_alt = (target_alt / 38000) * 2 - 1  # 0-38000 feet -> [-1, 1]
        norm_heading = (heading / 180) - 1  # 0-360 degrees -> [-1, 1]
        
        actions.append([norm_speed, norm_alt, norm_heading])
    
    return actions

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("Testing enhanced reward functions with single aircraft scenario")
    print(f"Wind scale: {args.wind_scale}, Scenario: {args.scenario}")
    
    # Determine render mode
    render_mode = 'headless' if args.headless else 'human'
    
    # Create environment with continuous action space
    sim_params = model.SimParameters(1.0, discrete_action_space=False)
    sim_params.reward_shaping = True  # Enable reward shaping
    
    # Set up scenario
    scenario = custom_scenario(args.scenario, args.wind_scale)
    
    # Create environment with a single aircraft
    env = AtcGym(
        airplane_count=1,
        sim_parameters=sim_params,
        scenario=scenario,
        render_mode=render_mode
    )
    
    # Get runway information
    runway_x = env._runway.x
    runway_y = env._runway.y
    runway_phi = env._runway.phi_from_runway
    
    print(f"Runway position: ({runway_x}, {runway_y}), Heading: {runway_phi}°")
    print(f"Landing direction: {(runway_phi + 180) % 360}°")
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"\n========== EPISODE {episode+1}/{args.episodes} ==========")
        
        # Reset the environment
        state, info = env.reset()
        
        # Get the position of our single aircraft
        airplane = env._airplanes[0]
        airplane_x = airplane.x
        airplane_y = airplane.y
        airplane_h = airplane.h
        
        print(f"Aircraft starting position: ({airplane_x:.1f}, {airplane_y:.1f}, {airplane_h:.0f})")
        
        # Generate action sequence for runway approach
        actions = generate_runway_approach_sequence(
            args.steps, runway_x, runway_y, runway_phi,
            airplane_x, airplane_y, airplane_h
        )
        
        # Track reward components
        episode_reward = 0
        reward_components = {
            "approach_position": 0,
            "approach_angle": 0,
            "glideslope": 0
        }
        
        # Run the episode
        for step in range(min(len(actions), args.steps)):
            # Add small random perturbations to make it more realistic
            action = actions[step] + np.random.uniform(-0.05, 0.05, size=3)
            
            # Execute the action
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Extract reward components if available
            if "reward_components" in info:
                components = info["reward_components"]
                reward_components["approach_position"] += components["approach_position_rewards"]
                reward_components["approach_angle"] += components["approach_angle_rewards"]
                reward_components["glideslope"] += components["glideslope_rewards"]
            
            # Render if not in headless mode
            if not args.headless:
                env.render()
                time.sleep(0.02)  # Slight delay to make visualization easier to follow
            
            # Break if episode is done
            if done:
                break
        
        # Episode summary
        print(f"\nEpisode {episode+1} Summary:")
        print(f"Steps completed: {step+1}/{args.steps}")
        print(f"Total reward: {episode_reward:.2f}")
        print("Reward components:")
        for component, value in reward_components.items():
            print(f"  - {component}: {value:.2f}")
        
        # Status message based on outcome
        if done and reward > 1000:
            print("Episode outcome: SUCCESS - Aircraft reached final approach!")
        elif done:
            print("Episode outcome: FAILED - Aircraft didn't complete approach")
        else:
            print("Episode outcome: INCOMPLETE - Ran out of steps")
            
        # Wait a bit between episodes
        time.sleep(1.0)
    
    # Close the environment
    env.close()
    print("\nTest completed.")

if __name__ == "__main__":
    main()