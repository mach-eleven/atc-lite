# minimal-atc-rl/test_env.py
import sys
import time
import os

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

def main():
    print("Running realistic ATC simulation with wind and fuel effects...")
    
    # Create environment with continuous action space and slower simulation speed
    # Use a longer timestep to increase fuel consumption effects
    sim_params = model.SimParameters(2.0, discrete_action_space=False)
    
    # Override default fuel parameters in the model module to show more dramatic fuel effects
    # This is just for testing purposes
    original_init = model.Airplane.__init__
    
    def modified_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Modify fuel parameters for more dramatic effects
        self.fuel_mass = 2000    # Reduced fuel quantity
        self.max_fuel = 2000     # Reduced max fuel
        
        # Increased consumption rates
        self.cruise_consumption = 2.0    # Higher base fuel flow (4x normal)
        
    # Apply the modified initialization
    model.Airplane.__init__ = modified_init
    
    # Create environment with 3 aircraft for different scenarios
    env = AtcGym(airplane_count=3, sim_parameters=sim_params)
    
    # Reset the environment
    state, info = env.reset()
    
    # Get airport runway position for approach planning
    runway_x = env._runway.x
    runway_y = env._runway.y
    runway_phi = env._runway.phi_from_runway
    
    # Calculate final approach fix position (7nm from runway on approach path)
    faf_x = runway_x + 7 * math.sin(math.radians(runway_phi)) 
    faf_y = runway_y + 7 * math.cos(math.radians(runway_phi))
    
    # Run one longer episode
    num_episodes = 1
    max_steps_per_episode = 600  # Much longer episode to see fuel effects
    
    for episode in range(num_episodes):
        print(f"\n*** REALISTIC SCENARIO SIMULATION ***")
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
        for step in range(max_steps_per_episode):
            # Get the pre-planned action for this step
            action = planned_actions[step]
            
            # Step the environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the environment with the correct mode
            env.render(mode='human')
            time.sleep(0.05)  # Slightly faster for longer simulation
            
            if step % 30 == 0 or step < 5:
                print(f"\n--- Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f} ---")
                for i, airplane in enumerate(env._airplanes):
                    print(f"Aircraft {airplane.name}:")
                    print(f"  Position: ({airplane.x:.1f}, {airplane.y:.1f}), Alt: {airplane.h:.0f} ft")
                    print(f"  Airspeed: {airplane.v:.1f} kts, Ground Speed: {airplane.ground_speed:.1f} kts")
                    print(f"  Heading: {airplane.phi:.0f}°, Track: {airplane.track:.0f}°")
                    print(f"  Fuel: {airplane.fuel_remaining_pct:.1f}%, Wind: {airplane.wind_x:.1f}/{airplane.wind_y:.1f} kts")
            
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