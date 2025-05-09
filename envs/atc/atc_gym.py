# minimal-atc-rl/envs/atc/atc_gym.py
import math
import random
import os
import logging
logger = logging.getLogger("train.atc_gym")
logger.setLevel(logging.INFO)

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from numba import jit
import pyglet

from .rendering import Label, FuelGauge
from .themes import ColorScheme
from . import model
from . import scenarios
from . import my_rendering as rendering

# Import headless rendering capabilities
from .headless_rendering import HeadlessViewer

# TODO: Airplane should have d_faf, etc. stored in it not the array system we have now

MAX_MVA_HEIGHT_FOR_VISUALIZATION = 10_000
# JIT-compiled helper function to calculate a sigmoid-based distance metric
# This creates a smooth transition between 0 and 1 as d approaches d_max
@jit(nopython=True)
def sigmoid_distance_func(d, d_max):
    """
    Provides a smooth normalization of distances using a tanh-based sigmoid
    
    Args:
        d: current distance
        d_max: maximum possible distance
        
    Returns:
        A value between 0 and 1, with smaller distances giving larger values
    """
    return (1.0 - math.tanh(4.0 * (d / d_max) - 2.0)) / 2.0


class AtcGym(gym.Env):
    """
    Air Traffic Control Gym Environment
    
    This class implements a reinforcement learning environment for training
    air traffic control agents to guide aircraft through airspace safely to a runway.
    """
    
    # Setting the metadata for rendering modes and frame rate
    metadata = {
        'render_modes': ['human', 'rgb_array', 'headless'],
        "render_fps": 50
    }

    def __init__(self, airplane_count=1, sim_parameters=model.SimParameters(1), scenario=None, render_mode='rgb_array', wind_badness=5, wind_dirn=270, starting_fuel=10000, reduced_time_penalty=False):
        """
        Initialize the ATC gym environment
        
        Args:
            airplane_count: Number of airplanes to simulate
            sim_parameters: Simulation parameters like timestep size
            scenario: The airspace scenario to use (runways, MVAs, etc.)
            render_mode: 'human' for window rendering, 'rgb_array' for array output, 'headless' for no rendering
            wind_badness: How strong and turbulent the wind should be (0-10)
            wind_dirn: Wind direction in degrees
            starting_fuel: Amount of fuel in kg that aircraft start with
        """

        self._airplane_count = airplane_count
        self._wind_badness = wind_badness  # Store the wind badness parameter
        self._wind_dirn = wind_dirn
        self._starting_fuel = starting_fuel  # Store the starting fuel amount
        # print(f"Wind badness: {self._wind_badness} | Wind direction: {self._wind_dirn}")

        self.reduced_time_penalty = reduced_time_penalty
        # logger.info(f"Reduced time penalty: {self.reduced_time_penalty}")
        self.render_mode = render_mode
        
        # Use a default scenario if none provided
        if scenario is None:
            scenario = scenarios.SimpleScenario()
        
        # Reward tracking variables
        self.last_reward = 0
        self.total_reward = 0
        self.actions_taken = 0

        # Episode and performance tracking
        self._episodes_run = 0
        self._actions_ignoring_resets = 0
        self._won_simulations_ignoring_resets = 0
        self._win_buffer_size = 10
        self._win_buffer = [0] * self._win_buffer_size  # Tracks recent win/loss results
        self.actions_per_timestep = 0
        self.timesteps = 0
        self.timestep_limit = 6000  # Maximum timesteps before episode termination
        self.winning_ratio = 0

        # Store scenario parameters
        self._sim_parameters = sim_parameters
        self._scenario = scenario
        self._mvas = scenario.mvas  # Minimum vectoring altitudes
        self._runway = scenario.runway
        self._airspace = scenario.airspace
        # Get the MVA at the Final Approach Fix (FAF)
        self._faf_mva = self._airspace.get_mva_height(self._runway.corridor.faf[0][0], self._runway.corridor.faf[1][0])

        # Calculate world boundaries from the airspace
        bbox = self._airspace.get_bounding_box()
        self._world_x_min = bbox[0]
        self._world_y_min = bbox[1]
        self._world_x_max = bbox[2]
        self._world_y_max = bbox[3]
        world_x_length = self._world_x_max - self._world_x_min
        world_y_length = self._world_y_max - self._world_y_min
        self._world_max_distance = self.euclidean_dist(world_x_length, world_y_length)

        # Initialize environment state
        self.done = True

        # Use airplane_count for normalization arrays
        n = self._airplane_count
        # Use typical aircraft values for normalization
        v_min = 100
        v_max = 300
        h_max = 38000
        # For derived features, use n as count
        self.normalization_state_min = np.array([
            *[self._world_x_min for _ in range(n)],          # Minimum x position
            *[self._world_y_min for _ in range(n)],          # Minimum y position
            *[0 for _ in range(n)],                          # Minimum altitude
            *[0 for _ in range(n)],                          # Minimum heading
            *[v_min for _ in range(n)],                      # Minimum speed
            *[0 for _ in range(n)],                          # Minimum height above MVA
            *[0 for _ in range(n)],                          # Minimum on-glidepath altitude
            *[0 for _ in range(n)],                          # Minimum distance to FAF
            *[-180 for _ in range(n)],                       # Minimum relative angle to FAF
            *[-180 for _ in range(n)],                       # Minimum relative angle to runway
            *[0 for _ in range(n)]                           # Minimum fuel percentage
        ], dtype=np.float32)
        self.normalization_state_max = np.array([
            *[self._world_x_max - self._world_x_min for _ in range(n)],          # x position range
            *[self._world_y_max - self._world_y_min for _ in range(n)],          # y position range
            *[h_max for _ in range(n)],   # maximum altitude
            *[360 for _ in range(n)],     # heading range
            *[v_max for _ in range(n)],   # maximum speed (FIXED)
            *[h_max for _ in range(n)],                        # maximum height above MVA
            *[h_max for _ in range(n)],                        # maximum on glidepath altitude
            *[self._world_max_distance for _ in range(n)],     # maximum distance to FAF
            *[360 for _ in range(n)],                          # maximum relative angle to FAF
            *[360 for _ in range(n)],                          # maximum relative angle to runway
            *[100 for _ in range(n)]                           # maximum fuel percentage
        ], dtype=np.float32)
        
        self.reset()
        self.viewer = None

        # Action space normalization parameters
        self.normalization_action_offset = np.array([
            *[airplane.v_min for airplane in self._airplanes], 
            *[0 for _ in self._airplanes], 
            *[0 for _ in self._airplanes]
        ])

        # Configure action space (discrete or continuous)
        if sim_parameters.discrete_action_space:
            # For discrete actions, define the scaling factors
            self.normalization_action_factor = np.array([10, 100, 1]*len(self._airplanes))

            # action space structure: v (speed), h (altitude), phi (heading)
            self.action_space = gym.spaces.MultiDiscrete([
                *[int((airplane.v_max - airplane.v_min) / 10) for airplane in self._airplanes],  # Speed buckets
                *[int(airplane.h_max / 100) for airplane in self._airplanes],                          # Altitude in hundreds of feet
                *[360 for _ in self._airplanes]                                                       # Heading in degrees
            ])
        else:
            # For continuous actions, define the scaling factors based on aircraft limits

            self.normalization_action_factor = np.array([
                *[airplane.v_max - airplane.v_min for airplane in self._airplanes], # Speed range
                *[airplane.h_max for airplane in self._airplanes],                  # Altitude range
                *[360 for _ in self._airplanes]                                      # Heading range (degrees)
            ])

            # Continuous action space between -1 and 1 for each dimension
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1]*len(self._airplanes)),
                high=np.array([1, 1, 1]*len(self._airplanes))
            )

        # Thresholds for what counts as a significant action change
        self._action_discriminator = [5, 50, 0.5]  # For speed, altitude, heading
        self.last_action = [0, 0, 0]*len(self._airplanes)

        # Define the observation space: x, y, h, phi, v, h-mva, on_gp, d_faf, phi_rel_faf, phi_rel_runway, fuel
        # All values normalized to [-1, 1]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.normalization_state_min),))
        self.reward_range = (-3000.0, 23000.0)  # Define the min/max possible rewards

        # logger.info(f"Gymnasium ATC Environment initialized: Airplanes: {len(self._airplanes)}, obs: {len(self.normalization_state_min)}, act: {len(self.normalization_action_offset)}")
        
        # Initialize trajectory visualization settings
        self.show_trajectories = True  # Default: trajectories enabled
        self.trajectory_colors = [
            (255, 0, 0),    # Red
            (0, 200, 255),  # Cyan
            (255, 165, 0),  # Orange
            (255, 0, 255),  # Magenta
            (0, 255, 0)     # Green
        ]
        self.max_trail_length = 90000000  # Maximum number of history points to render
        
        # Load airplane images in different colors to match trajectory colors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Base airplane image path
        airplane_path = os.path.join(script_dir, 'assets', 'airplane_green.png')
        try:
            # Try to load the image directly
            self.airplane_image = pyglet.image.load(airplane_path)
            
            # Set the anchor point for proper rotation
            self.airplane_image.anchor_x = self.airplane_image.width // 2
            self.airplane_image.anchor_y = self.airplane_image.height // 2
            
            self.use_airplane_image = True
            logger.debug(f"Successfully loaded airplane image from {airplane_path}")
        except Exception as e:
            logger.debug(f"Could not load airplane image from any path. Error: {e}")
            self.use_airplane_image = False

    def seed(self, seed=None):
        """
        Seeds the environment's random number generators for reproducibility
        
        Args:
            seed: Random seed value
            
        Returns:
            List containing the seed
        """
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def step(self, action_array):
        """
        Execute one step in the environment
        
        Args:
            action_array: Action vector with [speed, altitude, heading]*airplane commands
            
        Returns:
            (state, reward, done, truncated, info) tuple
        """
        self.timesteps += 1
        self.done = False
        # Small negative reward per timestep to encourage efficiency
        if self.reduced_time_penalty:
            time_penalty = -0.05 * self._sim_parameters.timestep
        else:
            time_penalty = -0.5 * self._sim_parameters.timestep  # Scaled up for normalization
        reward = time_penalty
        
        # Dictionary to track reward components for logging
        reward_components = {
            "time_penalty": time_penalty,
            "action_rewards": 0,
            "fuel_penalties": 0,
            "mva_penalties": 0,
            "airspace_penalties": 0,
            "success_rewards": 0,
            "approach_position_rewards": 0,
            "approach_angle_rewards": 0,
            "fuel_efficiency_rewards": 0
        }

        dones = [False] * len(self._airplanes)
        out_of_fuel = [False] * len(self._airplanes)
        
        prev_d_fafs = list(self._d_fafs) if hasattr(self, '_d_fafs') else [None]*len(self._airplanes)
        
        # Apply each action component and accumulate rewards
        c = 0
        for airplane in self._airplanes:
            # Skip completely if the plane has already reached the FAF
            if hasattr(airplane, 'reached_faf') and airplane.reached_faf:
                c += 1
                continue
                
            # Store previous position for corridor crossing check
            prev_x, prev_y, prev_h, prev_phi = airplane.x, airplane.y, airplane.h, airplane.phi
            action = action_array[c * 3: (c + 1) * 3] # for airplane 0: 0-2, for airplane 1: 3-5
            last_action = self.last_action[c * 3: (c + 1) * 3]

            # Process speed action
            action_reward, updated_last_action = self._action_with_reward(airplane.action_v, action, last_action, 0)
            reward += action_reward
            reward_components["action_rewards"] += action_reward
            self.last_action[(c * 3) + 0] = updated_last_action[0]
            self.last_action[(c * 3) + 1] = updated_last_action[1]
            self.last_action[(c * 3) + 2] = updated_last_action[2]
            
            # Process altitude action
            action_reward, updated_last_action = self._action_with_reward(airplane.action_h, action, last_action, 1)
            reward += action_reward
            reward_components["action_rewards"] += action_reward
            self.last_action[(c * 3) + 0] = updated_last_action[0]
            self.last_action[(c * 3) + 1] = updated_last_action[1]
            self.last_action[(c * 3) + 2] = updated_last_action[2]
            
            # Process heading action
            action_reward, updated_last_action = self._action_with_reward(airplane.action_phi, action, last_action, 2)
            reward += action_reward
            reward_components["action_rewards"] += action_reward
            self.last_action[(c * 3) + 0] = updated_last_action[0]
            self.last_action[(c * 3) + 1] = updated_last_action[1]
            self.last_action[(c * 3) + 2] = updated_last_action[2]

            # Update the airplane position based on its current state
            # Pass airspace and wind_badness to the update_wind method
            has_fuel = airplane.step(self._airspace, self._wind_badness, self._wind_dirn)
            
            # Check if airplane is out of fuel
            if not has_fuel:
                out_of_fuel[c] = True
                # Small penalty for running out of fuel
                fuel_penalty = -10
                reward += fuel_penalty
                reward_components["fuel_penalties"] += fuel_penalty
                logger.debug(f"Aircraft {airplane.name} is out of fuel!")

            # Check if airplane is above the MVA (minimum vectoring altitude)
            try:
                mva = self._airspace.get_mva_height(airplane.x, airplane.y)

                if airplane.h < mva:
                    # Airplane has descended below minimum safe altitude - failure
                    self._win_buffer.append(0)
                    mva_penalty = -50  # Scaled down for normalization
                    reward = mva_penalty
                    reward_components["mva_penalties"] += mva_penalty
                    dones[c] = True
                    logger.debug(f"Aircraft {airplane.name} has descended below MVA!")
            except ValueError:
                # Airplane has left the defined airspace - failure
                self._win_buffer.append(0)
                dones[c] = True
                airspace_penalty = -100  # Scaled down for normalization
                reward = airspace_penalty
                reward_components["airspace_penalties"] += airspace_penalty
                mva = 0  # Dummy MVA value for the final state
                logger.debug(f"Aircraft {airplane.name} has left the airspace!")

            # Check if airplane has successfully reached the final approach corridor
            # --- SIMPLIFIED SUCCESS: Only require entering the approach corridor (ignore heading/altitude) ---
            prev_in_corridor = self._runway.corridor.corridor_horizontal.contains(
                model.geom.Point(prev_x, prev_y))
            curr_in_corridor = self._runway.corridor.corridor_horizontal.contains(
                model.geom.Point(airplane.x, airplane.y))
            if (not prev_in_corridor) and curr_in_corridor:
                self._win_buffer.append(1)
                fuel_bonus = airplane.fuel_remaining_pct
                time_bonus = max((self.timestep_limit - self.timesteps) * 0.1, 0)
                success_reward = 200 + time_bonus + 0.1 * fuel_bonus
                reward += success_reward  # Add to reward instead of overwriting
                reward_components["success_rewards"] += success_reward
                dones[c] = True
                # Set a flag to mark this plane as having reached the FAF
                airplane.reached_faf = True
                logger.debug(f"Aircraft {airplane.name} has entered the approach corridor (position only, simplified success)!")
                logger.debug(f"  Fuel bonus: +{fuel_bonus:.2f} | Time bonus: +{time_bonus:.2f}")

            # --- Reward shaping: progress, alignment, circling penalty ---
            if self._sim_parameters.reward_shaping:
                prev_d_faf = prev_d_fafs[c]
                to_faf_x = self._runway.corridor.faf[0][0] - airplane.x
                to_faf_y = self._runway.corridor.faf[1][0] - airplane.y
                new_d_faf = np.hypot(to_faf_x, to_faf_y)
                delta = 0  # Fix: always define delta
                if prev_d_faf is not None:
                    delta = prev_d_faf - new_d_faf
                # --- Denser shaping: progress reward for reducing distance to runway ---
                progress_reward = 0.05 * max(0, delta)  # Positive for progress, zero otherwise
                reward += progress_reward
                reward_components["approach_position_rewards"] += progress_reward
                # --- Increased approach position reward weight ---
                approach_reward = 2.0 / (1.0 + np.exp(0.5 * (new_d_faf - 5)))
                reward += approach_reward
                reward_components["approach_position_rewards"] += approach_reward
                # Remove or reduce far_penalty when close
                if new_d_faf > 10:
                    far_penalty = -0.005 * new_d_faf  # Scaled down
                    reward += far_penalty
                    reward_components["approach_position_rewards"] += far_penalty
                # --- Slightly increase alignment reward when close ---
                dist_to_runway = new_d_faf
                alignment_reward = 0
                if dist_to_runway < 10:
                    heading_diff = abs(model.relative_angle(self._runway.phi_to_runway, airplane.phi))
                    alignment_reward = 0.4 * (1 - heading_diff / 180.0)  # Increased from 0.2
                    reward += alignment_reward
                    reward_components["approach_angle_rewards"] += alignment_reward

                if self.timesteps > 1:
                    prev_heading = getattr(airplane, 'prev_phi', airplane.phi)
                    heading_change = abs(model.relative_angle(airplane.phi, prev_heading))
                    circling_penalty = -0.01 * heading_change  # Scaled down
                    reward += circling_penalty
                    reward_components["action_rewards"] += circling_penalty
                    airplane.prev_phi = airplane.phi

                # Penalty for not making progress for 10 steps
                if hasattr(airplane, 'no_progress_steps'):
                    if delta <= 0:
                        airplane.no_progress_steps += 1
                    else:
                        airplane.no_progress_steps = 0
                else:
                    airplane.no_progress_steps = 0
                if airplane.no_progress_steps >= 10:
                    no_progress_penalty = -2.0  # Scaled down
                    reward += no_progress_penalty
                    reward_components["approach_position_rewards"] += no_progress_penalty
                    airplane.no_progress_steps = 0

            c += 1

        # Check if any aircraft reaches the approach corridor, terminate the episode immediately
        if any(dones):
            # Don't end the episode immediately when a single aircraft reaches the FAF
            # Instead, mark that plane as having reached the FAF and continue the episode
            # for the remaining aircraft
            for i, done_status in enumerate(dones):
                if done_status and hasattr(self._airplanes[i], 'reached_faf') and self._airplanes[i].reached_faf:
                    # This aircraft has already been marked as having reached the FAF, nothing more to do
                    pass
                elif done_status:
                    # This aircraft has just reached the FAF or failed in some way
                    # We only want to continue the episode for aircraft that have reached the FAF
                    # Aircraft that have failed for other reasons should still end the episode
                    success_case = (hasattr(self._airplanes[i], 'reached_faf') and self._airplanes[i].reached_faf)
                    if not success_case:
                        # If this is a failure case (not reaching the FAF), end the episode
                        self.done = True
                        state = self._get_obs(mva)
                        if self._sim_parameters.normalize_state:
                            state = 2 * (state - self.normalization_state_min) / (self.normalization_state_max - self.normalization_state_min) - 1
                        self._update_metrics(reward)
                        truncated = False
                        return state, reward, self.done, truncated, {"original_state": self.state, "reward_components": reward_components}

        # Check for collisions between aircraft
        if len(self._airplanes) > 1:
            for i in range(len(self._airplanes)):
                # Skip planes that have already reached the FAF
                if hasattr(self._airplanes[i], 'reached_faf') and self._airplanes[i].reached_faf:
                    continue
                    
                for j in range(i + 1, len(self._airplanes)):
                    # Skip collision check if the second plane has reached the FAF
                    if hasattr(self._airplanes[j], 'reached_faf') and self._airplanes[j].reached_faf:
                        continue
                        
                    plane1 = self._airplanes[i]
                    plane2 = self._airplanes[j]
                    
                    # Calculate horizontal distance between aircraft
                    distance = self.euclidean_dist(
                        plane1.x - plane2.x, 
                        plane1.y - plane2.y
                    )
                    
                    # Calculate vertical separation in feet
                    vertical_sep = abs(plane1.h - plane2.h)
                    
                    # Check if aircraft are too close (standard separation minima)
                    # Horizontal: 3 nautical miles, Vertical: 1000 feet
                    if distance < 5 and vertical_sep < 1000:
                        # Collision detected - terminate episode
                        collision_penalty = -100
                        reward = collision_penalty
                        reward_components["airspace_penalties"] += collision_penalty
                        self.done = True
                        logger.debug(f"Aircraft collision detected between {plane1.name} and {plane2.name}!")
                        # Update the win/loss buffer
                        self._win_buffer.append(0)
                        
                        # Get the current observation before returning
                        state = self._get_obs(mva)
                        self.state = state
                        
                        # Normalize state if configured to do so
                        if self._sim_parameters.normalize_state:
                            state = 2 * (state - self.normalization_state_min) / (self.normalization_state_max - self.normalization_state_min) - 1
                        
                        # Update metrics
                        self._update_metrics(reward)
                        
                        # Immediately return from the method when a collision is detected
                        truncated = False
                        return state, reward, self.done, truncated, {"original_state": self.state, "reward_components": reward_components}

        # Check if all aircraft are out of fuel or have reached the FAF
        all_reached_faf = all(hasattr(airplane, 'reached_faf') and airplane.reached_faf for airplane in self._airplanes)
        active_planes_out_of_fuel = all(out_of_fuel[i] for i in range(len(self._airplanes)) 
                                       if not (hasattr(self._airplanes[i], 'reached_faf') and self._airplanes[i].reached_faf))
        
        # End episode if all planes have reached the FAF
        if all_reached_faf:
            self.done = True
            logger.debug("All aircraft have successfully reached the approach corridor!")
        # End episode if all active planes (not reached FAF) are out of fuel
        elif active_planes_out_of_fuel and not all(out_of_fuel):
            self.done = True
            all_fuel_penalty = -50  # was -100
            reward += all_fuel_penalty  # Add to reward instead of overwriting
            reward_components["fuel_penalties"] += all_fuel_penalty
            logger.debug("All active aircraft are out of fuel!")
        # Otherwise, episode is done if all individual planes are done
        else:
            self.done = all(dones)

        # Check if time limit exceeded
        if self.timesteps > self.timestep_limit:
            time_limit_penalty = -500  # Stronger penalty
            reward = time_limit_penalty
            reward_components["time_penalty"] = time_limit_penalty
            self.done = True
            logger.debug("Time limit exceeded!")

        # Get the current observation
        state = self._get_obs(mva)
        self.state = state

        # Normalize state if configured to do so
        if self._sim_parameters.normalize_state:
            # Correct normalization to [-1, 1]
            state = 2 * (state - self.normalization_state_min) / (self.normalization_state_max - self.normalization_state_min) - 1

        # Update tracking metrics
        self._update_metrics(reward)
        
        # Gymnasium requires truncated flag (false in this implementation)
        truncated = False

        # Return the step result tuple
        return state, reward, self.done, truncated, {"original_state": self.state, "reward_components": reward_components}

    def _update_metrics(self, reward):
        """
        Update performance tracking metrics
        
        Args:
            reward: The reward received in the current step
        """
        self.last_reward = reward
        self.total_reward += reward
        self.actions_per_timestep = self.actions_taken / max(self.timesteps, 1)

    @staticmethod
    def _reward_approach_position(d_faf: float, phi_to_runway: float, phi_rel_to_faf: float,
                                  world_max_dist: float, faf_power: float = 0.4):
        """
        Reward function for aircraft position relative to final approach
        
        Provides a reward based on position relative to the final approach course.
        Higher rewards for being closer to the FAF and aligned with the approach course.
        
        Args:
            d_faf: Distance to final approach fix in nm
            phi_to_runway: Runway heading in degrees
            phi_rel_to_faf: Relative angle to the FAF in degrees
            world_max_dist: Maximum possible distance in the world
            faf_power: Power factor for distance scaling
            
        Returns:
            Position-based reward component
        """
        # Stronger reward for distance (higher when closer to FAF)
        reward_faf = sigmoid_distance_func(d_faf, world_max_dist) ** 0.7  # More aggressive scaling
        
        # Calculate alignment with approach course - angle between runway heading and bearing to FAF
        # Smaller angle = better alignment
        angle_diff = abs(model.relative_angle(phi_to_runway, phi_rel_to_faf))
        # Normalize to 0-1 range and invert so that smaller angles give higher rewards
        normalized_angle = 1.0 - (angle_diff / 180.0)
        # Apply non-linear scaling to create stronger gradient toward proper alignment
        reward_app_angle = normalized_angle ** 1.2
        
        # Combine rewards with increased scaling factor
        return reward_faf * reward_app_angle * 1.5  # Increased from 0.8 to 1.5

    @staticmethod
    def _reward_glideslope(h, on_gp_altitude, position_factor):
        """
        Reward function for being on the correct glideslope
        
        Rewards the aircraft for being at the correct altitude for its current
        distance from the runway (following the 3-degree glideslope).
        
        Args:
            h: Current aircraft altitude in feet
            on_gp_altitude: Target glidepath altitude for current position
            position_factor: Position-based reward component for weighting
            
        Returns:
            Glideslope-based reward component
        """
        # Calculate altitude difference from ideal glidepath
        altitude_diff = abs(h - on_gp_altitude)
        
        # Harsher penalties for being too low than too high (safety concern)
        altitude_penalty_factor = 1.5 if h < on_gp_altitude else 1.0
        
        # Apply the penalty factor
        weighted_diff = altitude_diff * altitude_penalty_factor
        
        # Convert to normalized reward (closer to glidepath = higher reward)
        # More aggressive scaling to create stronger gradient
        altitude_reward = sigmoid_distance_func(weighted_diff, 6000) ** 0.8
        
        # Scale by position factor (more important when closer to approach)
        # Increased scaling factor from 0.8 to 2.0
        return altitude_reward * position_factor * 2.0

    @staticmethod
    def _reward_approach_angle(phi_to_runway, phi_rel_to_faf, phi_plane, position_factor):
        """
        Reward function for correct approach angle
        
        Rewards the aircraft for having a heading that will intercept the
        final approach course at an appropriate angle.
        
        Args:
            phi_to_runway: Runway heading in degrees
            phi_rel_to_faf: Relative angle to the FAF in degrees
            phi_plane: Current aircraft heading in degrees
            position_factor: Position-based reward component for weighting
            
        Returns:
            Approach angle-based reward component
        """
        # Function modeling optimal approach angles
        def reward_model(angle):
            return (-((angle - 22.5) / 202.0) ** 2.0 + 1.0) ** 32.0

        # Calculate the relative angle between aircraft heading and runway heading
        plane_to_runway = model.relative_angle(phi_to_runway, phi_plane)
        # Determine which side of the final approach course the aircraft is on
        side = np.sign(model.relative_angle(phi_to_runway, phi_rel_to_faf))

        # Apply the reward model with position-based scaling
        return reward_model(side * plane_to_runway) * position_factor * 1.2

    def _get_obs(self, mva):
        """
        Get the current observation state vector
        
        Calculates all the observation components including derived values
        like distance to FAF, relative angles, and glideslope information.
        
        Args:
            mva: Current minimum vectoring altitude at aircraft position
            
        Returns:
            Observation state vector
        """

        phi_rels = []
        d_fafs = []
        on_gp_altitudes = []
        phi_rel_runways = []
        fuel_percentages = []

        for airplane in self._airplanes:

            # Calculate vector from aircraft to FAF
            to_faf_x = self._runway.corridor.faf[0][0] - airplane.x
            to_faf_y = self._runway.corridor.faf[1][0] - airplane.y
            
            # Calculate relative angle to runway heading
            phi_rel_runway = self._calculate_phi_rel_runway(self._runway.phi_to_runway, airplane.phi)
            # Calculate distance to FAF
            d_faf = self.euclidean_dist(to_faf_x, to_faf_y)
            # Calculate angle to FAF
            phi_rel_faf = self._calculate_phi_rel_faf(to_faf_x, to_faf_y)

            # Calculate the target altitude for current position on glidepath
            on_gp_alt = self._calculate_on_gp_altitude(d_faf, self._faf_mva)

            d_fafs.append(d_faf)
            phi_rel_runways.append(phi_rel_runway)
            phi_rels.append(phi_rel_faf)
            on_gp_altitudes.append(on_gp_alt)
            fuel_percentages.append(airplane.fuel_remaining_pct)

        self._d_fafs = d_fafs
        self._phi_rel_fafs = phi_rels
        self._phi_rel_runways = phi_rel_runways
        self._on_gp_altitudes = on_gp_altitudes
        

        # Create the full observation state vector
        state = np.array([
            *[airplane.x for airplane in self._airplanes],  # Aircraft x position
            *[airplane.y for airplane in self._airplanes],  # Aircraft y position
            *[airplane.h for airplane in self._airplanes],  # Aircraft altitude
            *[airplane.phi for airplane in self._airplanes],  # Aircraft heading
            *[airplane.v for airplane in self._airplanes],  # Aircraft speed
            *[airplane.h - mva for airplane in self._airplanes],  # Height above minimum safe altitude
            *self._on_gp_altitudes,                     # Target altitude for glidepath
            *self._d_fafs,                              # Distance to final approach fix
            *self._phi_rel_fafs,                        # Relative angle to final approach fix
            *self._phi_rel_runways,                     # Relative angle to runway heading
            *fuel_percentages                           # NEW: Fuel percentage remaining
        ], dtype=np.float32)
        
        return state

    @staticmethod
    def _calculate_on_gp_altitude(d_faf, faf_mva):
        """
        Calculate the target altitude on the 3-degree glidepath
        
        Args:
            d_faf: Distance to final approach fix in nm
            faf_mva: Minimum vectoring altitude at the FAF
            
        Returns:
            Target altitude in feet for current distance
        """
        # 318.4 is the tangent of 3 degrees in feet per nautical mile
        return 318.4 * d_faf + faf_mva - 200

    @staticmethod
    def _calculate_phi_rel_runway(phi_to_runway, airplane_phi):
        """
        Calculate relative angle between aircraft heading and runway heading
        
        Args:
            phi_to_runway: Runway heading in degrees
            airplane_phi: Aircraft heading in degrees
            
        Returns:
            Relative angle in degrees
        """
        return model.relative_angle(phi_to_runway, airplane_phi)

    @staticmethod
    def _calculate_phi_rel_faf(to_faf_x, to_faf_y):
        """
        Calculate the angle to the FAF from the aircraft position
        
        Args:
            to_faf_x: X component of vector to FAF
            to_faf_y: Y component of vector to FAF
            
        Returns:
            Angle to FAF in degrees
        """
        return np.degrees(np.arctan2(to_faf_y, to_faf_x))

    @staticmethod
    def euclidean_dist(to_faf_x, to_faf_y):
        """
        Calculate Euclidean distance between two points
        
        Args:
            to_faf_x: X component
            to_faf_y: Y component
            
        Returns:
            Euclidean distance
        """
        return np.hypot(to_faf_x, to_faf_y)

    def _action_with_reward(self, func, action, last_action, index):
        """
        Apply an action component and calculate the resulting reward
        
        Args:
            func: Action application function
            action: Action vector
            last_action: Last action vector
            index: Index of the action component to apply
            
        Returns:
            Reward for this action component
        """
        # Convert normalized action to actual value
        action_to_take = self._denormalized_action(action[index], index)
        reward = 0.0

        try:
            # Apply the action to the aircraft
            func(action_to_take)
            # Count significant changes as distinct actions
            if not abs(action_to_take - last_action[index]) < self._action_discriminator[index]:
                self.actions_taken += 1
            
            last_action[index] = action_to_take
        except ValueError as e:
            # Invalid action (outside permissible range)
            logger.debug(f"Warning invalid action: {action_to_take} for index: {index}")
            reward -= 1.0  # Penalty for invalid action

        return reward, last_action

    def _denormalized_action(self, action, index):
        """
        Convert normalized action value to actual control value
        
        Args:
            action: Normalized action value
            index: Action component index
            
        Returns:
            Denormalized action value in appropriate units
        """
        if self._sim_parameters.discrete_action_space:
            # For discrete actions, scale by factor and add offset
            return action * self.normalization_action_factor[index] + \
                   self.normalization_action_offset[index]
        else:
            # For continuous actions, rescale from [-1,1] to actual range
            return action * self.normalization_action_factor[index] / 2 + \
                   self.normalization_action_factor[index] / 2 + \
                   self.normalization_action_offset[index]

    def render(self, mode=None):
        """
        Render the current state of the environment
        
        Args:
            mode: If provided, overrides the render_mode set during initialization
                ('human', 'rgb_array', or 'headless')
                
        Returns:
            Rendered frame or None in headless mode
        """
        # Use the provided mode or fall back to the default
        render_mode = mode if mode is not None else self.render_mode
        
        # In headless mode, don't render anything
        if render_mode == 'headless':
            return None
        
        # Initialize viewer if not already created
        if self.viewer is None:
            self._padding = 10
            
            # Use a larger fixed size for better visibility
            screen_width = 2000
            screen_height = 1600

            # Calculate dimensions based on world size
            world_size_x = self._world_x_max - self._world_x_min
            world_size_y = self._world_y_max - self._world_y_min
            
            logger.debug(screen_height, screen_width)
            logger.debug(world_size_x, world_size_y)
            # Calculate the scaling factor to fit the world to the screen
            self._scale = min(
                (screen_width - 2 * self._padding) / world_size_x,
                (screen_height - 2 * self._padding) / world_size_y
            )
            
            # Create the viewer - regular or headless depending on mode
            if render_mode == 'headless':
                self.viewer = HeadlessViewer(screen_width, screen_height)
            else:
                # Create the viewer and background (regular window but larger)
                self.viewer = rendering.Viewer(screen_width, screen_height)

                background = rendering.FilledPolygon([
                    (0, 0), 
                    (0, screen_height),
                    (screen_width, screen_height),
                    (screen_width, 0)
                ])
                background.set_color(*ColorScheme.background_inactive)
                self.viewer.add_geom(background)

                # Render static environment elements
                self._render_mvas()    # Minimum vectoring altitude areas
                self._render_runway()  # Runway
                self._render_faf()     # Final approach fix
                self._render_approach()  # Approach path
                self._render_wind()   # Wind direction
                self._render_decoration() # Additional decorations

        # Only render dynamic elements if not in headless mode
        if render_mode != 'headless':
            # Render trajectories before aircraft to keep them in the background
            self._render_trajectories()
            
            # Render dynamic elements
            for airplane in self._airplanes:
                self._render_airplane(airplane) # Aircraft symbols
            self._render_all_aircraft_info_panel(self._airplanes) # Aircraft information panel
            self._render_reward()      # Reward information



        # Return the rendered frame
        return self.viewer.render(render_mode == 'rgb_array')
    
    def _render_reward(self):
        """
        Render reward information on the screen
        """
        # Create reward text
        total_reward = f"Total reward: {self.total_reward:.2f}"
        last_reward = f"Last reward: {self.last_reward:.2f}"

        # Create and add label geometries
        label_total = Label(total_reward, 10, 48, bold=False)
        label_last = Label(last_reward, 10, 25, bold=False)

        self.viewer.add_onetime(label_total)
        self.viewer.add_onetime(label_last)
    
    def _render_airplane(self, airplane: model.Airplane):
        """
        Render the airplane using an image and information with high visibility
        
        Args:
            airplane: The aircraft to render
        """
        # Check if the plane has reached the FAF
        plane_reached_faf = hasattr(airplane, 'reached_faf') and airplane.reached_faf
        
        # For planes that have reached the FAF, place them on the runway
        # Otherwise, use their actual position
        if plane_reached_faf:
            # Position the aircraft at runway coordinates with slight offset
            # Calculate position along the runway centerline
            runway_vector = self._screen_vector(self._runway.x, self._runway.y)
            
            # Get the airplane index to add variation in touchdown point for multiple aircraft
            airplane_index = next((i for i, a in enumerate(self._airplanes) if a is airplane), 0)
            
            # Calculate an offset so planes don't stack on top of each other
            # Position aircraft at different points along the runway
            runway_offset = np.array([[0], [airplane_index * 20]])  # 20 pixel spacing
            
            # Rotate offset to align with runway heading
            rotated_offset = np.dot(model.rot_matrix(self._runway.phi_from_runway), runway_offset)
            
            # Final vector for plane that has reached FAF
            vector = runway_vector + rotated_offset
        else:
            # Normal case - use actual aircraft position
            vector = self._screen_vector(airplane.x, airplane.y)
        
        # Get airplane index to determine which color to use
        airplane_index = next((i for i, a in enumerate(self._airplanes) if a is airplane), 0)
        # Use modulo to handle case where we have more airplanes than colors
        color_index = airplane_index % len(self.trajectory_colors)
        # Get this airplane's trajectory color
        traj_color = self.trajectory_colors[color_index]
        
        # Set opacity based on FAF status
        opacity = 100 if plane_reached_faf else 255
        
        # If we can use the airplane image
        if hasattr(self, 'use_airplane_image') and self.use_airplane_image and hasattr(self, 'airplane_image'):
            # Create sprite and apply color tint to match trajectory color
            sprite = pyglet.sprite.Sprite(self.airplane_image)
            
            # Apply color tint to match trajectory color
            sprite.color = traj_color
            
            # Apply opacity based on FAF status
            sprite.opacity = opacity
            
            # Scale the sprite (adjust this value to change size)
            scale_factor = 0.15
            sprite.scale = scale_factor
            
            # Position the sprite at the calculated position
            sprite.x = vector[0][0]
            sprite.y = vector[1][0]
            
            # For planes that reached the FAF, align with runway heading
            # Otherwise, use plane's actual track
            sprite_rotation = self._runway.phi_to_runway if plane_reached_faf else airplane.track
            sprite.rotation = sprite_rotation
            
            # Draw the sprite
            self.viewer.add_onetime_sprite(sprite)
        else:
            # Fallback to original diamond shape if image can't be loaded
            render_size = 12
            corner_vector = np.array([[0], [render_size]])
            corner_top_right = np.dot(model.rot_matrix(45), corner_vector) + vector
            corner_bottom_right = np.dot(model.rot_matrix(135), corner_vector) + vector
            corner_bottom_left = np.dot(model.rot_matrix(225), corner_vector) + vector
            corner_top_left = np.dot(model.rot_matrix(315), corner_vector) + vector
            
            symbol = rendering.PolyLine([
                (corner_top_right[0][0], corner_top_right[1][0]),
                (corner_bottom_right[0][0], corner_bottom_right[1][0]),
                (corner_bottom_left[0][0], corner_bottom_left[1][0]),
                (corner_top_left[0][0], corner_top_left[1][0])
            ], True, linewidth=4)
            
            # Set color to match trajectory color
            symbol.set_color(traj_color[0], traj_color[1], traj_color[2])
                
            self.viewer.add_onetime(symbol)

            filled_symbol = rendering.FilledPolygon([
                (corner_top_right[0][0], corner_top_right[1][0]),
                (corner_bottom_right[0][0], corner_bottom_right[1][0]),
                (corner_bottom_left[0][0], corner_bottom_left[1][0]),
                (corner_top_left[0][0], corner_top_left[1][0])
            ])
            
            # Set fill color with appropriate opacity
            filled_symbol.set_color_opacity(traj_color[0], traj_color[1], traj_color[2], opacity)
                
            self.viewer.add_onetime(filled_symbol)

            aircraft_symbol_transform = rendering.Transform()
            # Set rotation to runway heading if the plane reached FAF
            rotation = math.radians(self._runway.phi_to_runway if plane_reached_faf else airplane.track)
            aircraft_symbol_transform.set_rotation(rotation)
            symbol.add_attr(aircraft_symbol_transform)
            filled_symbol.add_attr(aircraft_symbol_transform)

        # If plane reached FAF, don't render additional graphics like arrows and labels
        if plane_reached_faf:
            # Just add a simple "Landed" indicator
            landed_text = f"{airplane.name} LANDED"
            label_landed = Label(landed_text, vector[0][0] + 20, vector[1][0] + 20, bold=True)
            self.viewer.add_onetime(label_landed)
            return
            
        # The rest of the render code for active planes (arrows, labels, etc.)
        # Add track arrow to show actual direction of movement
        track_arrow_length = 12 * 6  # Long arrow
        track_arrow_vector = np.array([[track_arrow_length], [0]])
        
        # Use the airplane's track angle to rotate the vector correctly
        # Apply -90 degree adjustment to align with the airplane image
        adjusted_track_angle = airplane.track - 90
        rotated_track_arrow = np.dot(model.rot_matrix(adjusted_track_angle), track_arrow_vector)
        
        track_arrow = rendering.Line(
            (vector[0][0], vector[1][0]),
            (vector[0][0] + rotated_track_arrow[0][0], vector[1][0] + rotated_track_arrow[1][0]),
            attrs={"linewidth": 3}
        )
        # Use trajectory color for the arrow
        track_arrow.set_color(traj_color[0], traj_color[1], traj_color[2])
        self.viewer.add_onetime(track_arrow)
        
        # Add arrowhead to the track line
        arrowhead_size = 6 * 2.5
        arrowhead_vector1 = np.array([[-arrowhead_size], [-arrowhead_size]])
        arrowhead_vector2 = np.array([[-arrowhead_size], [arrowhead_size]])
        
        arrowhead_pos = vector + rotated_track_arrow
        
        # Use the same adjusted track angle for the arrowhead rotation
        rotated_arrowhead1 = np.dot(model.rot_matrix(adjusted_track_angle), arrowhead_vector1) + arrowhead_pos
        rotated_arrowhead2 = np.dot(model.rot_matrix(adjusted_track_angle), arrowhead_vector2) + arrowhead_pos
        
        arrowhead1 = rendering.Line(
            (arrowhead_pos[0][0], arrowhead_pos[1][0]),
            (rotated_arrowhead1[0][0], rotated_arrowhead1[1][0]),
            attrs={"linewidth": 3}
        )
        arrowhead2 = rendering.Line(
            (arrowhead_pos[0][0], arrowhead_pos[1][0]),
            (rotated_arrowhead2[0][0], rotated_arrowhead2[1][0]),
            attrs={"linewidth": 3}
        )
        # Use trajectory color for arrowheads as well
        arrowhead1.set_color(traj_color[0], traj_color[1], traj_color[2])
        arrowhead2.set_color(traj_color[0], traj_color[1], traj_color[2])
        self.viewer.add_onetime(arrowhead1)
        self.viewer.add_onetime(arrowhead2)
        
        # Calculate crab angle (difference between heading and track)
        crab_angle = model.relative_angle(airplane.phi, airplane.track)

        # Calculate offset for heading arrow to show crab angle difference from track
        # Only display heading arrow if there's a significant crab angle
        if abs(crab_angle) > 2.0:
            # Add heading arrow (blue) to show the direction the nose is pointing
            heading_arrow_length = 12 * 5
            heading_arrow_vector = np.array([[heading_arrow_length], [0]])
            
            # Apply -90 degree adjustment to the heading arrow as well
            adjusted_heading_angle = airplane.phi - 90
            rotated_heading_arrow = np.dot(model.rot_matrix(adjusted_heading_angle), heading_arrow_vector)
            
            heading_arrow = rendering.Line(
                (vector[0][0], vector[1][0]),
                (vector[0][0] + rotated_heading_arrow[0][0], vector[1][0] + rotated_heading_arrow[1][0]),
                attrs={"linewidth": 2}
            )
            heading_arrow.set_color(50, 50, 255)  # Keep heading arrow blue for visibility
            self.viewer.add_onetime(heading_arrow)
            
            # Add blue arrowhead to heading arrow
            heading_arrowhead_size = 12 * 1.5
            heading_arrowhead_vector1 = np.array([[-heading_arrowhead_size], [-heading_arrowhead_size]])
            heading_arrowhead_vector2 = np.array([[-heading_arrowhead_size], [heading_arrowhead_size]])
            
            heading_arrowhead_pos = vector + rotated_heading_arrow
            
            # Use the adjusted heading angle for the arrowheads too
            rotated_heading_arrowhead1 = np.dot(model.rot_matrix(adjusted_heading_angle), heading_arrowhead_vector1) + heading_arrowhead_pos
            rotated_heading_arrowhead2 = np.dot(model.rot_matrix(adjusted_heading_angle), heading_arrowhead_vector2) + heading_arrowhead_pos
            
            heading_arrowhead1 = rendering.Line(
                (heading_arrowhead_pos[0][0], heading_arrowhead_pos[1][0]),
                (rotated_heading_arrowhead1[0][0], rotated_heading_arrowhead1[1][0]),
                attrs={"linewidth": 2}
            )
            heading_arrowhead2 = rendering.Line(
                (heading_arrowhead_pos[0][0], heading_arrowhead_pos[1][0]),
                (rotated_heading_arrowhead2[0][0], rotated_heading_arrowhead2[1][0]),
                attrs={"linewidth": 2}
            )
            heading_arrowhead1.set_color(50, 50, 255)  # Keep heading arrow blue
            heading_arrowhead2.set_color(50, 50, 255)  # Keep heading arrow blue
            self.viewer.add_onetime(heading_arrowhead1)
            self.viewer.add_onetime(heading_arrowhead2)

        # Create labels with aircraft information
        render_altitude = round(airplane.h) // 100
        render_speed = round(airplane.v)
        render_text = f"FL{render_altitude:03} {render_speed}kt"
        
        # Position labels a bit offset from the aircraft
        label_offset = 30  # Pixels
        label_x = vector[0][0] - label_offset + 10
        label_y = vector[1][0] - label_offset 

        # Add aircraft callsign and details labels
        label_name = Label(airplane.name, x=label_x, y=label_y)
        label_details = Label(render_text, x=label_x, y=label_y - 20)
        label_hdg_trk = Label(f"HDG:{airplane.phi:.0f}° TRK:{airplane.track:.0f}°", 
                              x=label_x, y=label_y - 40)
        label_wind = Label(f"Wind: {math.sqrt(airplane.wind_x**2 + airplane.wind_y**2):.1f}kt", 
                           x=label_x, y=label_y - 60)
        
        self.viewer.add_onetime(label_name)
        self.viewer.add_onetime(label_details)
        self.viewer.add_onetime(label_hdg_trk)
        self.viewer.add_onetime(label_wind)
        
        # Add a fuel gauge near the airplane
        fuel_gauge = FuelGauge(
            x=label_x, 
            y=label_y - 100,
            width=50,
            height=8,
            fuel_percentage=airplane.fuel_remaining_pct,
        )
        self.viewer.add_onetime(fuel_gauge)
        
    def _render_all_aircraft_info_panel(self, airplanes: list[model.Airplane]):
        """
        Render aircraft parameters as text labels in the corner of the screen
        with improved spacing for better readability
        """
        # Position for the labels - bottom right corner with padding
        x_pos = self.viewer.width - 300  # Further right for more space
        y_pos = self.viewer.height - 20  # Top side with margin

        # Create title with larger font
        title = Label("AIRCRAFT PARAMETERS", x_pos, y_pos, bold=True)
        self.viewer.add_onetime(title)
        
        # # Add a legend for the visualization with better spacing
        # legend_y = y_pos - 40  # Increased spacing below title
        # legend_title = Label("LEGEND:", x_pos, legend_y, bold=True)
        # legend_heading = Label("Blue Arrow: Aircraft Heading", x_pos, legend_y - 30)
        # legend_track = Label("Green Arrow: Ground Track", x_pos, legend_y - 60)
        
        # self.viewer.add_onetime(legend_title)
        # self.viewer.add_onetime(legend_heading)
        # self.viewer.add_onetime(legend_track)
        
        # Start aircraft details below the legend with more spacing
        # panel_y = legend_y - 100  # Increased spacing after legend
        
        # Add horizontal separator line
        # separator = rendering.Line((x_pos - 10, panel_y + 10), (x_pos + 280, panel_y + 10))
        # separator.set_color(200, 200, 200)  # Light gray
        # self.viewer.add_onetime(separator)
        
        for airplane_index, airplane in enumerate(airplanes):
            # Add aircraft title with colored background to match aircraft color
            aircraft_title_y = y_pos - 30
            
            # Get airplane's color from trajectory colors
            color_index = airplane_index % len(self.trajectory_colors)
            traj_color = self.trajectory_colors[color_index]

            x_pos = x_pos + 5  # Indent for better visibility
            # Add colored rectangle behind aircraft title
            title_bg = rendering.FilledPolygon([
                (x_pos - 5, aircraft_title_y + 5),
                (x_pos + 270, aircraft_title_y + 5),
                (x_pos + 270, aircraft_title_y - 30),
                (x_pos - 5, aircraft_title_y - 30)
            ])
            title_bg.set_color_opacity(traj_color[0], traj_color[1], traj_color[2], 100)
            self.viewer.add_onetime(title_bg)
            
            # Add aircraft title with larger font
            aircraft_title = Label(f"FLIGHT {airplane.name}", x_pos, aircraft_title_y, bold=True)
            self.viewer.add_onetime(aircraft_title)
            
            # Get current MVA with error handling
            try:
                current_mva = self._airspace.get_mva_height(airplane.x, airplane.y)
                height_above_mva = airplane.h - current_mva
            except ValueError:
                height_above_mva = 0
            
            # Calculate crab angle (difference between heading and track)
            crab_angle = model.relative_angle(airplane.phi, airplane.track)
            
            # Create parameter strings with improved formatting and grouping
            position_params = [
                f"Position: ({airplane.x:.1f}, {airplane.y:.1f})",
                f"Altitude: {airplane.h} ft"
            ]
            
            direction_params = [
                f"Heading: {airplane.phi:.1f}°",
                f"Ground Track: {airplane.track:.1f}°",
                f"Crab Angle: {crab_angle:.1f}°"
            ]
            
            speed_params = [
                f"Airspeed: {airplane.v:.0f} knots",
                f"Ground Speed: {airplane.ground_speed:.0f} knots",
                f"Wind: {math.sqrt(airplane.wind_x**2 + airplane.wind_y**2):.1f} knots"
            ]
            
            approach_params = [
                f"Distance to FAF: {self._d_fafs[airplane_index]:.1f} nm",
                f"Height above MVA: {height_above_mva:.0f} ft",
                f"Fuel remaining: {airplane.fuel_remaining_pct:.1f}%"
            ]
            
            # Start position for parameter groups
            group_y = aircraft_title_y - 40
            line_spacing = 30  # Increased line spacing
            
            # Function to render a group of parameters
            def render_param_group(params, start_y, group_title=None):
                y = start_y
                if group_title:
                    group_label = Label(group_title, x_pos, y, bold=True)
                    self.viewer.add_onetime(group_label)
                    y -= 25  # Space after group title
                
                for param in params:
                    param_label = Label(param, x_pos + 10, y)  # Indent parameters
                    self.viewer.add_onetime(param_label)
                    y -= line_spacing
                
                return y  # Return the new y position
            
            # Render each group with titles
            group_y = render_param_group(position_params, group_y, "Position")
            group_y -= 10  # Extra space between groups
            group_y = render_param_group(direction_params, group_y, "Direction")
            group_y -= 10  # Extra space between groups
            group_y = render_param_group(speed_params, group_y, "Speed")
            group_y -= 10  # Extra space between groups
            group_y = render_param_group(approach_params, group_y, "Approach")
            group_y -= 20  # Extra space for fuel gauge

            # Add fuel gauge below parameters
            fuel_gauge = FuelGauge(
                x=x_pos + 10,  # Indent gauge
                y=group_y,
                width=220,
                height=20,  # Taller gauge for better visibility
                fuel_percentage=airplane.fuel_remaining_pct,
            )
            self.viewer.add_onetime(fuel_gauge)

            # Add fuel label
            # fuel_label = Label(f"Fuel: {airplane.fuel_remaining_pct:.1f}%", x_pos + 120, group_y + 10)
            # self.viewer.add_onetime(fuel_label)

            # Add horizontal separator line between aircraft
            panel_y = group_y - 40
            separator = rendering.Line((x_pos - 10, panel_y + 10), (x_pos + 280, panel_y + 10))
            separator.set_color(200, 200, 200)  # Light gray
            self.viewer.add_onetime(separator)

            y_pos = panel_y - 5  # Update y position for next aircraft

            
    def _render_approach(self):
        """
        Render the approach path on the screen
        """
        # Get initial approach fix coordinates
        iaf_x = self._runway.corridor.iaf[0][0]
        iaf_y = self._runway.corridor.iaf[1][0]
        
        # Create a dashed line from runway to IAF
        dashes = 48
        runway_vector = self._screen_vector(self._runway.x, self._runway.y)
        runway_iaf = np.array([[iaf_x - self._runway.x], [iaf_y - self._runway.y]]) * self._scale
        
        # Draw the dashed line segments
        for i in range(int(dashes / 2 + 1)):
            start = runway_vector + runway_iaf / dashes * 2 * i
            end = runway_vector + runway_iaf / dashes * (2 * i + 1)
            dash = rendering.Line(
                (start[0][0], start[1][0]),
                (end[0][0], end[1][0])
            )
            dash.set_color(*ColorScheme.lines_info)
            self.viewer.add_geom(dash)
            
    def _render_wind(self):
        """
        Render the wind as flow streamlines with color indicating wind speed
        """
        # Get wind field from scenario
        self._wind = self._scenario.wind
        wind_field = self._wind.wind_field
        
        # Parameters for flow lines - density reduced for better visualization
        flow_discretization = int(self._wind.resolution * 4)  # Increase spacing between flow lines
        
        # Parameters for streamlines
        flow_length = 12.0        # Length of each flow line - longer for more visible flows
        num_segments = 24         # Number of segments in each flow line - more segments for smoother curves
        segment_length = flow_length / num_segments
        
        # Calculate max wind speed for color normalization
        max_wind_speed = 0.1  # Minimum to avoid division by zero
        for x in range(0, len(wind_field), flow_discretization):
            for y in range(0, len(wind_field[0]), flow_discretization):
                try:
                    mva_type = self._airspace.find_mva(x, y).mva_type
                except ValueError:
                    mva_type = model.MvaType.GENERIC
                
                wind_vector = model.get_wind_speed(x, y, 20_000, mva_type, self._wind_badness, self._wind_dirn)
                wind_speed = np.linalg.norm(wind_vector)
                max_wind_speed = max(max_wind_speed, wind_speed)
        
        # Create streamlines
        for x in range(0, len(wind_field), flow_discretization):
            for y in range(0, len(wind_field[0]), flow_discretization):
                try:
                    # Find which MVA we're in and get its type
                    mva = self._airspace.find_mva(x, y)
                    mva_type = mva.mva_type
                except ValueError:
                    # Outside airspace, use generic wind
                    mva_type = model.MvaType.GENERIC
                
                # Get wind vector at starting point
                start_vector = model.get_wind_speed(x, y, 20_000, mva_type, self._wind_badness, self._wind_dirn)
                wind_speed = np.linalg.norm(start_vector)
                
                # Skip areas with negligible wind
                if wind_speed < 0.7:  # Slightly higher threshold for cleaner visualization
                    continue
                
                # Normalize wind vector
                unit_vector = start_vector / wind_speed if wind_speed > 0 else np.array([0, 0])
                
                # Calculate wind speed ratio for color interpolation
                speed_ratio = min(wind_speed / max_wind_speed, 1.0)
                
                # Select color based on wind speed - matching the reference image
                if speed_ratio < 0.25:
                    r, g, b = 255, 255, 255  # White for slowest winds
                elif speed_ratio < 0.45:
                    r, g, b = 180, 225, 250  # Light blue for light winds
                elif speed_ratio < 0.7:
                    r, g, b = 60, 170, 230   # Medium blue for moderate winds
                else:
                    r, g, b = 20, 80, 180    # Dark blue for strong winds
                
                # Start position for streamline
                current_x, current_y = x, y
                
                # Generate streamline points by following the wind field
                flow_points = [(current_x, current_y)]
                
                # Create curved streamline by following wind vectors
                for i in range(num_segments):
                    # Get the next position by following the wind
                    next_x = current_x + unit_vector[0] * segment_length
                    next_y = current_y + unit_vector[1] * segment_length
                    
                    # Check if we're still in bounds
                    if (0 <= next_x < len(wind_field) and 0 <= next_y < len(wind_field[0])):
                        # Add point to flow line
                        flow_points.append((next_x, next_y))
                        
                        # Update current position
                        current_x, current_y = next_x, next_y
                        
                        # Get new wind vector at this position for smooth curve
                        try:
                            mva = self._airspace.find_mva(int(current_x), int(current_y))
                            mva_type = mva.mva_type
                        except ValueError:
                            mva_type = model.MvaType.GENERIC
                        
                        next_vector = model.get_wind_speed(int(current_x), int(current_y), 20_000, 
                                                          mva_type, self._wind_badness, self._wind_dirn)
                        next_wind_speed = np.linalg.norm(next_vector)
                        
                        # Update unit_vector (creates the curve effect)
                        unit_vector = next_vector / next_wind_speed if next_wind_speed > 0 else np.array([0, 0])
                    else:
                        # Stop if we go out of bounds
                        break
                
                # Draw the flow line if we have enough points
                if len(flow_points) > 3:  # Need at least a few points for a meaningful line
                    # Convert world coordinates to screen coordinates
                    screen_points = []
                    for point_x, point_y in flow_points:
                        point = self._screen_vector(point_x, point_y)
                        screen_points.append((point[0][0], point[1][0]))
                    
                    # Skip if no valid screen points
                    if not screen_points:
                        continue
                    
                    # Line width based on wind speed (1-3 pixels)
                    line_width = 1 + int(speed_ratio * 2)
                    
                    # Draw the streamline
                    flow_line = rendering.PolyLine(screen_points, False, linewidth=line_width)
                    flow_line.set_color_opacity(r, g, b, 200)
                    self.viewer.add_geom(flow_line)
                    
                    # Add arrow at the end of the streamline
                    if len(screen_points) > 3:
                        # Use second-to-last and last points for direction
                        last_idx = len(screen_points) - 1
                        prev_idx = last_idx - 1
                        
                        # Get arrow direction from the last segment
                        dx = screen_points[last_idx][0] - screen_points[prev_idx][0]
                        dy = screen_points[last_idx][1] - screen_points[prev_idx][1]
                        angle = math.atan2(dy, dx)
                        
                        # Arrow size based on wind speed
                        arrow_size = 5 + int(speed_ratio * 5)
                        arrow_pt = screen_points[last_idx]
                        
                        # Create arrowhead
                        arrowhead = rendering.FilledPolygon([
                            arrow_pt,
                            (arrow_pt[0] - arrow_size * math.cos(angle - math.pi/6), 
                             arrow_pt[1] - arrow_size * math.sin(angle - math.pi/6)),
                            (arrow_pt[0] - arrow_size * math.cos(angle + math.pi/6), 
                             arrow_pt[1] - arrow_size * math.sin(angle + math.pi/6))
                        ])
                        arrowhead.set_color_opacity(r, g, b, 230)
                        self.viewer.add_geom(arrowhead)

    def _render_faf(self):
        """
        Render the final approach fix symbol (triangle)
        """
        faf_screen_render_size = 6

        # Get FAF coordinates
        faf_x = self._runway.corridor.faf[0][0]
        faf_y = self._runway.corridor.faf[1][0]
        faf_vector = self._screen_vector(faf_x, faf_y)

        # Create a triangle symbol for the FAF
        corner_vector = np.array([[0], [faf_screen_render_size]])
        corner_top = faf_vector + corner_vector
        corner_right = np.dot(model.rot_matrix(121), corner_vector) + faf_vector
        corner_left = np.dot(model.rot_matrix(242), corner_vector) + faf_vector

        poly_line = rendering.PolyLine([
            (corner_top[0][0], corner_top[1][0]),
            (corner_right[0][0], corner_right[1][0]),
            (corner_left[0][0], corner_left[1][0]),
        ], True)
        poly_line.set_color(*ColorScheme.lines_info)
        self.viewer.add_geom(poly_line)
                
    def _render_mvas(self):
        """
        Renders the outlines of the minimum vectoring altitudes onto the screen.
        Renders largest MVAs first, then smaller ones on top for proper layering.
        """
        def transform_world_to_screen(coords):
            return [((coord[0] - self._world_x_min) * self._scale + self._padding,
                     (coord[1] - self._world_y_min) * self._scale + self._padding) for coord in coords]
        
        # Sort MVAs by area size (largest first)
        sorted_mvas = sorted(self._mvas, key=lambda mva: mva.area.area, reverse=True)

        # Render the fill for each MVA in size order (largest to smallest)
        for mva in sorted_mvas:
            coordinates = transform_world_to_screen(mva.area.exterior.coords)

            fill = rendering.FilledPolygon(coordinates)
            
            # Select color based on MVA type
            color = None
            match mva.mva_type:
                case model.MvaType.GENERIC:
                    color = ColorScheme.generic_mva_color
                case model.MvaType.MOUNTAINOUS:
                    color = ColorScheme.mountainous_mva_color
                case model.MvaType.WEATHER:
                    color = ColorScheme.weather_mva_color
                case model.MvaType.OCEANIC:
                    color = ColorScheme.oceanic_mva_color
            
            fill.set_color_opacity(*color)
            self.viewer.add_geom(fill)

        # Render outlines in same order (largest to smallest)
        for mva in sorted_mvas:
            coordinates = transform_world_to_screen(mva.area.exterior.coords)
            outline = rendering.PolyLine(coordinates, True)
            outline.set_color(*ColorScheme.mva)
            self.viewer.add_geom(outline)

        # Render labels in same order (largest to smallest)
        for mva in sorted_mvas:
            # Add label on edge of the mva indicating its FL
            label_pos = transform_world_to_screen(mva.area.centroid.coords)[0]
            
            # Create the text content for both labels
            fl_text = f"FL{(mva.height // 100):03}"
            type_text = f"{mva.mva_type.value}"
            
            # Get approximate text widths (assuming monospace, around 7 pixels per character)
            fl_text_width = len(fl_text) * 7
            type_text_width = len(type_text) * 7
            
            # Position labels with horizontal centering adjustment
            fl_label = Label(fl_text, 
                          label_pos[0] - fl_text_width/2,  # Center horizontally
                          label_pos[1], 
                          bold=False)
            type_label = Label(type_text, 
                            label_pos[0] - type_text_width/2,  # Center horizontally
                            label_pos[1] - 15, 
                            bold=False)
            
            self.viewer.add_geom(fl_label)
            self.viewer.add_geom(type_label)

    def _render_runway(self):
        """
        Renders the runway symbol onto the screen with approach corridor
        """
        runway_length = 1.7 * self._scale
        runway_to_threshold_vector = \
            np.dot(model.rot_matrix(self._runway.phi_from_runway), np.array([[0], [runway_length / 2]]))
        runway_vector = self._screen_vector(self._runway.x, self._runway.y)
        
        start_point = (runway_vector[0][0] - runway_to_threshold_vector[0][0], 
                       runway_vector[1][0] - runway_to_threshold_vector[1][0])
        end_point = (runway_vector[0][0] + runway_to_threshold_vector[0][0], 
                     runway_vector[1][0] + runway_to_threshold_vector[1][0])
        
        runway_line = rendering.Line(start_point, end_point, attrs={"linewidth": 10})
        runway_line.set_color(*ColorScheme.runway)
        self.viewer.add_geom(runway_line)

        # Render the approach corridor shape
        corridor = self._runway.corridor
        if hasattr(corridor, 'corridor_horizontal') and corridor.corridor_horizontal is not None:
            # Get the coordinates of the horizontal corridor polygon
            coords = corridor.corridor_horizontal.exterior.coords
            
            # Transform world coordinates to screen coordinates
            screen_coords = []
            for coord in coords:
                point = self._screen_vector(coord[0], coord[1])
                screen_coords.append((point[0][0], point[1][0]))
            
            # Create filled polygon with transparent fill
            corridor_fill = rendering.FilledPolygon(screen_coords)
            corridor_fill.set_color_opacity(0, 150, 0, 50)  # Light green with transparency
            self.viewer.add_geom(corridor_fill)
            
            # Add outline
            corridor_outline = rendering.PolyLine(screen_coords, True, linewidth=2)
            corridor_outline.set_color(0, 200, 0)  # Green outline
            self.viewer.add_geom(corridor_outline)
            
            # Add a label indicating this is the approach corridor
            # Find a good position for the label (center of the polygon)
            if len(screen_coords) > 0:
                x_coords = [p[0] for p in screen_coords]
                y_coords = [p[1] for p in screen_coords]
                label_x = sum(x_coords) / len(x_coords)
                label_y = sum(y_coords) / len(y_coords)
                
                # corridor_label = Label("APPROACH CORRIDOR", label_x, label_y, bold=True)
                # self.viewer.add_geom(corridor_label)

    def _screen_vector(self, x, y):
        """
        Converts an in world vector to an on screen vector by shifting and scaling
        :param x: World vector x
        :param y: World vector y
        :return: Numpy array vector with on screen coordinates
        """
        return np.array([
            [(x - self._world_x_min) * self._scale + self._padding],
            [(y - self._world_y_min) * self._scale + self._padding]
        ])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self, seed=None, options=None):
            """
            Reset the environment to a new initial state
            
            Creates a new airplane instance in a safe location
            
            Args:
                seed: Random seed (unused in this implementation)
                options: Additional options (unused)
                
            Returns:
                Initial state and info dictionary
            """
            self.done = False

            # Set random seed for deterministic reset
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            else:
                np.random.seed(42)
                random.seed(42)

            # Calculate the center of the airspace for a safer starting position
            center_x = (self._world_x_min + self._world_x_max) / 2
            center_y = (self._world_y_min + self._world_y_max) / 2
            
            self._airplanes: list[model.Airplane] = []
            self._d_fafs = []
            self._phi_rel_fafs = []
            self._phi_rel_runways = []
            self._on_gp_altitudes = []
            
            # Get all available entry points
            available_entrypoints = self._scenario.entrypoints.copy()
            
            # If there are fewer entry points than requested airplanes, duplicate some
            while len(available_entrypoints) < self._airplane_count:
                available_entrypoints.extend(self._scenario.entrypoints)
            
            # Always use the first entry point and first altitude for each aircraft for deterministic start
            selected_entrypoints = available_entrypoints[:self._airplane_count]

            for i in range(self._airplane_count):
                # Use a different entry point for each aircraft
                entry_point = selected_entrypoints[i]
                
                # Handle the case where entry_point is a list (for multiple planes in curriculum)
                # This happens when we have a list of entry points for multiple planes
                if isinstance(entry_point, list):
                    # If entry_point is a list, use the appropriate entry point for this plane
                    if i < len(entry_point):
                        # If there are enough entry points in the list, use the corresponding one
                        actual_entry_point = entry_point[i]
                    else:
                        # Otherwise, use the first entry point in the list (fallback)
                        actual_entry_point = entry_point[0]
                else:
                    # Normal case: entry_point is a single EntryPoint object
                    actual_entry_point = entry_point
                
                # Place the airplane exactly at the entry point (no randomization)
                x = actual_entry_point.x
                y = actual_entry_point.y
                altitude = actual_entry_point.levels[0] * 100  # Always use the first altitude
                phi = actual_entry_point.phi
                
                # Create new airplane instances for the simulation
                self._airplanes.append(
                    model.Airplane(
                        self._sim_parameters,
                        f"FLT{i+1:03}",                                    # Flight identifier
                        x, y,                                       # Position
                        altitude,                                    # Altitude
                        phi,                            # Heading
                        250,                                         # Initial speed
                        starting_fuel=self._starting_fuel            # Starting fuel
                    )
                )
                
                # Initialize arrays
                self._d_fafs.append(0)
                self._phi_rel_fafs.append(0)
                self._phi_rel_runways.append(0)
                self._on_gp_altitudes.append(0)
        
            # Reset state and tracking variables
            self.state = self._get_obs(0)
            # Normalize state if configured to do so
            if self._sim_parameters.normalize_state:
                self.state = 2 * (self.state - self.normalization_state_min) / (self.normalization_state_max - self.normalization_state_min) - 1
            self.total_reward = 0
            self.last_reward = 0
            self._actions_ignoring_resets += self.actions_taken
            self.actions_taken = 0
            self.timesteps = 0
            self._episodes_run += 1

            # Update the win/loss tracking buffer
            if len(self._win_buffer) < self._win_buffer_size:
                # Simulation ended from outside (time limit, etc.)
                self._win_buffer.append(0)
            
            # Calculate the new winning ratio (moving average over recent episodes)
            self.winning_ratio = self.winning_ratio + 1 / self._win_buffer_size * \
                                (self._win_buffer[-1] - self._win_buffer.pop(0))

            return self.state, {"info": "Environment reset"}

    def _render_trajectories(self):
        """
        Render the trajectory trails of aircraft based on their position history.
        """
        if not self.show_trajectories:
            return
            
        for i, airplane in enumerate(self._airplanes):
            # Get position history from the airplane
            history = getattr(airplane, 'position_history', [])
            # Skip if history is empty
            if not history:
                continue
                
            # Convert world coordinates to screen coordinates
            screen_points = []
            for point in history:
                if len(point) == 2:  # Ensure the point has x, y coordinates
                    x, y = point
                    point_vector = self._screen_vector(x, y)
                    screen_points.append((point_vector[0][0], point_vector[1][0]))
            
            # Skip if no valid screen points
            if not screen_points:
                continue
                
            # Draw polyline connecting history points
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            trail = rendering.PolyLine(screen_points, False, linewidth=2)
            trail.set_color(*color)
            self.viewer.add_onetime(trail)

    def _render_decoration(self):
        """
        Render additional decorations on the screen (if any).
        """
        # Placeholder for future decorations

        def transform_world_to_screen(coords):
            return [((coord[0] - self._world_x_min) * self._scale + self._padding,
                     (coord[1] - self._world_y_min) * self._scale + self._padding) for coord in coords]

        if hasattr(self._scenario, "decorations") and self._scenario.decorations:
            # Check if decorations are present in the scenario
            logger.info(f"Decorations found: {self._scenario.decorations}")
        else:
            return
        
        for decor in self._scenario.decorations:
            # decor is a shapely geometry
            # match it based on type
            match decor.geom_type:
                case "Point":
                    # render a point
                    point = rendering.make_circle(5)
                    point.set_color(*ColorScheme.decoration)
                    point.set_translation(decor.x, decor.y)
                    self.viewer.add_geom(point)
                case "LineString":
                    # render a line

                    # transform the coordinates to screen coordinates
                    coords = transform_world_to_screen(decor.coords)


                    line = rendering.PolyLine(coords, False, linewidth=2)
                    line.set_color_opacity(*ColorScheme.decoration)
                    self.viewer.add_geom(line)
                    logger.info(f"LineString added.")
                case "Polygon":
                    # render a polygon
                    polygon = rendering.FilledPolygon(decor.exterior.coords)
                    polygon.set_color(*ColorScheme.decoration)
                    self.viewer.add_geom(polygon)
                case _:
                    # Unsupported geometry type
                    logger.warning(f"Unsupported decoration geometry type: {decor.geom_type}")

    def toggle_trajectories(self):
        """
        Toggle the display of aircraft trajectories
        
        Returns:
            Current state of trajectory display (True=enabled, False=disabled)
        """
        self.show_trajectories = not self.show_trajectories
        return self.show_trajectories