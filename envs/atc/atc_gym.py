# minimal-atc-rl/envs/atc/atc_gym.py
import math
import random
import os

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

    def __init__(self, airplane_count=1, sim_parameters=model.SimParameters(1), scenario=None, render_mode='rgb_array'):
        """
        Initialize the ATC gym environment
        
        Args:
            airplane_count: Number of airplanes to simulate
            sim_parameters: Simulation parameters like timestep size
            scenario: The airspace scenario to use (runways, MVAs, etc.)
            render_mode: 'human' for window rendering, 'rgb_array' for array output, 'headless' for no rendering
        """

        self._airplane_count = airplane_count

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
        self._wind = scenario.wind
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

        print("Number of airplanes: ", len(self._airplanes))
        print("Observation space: ", len(self.normalization_state_min))
        print("Action space: ", len(self.normalization_action_offset))

        # Initialize trajectory visualization settings
        self.show_trajectories = True  # Default: trajectories enabled
        self.trajectory_colors = [
            (255, 0, 0),    # Red
            (0, 200, 255),  # Cyan
            (255, 165, 0),  # Orange
            (255, 0, 255),  # Magenta
            (0, 255, 0)     # Green
        ]
        self.max_trail_length = 500  # Maximum number of history points to render
        
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
            print(f"Successfully loaded airplane image from {airplane_path}")
        except Exception as e:
            print(f"Could not load airplane image from any path. Error: {e}")
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
            has_fuel = airplane.step()
            
            # Check if airplane is out of fuel
            if not has_fuel:
                out_of_fuel[c] = True
                # Small penalty for running out of fuel
                fuel_penalty = -10
                reward += fuel_penalty
                reward_components["fuel_penalties"] += fuel_penalty
                print(f"Aircraft {airplane.name} is out of fuel!")

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
                    print(f"Aircraft {airplane.name} has descended below MVA!")
            except ValueError:
                # Airplane has left the defined airspace - failure
                self._win_buffer.append(0)
                dones[c] = True
                airspace_penalty = -100  # Scaled down for normalization
                reward = airspace_penalty
                reward_components["airspace_penalties"] += airspace_penalty
                mva = 0  # Dummy MVA value for the final state
                print(f"Aircraft {airplane.name} has left the airspace!")

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
                reward = success_reward
                reward_components["success_rewards"] += success_reward
                dones[c] = True
                print(f"Aircraft {airplane.name} has entered the approach corridor (position only, simplified success)!")
                print(f"  Fuel bonus: +{fuel_bonus:.2f} | Time bonus: +{time_bonus:.2f}")

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

        # If any aircraft reaches the approach corridor, terminate the episode immediately
        if any(dones):
            self.done = True
            # Return early to avoid overwriting self.done
            state = self._get_obs(mva)
            if self._sim_parameters.normalize_state:
                state = 2 * (state - self.normalization_state_min) / (self.normalization_state_max - self.normalization_state_min) - 1
            self._update_metrics(reward)
            truncated = False
            return state, reward, self.done, truncated, {"original_state": self.state, "reward_components": reward_components}

        # NEW: Check if all aircraft are out of fuel
        if all(out_of_fuel):
            self.done = True
            all_fuel_penalty = -50  # was -100
            reward = all_fuel_penalty
            reward_components["fuel_penalties"] += all_fuel_penalty
            print("All aircraft are out of fuel!")
        else:
            self.done = all(dones)

        # Check if time limit exceeded
        if self.timesteps > self.timestep_limit:
            time_limit_penalty = -500  # Stronger penalty
            reward = time_limit_penalty
            reward_components["time_penalty"] = time_limit_penalty
            self.done = True
            print("Time limit exceeded!")

        # Get the current observation
        state = self._get_obs(mva)
        self.state = state

        # Normalize state if configured to do so
        if self._sim_parameters.normalize_state:
            # Correct normalization to [-1, 1]
            state = 2 * (state - self.normalization_state_min) / (self.normalization_state_max - self.normalization_state_min) - 1

        # Update tracking metrics
        self._update_metrics(reward)
        
        # Print detailed reward breakdown
        if self.timesteps % 20 == 0 or self.done:  # Print every 20 steps and at episode end
            print("\n" + "="*60)
            print(f"Step {self.timesteps} Reward Breakdown:")
            print("-"*60)
            for component, value in reward_components.items():
                if abs(value) > 0.001:  # Only show non-zero components
                    print(f"{component.replace('_', ' ').title()}: {value:.4f}")
            print("-"*60)
            print(f"Total Reward: {reward:.4f}")
            print(f"Total Cumulative Reward: {self.total_reward:.4f}")
            print("="*60 + "\n")
        
        # More frequent but compact reward updates
        elif self.timesteps % 5 == 0:
            significant_rewards = {k: v for k, v in reward_components.items() if abs(v) > 0.001}
            components_str = " | ".join([f"{k.split('_')[0]}: {v:.2f}" for k, v in significant_rewards.items()])
            print(f"Step {self.timesteps}: Reward = {reward:.2f} ({components_str})")

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

        # Calculate the relative angle between aircraft heading and runway
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
            # print(f"Warning invalid action: {action_to_take} for index: {index}")
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
            
            print(screen_height, screen_width)
            print(world_size_x, world_size_y)
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
        label_total = Label(total_reward, 10, 40, bold=False)
        label_last = Label(last_reward, 10, 25, bold=False)

        self.viewer.add_onetime(label_total)
        self.viewer.add_onetime(label_last)
    
    def _render_airplane(self, airplane: model.Airplane):
        """
        Render the airplane using an image and information with high visibility
        
        Args:
            airplane: The aircraft to render
        """
        # Get screen coordinates for the airplane
        vector = self._screen_vector(airplane.x, airplane.y)
        
        # Get airplane index to determine which color to use
        airplane_index = next((i for i, a in enumerate(self._airplanes) if a is airplane), 0)
        # Use modulo to handle case where we have more airplanes than colors
        color_index = airplane_index % len(self.trajectory_colors)
        # Get this airplane's trajectory color
        traj_color = self.trajectory_colors[color_index]
        
        # If we can use the airplane image
        if hasattr(self, 'use_airplane_image') and self.use_airplane_image and hasattr(self, 'airplane_image'):
            # Create sprite and apply color tint to match trajectory color
            sprite = pyglet.sprite.Sprite(self.airplane_image)
            
            # Apply color tint to match trajectory color
            sprite.color = traj_color
            
            # Scale the sprite (adjust this value to change size)
            scale_factor = 0.15
            sprite.scale = scale_factor
            
            # Position the sprite at the airplane's coordinates
            sprite.x = vector[0][0]
            sprite.y = vector[1][0]
            
            # Rotate the sprite to match the airplane's TRACK 
            sprite_rotation = airplane.track
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
            
            # Set fill color to match trajectory color but with some transparency
            filled_symbol.set_color_opacity(traj_color[0], traj_color[1], traj_color[2], 150)
                
            self.viewer.add_onetime(filled_symbol)

            aircraft_symbol_transform = rendering.Transform()
            aircraft_symbol_transform.set_rotation(math.radians(airplane.track))
            symbol.add_attr(aircraft_symbol_transform)
            filled_symbol.add_attr(aircraft_symbol_transform)

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
        label_offset = 20  # Pixels
        label_x = vector[0][0] + label_offset
        label_y = vector[1][0] + label_offset

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
            y=label_y - 80,
            width=50,
            height=8,
            fuel_percentage=airplane.fuel_remaining_pct,
        )
        self.viewer.add_onetime(fuel_gauge)
        
    def _render_all_aircraft_info_panel(self, airplanes: list[model.Airplane]):
        """
        Render aircraft parameters as text labels in the corner of the screen
        """
        # Position for the labels - bottom right corner with padding
        x_pos = self.viewer.width - 280  # Further right for more space
        y_pos = self.viewer.height - 10  # Top side with margin

        # Create labels for aircraft parameters
        title = Label("AIRCRAFT PARAMETERS", x_pos, y_pos)
        self.viewer.add_onetime(title)
        
        # Add a legend for the visualization - removed dotted line reference
        legend_y = y_pos - 30
        legend_title = Label("LEGEND:", x_pos, legend_y)
        legend_heading = Label("Blue Arrow: Aircraft Heading", x_pos, legend_y - 25)
        legend_track = Label("Green Arrow: Ground Track", x_pos, legend_y - 50)
        
        self.viewer.add_onetime(legend_title)
        self.viewer.add_onetime(legend_heading)
        self.viewer.add_onetime(legend_track)
        
        # Start aircraft details below the legend with more spacing
        panel_y = legend_y - 85  # Reduced spacing since we removed one legend item
        
        for airplane_index, airplane in enumerate(airplanes):
            
            # Get current MVA with error handling
            try:
                current_mva = self._airspace.get_mva_height(airplane.x, airplane.y)
                height_above_mva = airplane.h - current_mva
            except ValueError:
                height_above_mva = 0
            
            # Calculate crab angle (difference between heading and track)
            crab_angle = model.relative_angle(airplane.phi, airplane.track)
            
            # Create parameter strings
            params = [
                f"Flight: {airplane.name}",
                f"Position: ({airplane.x:.1f}, {airplane.y:.1f})",
                f"Altitude: {airplane.h} ft",
                f"Heading: {airplane.phi:.1f}°",
                f"Ground Track: {airplane.track:.1f}°",
                f"Crab Angle: {crab_angle:.1f}°",
                f"Airspeed: {airplane.v:.0f} knots",
                f"Ground Speed: {airplane.ground_speed:.0f} knots",
                f"Wind: {math.sqrt(airplane.wind_x**2 + airplane.wind_y**2):.1f} knots",
                f"Distance to FAF: {self._d_fafs[airplane_index]:.1f} nm",
                f"Height above MVA: {height_above_mva:.0f} ft",
                f"Fuel remaining: {airplane.fuel_remaining_pct:.1f}%"
            ]
            
            # Add parameter labels with more spacing between lines
            for i, param in enumerate(params):
                y = panel_y - 25 * i  # Increased spacing between lines
                param_label = Label(param, x_pos, y)
                self.viewer.add_onetime(param_label)

            # Add fuel gauge below parameters
            fuel_gauge = FuelGauge(
                x=x_pos,
                y=panel_y - 25 * (len(params) + 1),
                width=200,
                height=15,  # Taller gauge
                fuel_percentage=airplane.fuel_remaining_pct,
            )
            self.viewer.add_onetime(fuel_gauge)

            panel_y -= 25 * (len(params) + 3)  # Move down for next aircraft with more space
            
            
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
        Render the wind direction arrow on the screen
        """
        # airspace 
        # self._wind is a Wind object
        wind_field = self._wind.wind_field
        # at each point in airspace, draw a line in the direction of the wind based on self._wind.resolution, we can discretize
        discretization = int(self._wind.resolution)
        for x in range(0, len(wind_field), discretization):
            for y in range(0, len(wind_field[0]), discretization):
                wind_vector = wind_field[x][y]
                wind_speed = np.linalg.norm(wind_vector)
                if wind_speed > 0:
                    wind_vector /= wind_speed
                    wind_vector *= 0.5 * self._scale
                    vector = self._screen_vector(x, y)
                    arrow = rendering.Line(
                        (vector[0][0], vector[1][0]),
                        (vector[0][0] + wind_vector[0], vector[1][0] + wind_vector[1])
                    )
                    arrow.set_color_opacity(*ColorScheme.wind)
                    self.viewer.add_geom(arrow)
                    
                    # arrow base
                    size = 2
                    sqr = rendering.FilledPolygon([
                        (arrow.start[0] - size, arrow.start[1] - size),
                        (arrow.start[0] + size, arrow.start[1] - size),
                        (arrow.start[0] + size, arrow.start[1] + size),
                        (arrow.start[0] - size, arrow.start[1] + size)
                    ])
                    sqr.set_color_opacity(*ColorScheme.wind_arrow_base)

                    self.viewer.add_geom(sqr)
                    
                


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
        """
        def transform_world_to_screen(coords):
            return [((coord[0] - self._world_x_min) * self._scale + self._padding,
                     (coord[1] - self._world_y_min) * self._scale + self._padding) for coord in coords]

        for mva in self._mvas:
            coordinates = transform_world_to_screen(mva.area.exterior.coords)

            fill = rendering.FilledPolygon(coordinates)
            # based on height, choose color from ColorScheme.mva_height_colormap which is a matplotlib linear colormap
            norm_height_btn_0_1 = mva.height / MAX_MVA_HEIGHT_FOR_VISUALIZATION
            color = [int(x*255) for x in ColorScheme.mva_height_colormap(norm_height_btn_0_1)]
            fill.set_color_opacity(*color)
            self.viewer.add_geom(fill)

        for mva in self._mvas:
            coordinates = transform_world_to_screen(mva.area.exterior.coords)
            outline = rendering.PolyLine(coordinates, True)
            outline.set_color(*ColorScheme.mva)
            self.viewer.add_geom(outline)

        for mva in self._mvas:
            # add label on edge of the mva indicating its FL
            label_pos = transform_world_to_screen(mva.area.centroid.coords)[0]
            label = Label(f"FL{(mva.height // 100):03}", label_pos[0], label_pos[1], bold=False)
            label2 = Label(f"{mva.mva_type.value}", label_pos[0], label_pos[1] - 15, bold=False)
            self.viewer.add_geom(label)
            self.viewer.add_geom(label2)

    def _render_runway(self):
        """
        Renders the runway symbol onto the screen
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
            
            self._airplanes = []
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
            selected_entrypoints = self._scenario.entrypoints[:self._airplane_count]

            for i in range(self._airplane_count):
                # Use a different entry point for each aircraft
                entry_point = selected_entrypoints[i]
                
                # Place the airplane exactly at the entry point (no randomization)
                x = entry_point.x
                y = entry_point.y
                altitude = entry_point.levels[0] * 100  # Always use the first altitude
                phi = entry_point.phi
                
                # Create new airplane instances for the simulation
                self._airplanes.append(
                    model.Airplane(
                        self._sim_parameters,
                        f"FLT{i+1:03}",                                    # Flight identifier
                        x, y,                                       # Position
                        altitude,                                    # Altitude
                        phi,                            # Heading
                        250                                         # Initial speed
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
            history = airplane.position_history
            
            # Skip if history is too short
            if len(history) < 2:
                continue
                
            # Limit the trail length to prevent performance issues
            trail_points = history[-self.max_trail_length:]
            
            # Convert world coordinates to screen coordinates
            screen_points = []
            for x, y in trail_points:
                point = self._screen_vector(x, y)
                screen_points.append((point[0][0], point[1][0]))
            
            # Draw polyline connecting history points
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            trail = rendering.PolyLine(screen_points, False, linewidth=2)
            trail.set_color(*color)
            self.viewer.add_onetime(trail)

    def toggle_trajectories(self):
        """
        Toggle the display of aircraft trajectories
        
        Returns:
            Current state of trajectory display (True=enabled, False=disabled)
        """
        self.show_trajectories = not self.show_trajectories
        return self.show_trajectories