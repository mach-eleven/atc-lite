# minimal-atc-rl/envs/atc/atc_gym.py
import math
import random

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from numba import jit

from .rendering import Label
from .themes import ColorScheme
from . import model
from . import scenarios
from . import my_rendering as rendering
import pyglet


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
        'render_modes': ['human', 'rgb_array'],
        "render_fps": 50
    }

    def __init__(self, sim_parameters=model.SimParameters(1), scenario=None):
        """
        Initialize the ATC gym environment
        
        Args:
            sim_parameters: Simulation parameters like timestep size
            scenario: The airspace scenario to use (runways, MVAs, etc.)
        """
        self.render_mode = 'rgb_array'
        
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
        self.reset()
        self.viewer = None

        # Action space normalization parameters
        self.normalization_action_offset = np.array([self._airplane.v_min, 0, 0])

        # Configure action space (discrete or continuous)
        if sim_parameters.discrete_action_space:
            # For discrete actions, define the scaling factors
            self.normalization_action_factor = np.array([10, 100, 1])

            # action space structure: v (speed), h (altitude), phi (heading)
            self.action_space = gym.spaces.MultiDiscrete([
                int((self._airplane.v_max - self._airplane.v_min) / 10),  # Speed buckets
                int(self._airplane.h_max / 100),                          # Altitude in hundreds of feet
                360                                                        # Heading in degrees
            ])
        else:
            # For continuous actions, define the scaling factors based on aircraft limits
            self.normalization_action_factor = np.array([
                self._airplane.v_max - self._airplane.v_min,  # Speed range
                self._airplane.h_max,                         # Altitude range
                360                                           # Heading range (degrees)
            ])

            # Continuous action space between -1 and 1 for each dimension
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1])
            )

        # Thresholds for what counts as a significant action change
        self._action_discriminator = [5, 50, 0.5]  # For speed, altitude, heading
        self.last_action = [0, 0, 0]

        # Define the observation space normalization parameters
        self.normalization_state_min = np.array([
            self._world_x_min,          # Minimum x position
            self._world_y_min,          # Minimum y position
            0,                          # Minimum altitude
            0,                          # Minimum heading
            self._airplane.v_min,       # Minimum speed
            0,                          # Minimum height above MVA
            0,                          # Minimum on-glidepath altitude
            0,                          # Minimum distance to FAF
            -180,                       # Minimum relative angle to FAF
            -180                        # Minimum relative angle to runway
        ], dtype=np.float32)
        
        self.normalization_state_max = np.array([
            world_x_length,                          # x position range in nautical miles
            world_y_length,                          # y position range in nautical miles
            self._airplane.h_max,                    # maximum altitude range in feet
            360,                                     # heading range in degrees
            self._airplane.v_max - self._airplane.v_min,  # speed range in knots
            self._airplane.h_max,                    # maximum height above MVA in feet
            self._airplane.h_max,                    # maximum on glidepath altitude in feet
            self._world_max_distance,                # maximum distance to FAF in nautical miles
            360,                                     # maximum relative angle to FAF in degrees
            360                                      # maximum relative angle to runway in degrees
        ], dtype=np.float32)

        # Define the observation space: x, y, h, phi, v, h-mva, on_gp, d_faf, phi_rel_faf, phi_rel_runway
        # All values normalized to [-1, 1]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,))
        self.reward_range = (-3000.0, 23000.0)  # Define the min/max possible rewards

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

    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Action vector with [speed, altitude, heading] commands
            
        Returns:
            (state, reward, done, truncated, info) tuple
        """
        self.timesteps += 1
        self.done = False
        # Small negative reward per timestep to encourage efficiency
        reward = -0.05 * self._sim_parameters.timestep
        
        # Apply each action component and accumulate rewards
        reward += self._action_with_reward(self._airplane.action_v, action, 0)    # Speed
        reward += self._action_with_reward(self._airplane.action_h, action, 1)    # Altitude
        reward += self._action_with_reward(self._airplane.action_phi, action, 2)  # Heading

        # Update the airplane position based on its current state
        self._airplane.step()

        # Check if airplane is above the MVA (minimum vectoring altitude)
        try:
            mva = self._airspace.get_mva_height(self._airplane.x, self._airplane.y)

            if self._airplane.h < mva:
                # Airplane has descended below minimum safe altitude - failure
                self._win_buffer.append(0)
                reward = -200  # Large negative reward
                self.done = True
        except ValueError:
            # Airplane has left the defined airspace - failure
            self._win_buffer.append(0)
            self.done = True
            reward = -50  # Negative reward
            mva = 0  # Dummy MVA value for the final state

        # Check if airplane has successfully reached the final approach corridor
        if self._runway.inside_corridor(self._airplane.x, self._airplane.y, self._airplane.h, self._airplane.phi):
            # GAME WON! Aircraft successfully guided to final approach
            self._win_buffer.append(1)
            # Large positive reward plus bonus for finishing quickly
            reward = 10000 + max((self.timestep_limit - self.timesteps) * 5, 0)
            self.done = True

        # Check if time limit exceeded
        if self.timesteps > self.timestep_limit:
            reward = -200  # Negative reward for timeout
            self.done = True

        # Get the current observation
        state = self._get_obs(mva)
        self.state = state

        # Apply additional reward shaping to guide learning
        if self._sim_parameters.reward_shaping:
            # Reward for being in a good approach position
            app_position_reward = self._reward_approach_position(
                self._d_faf, self._runway.phi_to_runway,
                self._phi_rel_faf, self._world_max_distance
            )
            reward += app_position_reward
            
            # Reward for correct approach angle
            reward += self._reward_approach_angle(
                self._runway.phi_to_runway,
                self._phi_rel_faf, self._airplane.phi, app_position_reward
            )
            
            # Reward for being on the correct glideslope
            reward += self._reward_glideslope(
                self._airplane.h, self._on_gp_altitude, app_position_reward
            )

        # Normalize state if configured to do so
        if self._sim_parameters.normalize_state:
            state = (state - self.normalization_state_min - 0.5 * self.normalization_state_max) \
                    / (0.5 * self.normalization_state_max)

        # Update tracking metrics
        self._update_metrics(reward)

        # Gymnasium requires truncated flag (false in this implementation)
        truncated = False

        # Return the step result tuple
        return state, reward, self.done, truncated, {"original_state": self.state}

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
        # Reward for distance (higher when closer to FAF)
        reward_faf = sigmoid_distance_func(d_faf, world_max_dist)
        # Reward for alignment with approach course
        reward_app_angle = (abs(model.relative_angle(phi_to_runway, phi_rel_to_faf)) / 180.0) ** 1.5
        # Combine rewards with scaling factor
        return reward_faf * reward_app_angle * 0.8

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
        # Reward for being close to the correct glideslope altitude
        altitude_diff_factor = sigmoid_distance_func(abs(h - on_gp_altitude), 36000)
        # Scale by position factor (more important when closer to approach)
        return altitude_diff_factor * position_factor * 0.8

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
        # Calculate vector from aircraft to FAF
        to_faf_x = self._runway.corridor.faf[0][0] - self._airplane.x
        to_faf_y = self._runway.corridor.faf[1][0] - self._airplane.y
        
        # Calculate relative angle to runway heading
        phi_rel_runway = self._calculate_phi_rel_runway(self._runway.phi_to_runway, self._airplane.phi)
        
        # Calculate distance to FAF
        self._d_faf = self.euclidean_dist(to_faf_x, to_faf_y)
        
        # Calculate angle to FAF
        self._phi_rel_faf = self._calculate_phi_rel_faf(to_faf_x, to_faf_y)
        
        # Calculate the target altitude for current position on glidepath
        self._on_gp_altitude = self._calculate_on_gp_altitude(self._d_faf, self._faf_mva)

        # Create the full observation state vector
        state = np.array([
            self._airplane.x,              # Aircraft x position
            self._airplane.y,              # Aircraft y position
            self._airplane.h,              # Aircraft altitude
            self._airplane.phi,            # Aircraft heading
            self._airplane.v,              # Aircraft speed
            self._airplane.h - mva,        # Height above minimum safe altitude
            self._on_gp_altitude,          # Target glidepath altitude
            self._d_faf,                   # Distance to final approach fix
            self._phi_rel_faf,             # Relative angle to final approach fix
            phi_rel_runway                 # Relative angle to runway heading
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

    def _action_with_reward(self, func, action, index):
        """
        Apply an action component and calculate the resulting reward
        
        Args:
            func: Action application function
            action: Action vector
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
            if not abs(action_to_take - self.last_action[index]) < self._action_discriminator[index]:
                self.actions_taken += 1
            self.last_action[index] = action_to_take
        except ValueError:
            # Invalid action (outside permissible range)
            print(f"Warning invalid action: {action_to_take} for index: {index}")
            reward -= 1.0  # Penalty for invalid action

        return reward

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

        # Choose a random entry point from the scenario
        entry_point = random.choice(self._scenario.entrypoints)
        
        # Calculate the center of the airspace for a safer starting position
        center_x = (self._world_x_min + self._world_x_max) / 2
        center_y = (self._world_y_min + self._world_y_max) / 2
        
        # Place the airplane at a position 25% of the way from the center to the entry point
        # This ensures it starts well within the airspace
        x = center_x + 0.25 * (entry_point.x - center_x)
        y = center_y + 0.25 * (entry_point.y - center_y)
        
        # Create a new airplane with a moderate initial altitude
        self._airplane = model.Airplane(
            self._sim_parameters, 
            "FLT01",                                    # Flight identifier
            x, y,                                       # Position
            random.choice(entry_point.levels) * 100,    # Altitude
            entry_point.phi,                            # Heading 
            250                                         # Initial speed
        )

        # Reset state and tracking variables
        self.state = self._get_obs(0)
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

    def render(self, mode='rgb_array'):
        """
        Render the current state of the environment
        
        Args:
            mode: Either "human" for direct screen rendering or "rgb_array"
            
        Returns:
            Rendered frame
        """
        # Initialize viewer if not already created
        if self.viewer is None:
            self._padding = 10
            
            # Use a larger fixed size for better visibility
            screen_width = 1600
            screen_height = 1200

            # Calculate dimensions based on world size
            world_size_x = self._world_x_max - self._world_x_min
            world_size_y = self._world_y_max - self._world_y_min

            # Calculate the scaling factor to fit the world to the screen
            self._scale = min(
                (screen_width - 2 * self._padding) / world_size_x,
                (screen_height - 2 * self._padding) / world_size_y
            )
            
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

        # Render dynamic elements
        self._render_airplane(self._airplane)  # Aircraft
        self._render_reward()      # Reward information
        self._render_aircraft_panel()  # Aircraft parameter panel

        # Return the rendered frame
        return self.viewer.render(mode == 'rgb_array')
    
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
        Render the airplane symbol and information
        
        Args:
            airplane: The aircraft to render
        """
        # Create diamond shape for aircraft symbol
        render_size = 4
        vector = self._screen_vector(airplane.x, airplane.y)
        corner_vector = np.array([[0], [render_size]])
        corner_top_right = np.dot(model.rot_matrix(45), corner_vector) + vector
        corner_bottom_right = np.dot(model.rot_matrix(135), corner_vector) + vector
        corner_bottom_left = np.dot(model.rot_matrix(225), corner_vector) + vector
        corner_top_left = np.dot(model.rot_matrix(315), corner_vector) + vector

        # Create the airplane symbol
        symbol = rendering.PolyLine([
            (corner_top_right[0][0], corner_top_right[1][0]),
            (corner_bottom_right[0][0], corner_bottom_right[1][0]),
            (corner_bottom_left[0][0], corner_bottom_left[1][0]),
            (corner_top_left[0][0], corner_top_left[1][0])
        ], True)
        symbol.set_color(*ColorScheme.airplane)
        self.viewer.add_onetime(symbol)

        # Create labels with aircraft information
        label_pos = np.dot(model.rot_matrix(135), 2 * corner_vector) + vector
        render_altitude = round(airplane.h / 100)  # Flight level (hundreds of feet)
        render_speed = round(airplane.v / 10)      # Speed in tens of knots
        render_text = f"{render_altitude}  {render_speed}"
        
        # Add aircraft callsign and details labels
        label_name = Label(airplane.name, x=label_pos[0][0], y=label_pos[1][0])
        label_details = Label(render_text, x=label_pos[0][0], y=label_pos[1][0] - 15)
        self.viewer.add_onetime(label_name)
        self.viewer.add_onetime(label_details)

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
                
    def _render_aircraft_panel(self):
        """
        Render aircraft parameters as text labels in the corner of the screen
        """
        # Position for the labels - bottom right corner with padding
        x_pos = self.viewer.width - 230  # Right side with margin
        y_pos = 140  # Bottom with some margin
        
        # Create labels for aircraft parameters
        title = Label("AIRCRAFT PARAMETERS", x_pos, y_pos, bold=True)
        self.viewer.add_onetime(title)
        
        # Get current MVA with error handling
        try:
            current_mva = self._airspace.get_mva_height(self._airplane.x, self._airplane.y)
            height_above_mva = self._airplane.h - current_mva
        except ValueError:
            height_above_mva = 0
        
        # Create parameter strings
        params = [
            f"Position: ({self._airplane.x:.1f}, {self._airplane.y:.1f})",
            f"Altitude: {self._airplane.h:.0f} ft",
            f"Heading: {self._airplane.phi:.1f}°",
            f"Speed: {self._airplane.v:.0f} knots",
            f"Distance to FAF: {self._d_faf:.1f} nm",
            f"Height above MVA: {height_above_mva:.0f} ft"
        ]
        
        # Add parameter labels starting from the bottom
        for i, param in enumerate(params):
            y = y_pos - 20 * (i + 1)
            param_label = Label(param, x_pos, y, bold=False)
            self.viewer.add_onetime(param_label)
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
            fill.set_color(*ColorScheme.background_active)
            self.viewer.add_geom(fill)

        for mva in self._mvas:
            coordinates = transform_world_to_screen(mva.area.exterior.coords)

            outline = rendering.PolyLine(coordinates, True)
            outline.set_color(*ColorScheme.mva)
            self.viewer.add_geom(outline)

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
        
        runway_line = rendering.Line(start_point, end_point)
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