# minimal-atc-rl/envs/atc/atc_gym.py
import math
import random

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from numba import jit

from .rendering import Label, FuelGauge
from .themes import ColorScheme
from . import model
from . import scenarios
from . import my_rendering as rendering
import pyglet

# TODO: Airplane should have d_faf, etc. stored in it not the array system we have now


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

    def __init__(self, airplane_count=1, sim_parameters=model.SimParameters(1), scenario=None):
        """
        Initialize the ATC gym environment
        
        Args:
            sim_parameters: Simulation parameters like timestep size
            scenario: The airspace scenario to use (runways, MVAs, etc.)
        """

        self._airplane_count = airplane_count

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

        # Define the observation space normalization parameters
        self.normalization_state_min = np.array([
            *[self._world_x_min for _ in self._airplanes],          # Minimum x position
            *[self._world_y_min for _ in self._airplanes],          # Minimum y position
            *[0 for _ in self._airplanes],                          # Minimum altitude
            *[0 for _ in self._airplanes],                          # Minimum heading
            *[airplane.v_min for airplane in self._airplanes],       # Minimum speed
            *[0 for _ in self._airplanes],                          # Minimum height above MVA
            *[0 for _ in self._on_gp_altitudes],                          # Minimum on-glidepath altitude
            *[0 for _ in self._d_fafs],                          # Minimum distance to FAF
            *[-180 for _ in self._phi_rel_fafs],                       # Minimum relative angle to FAF
            *[-180 for _ in self._phi_rel_runways],                        # Minimum relative angle to runway
            *[0 for _ in self._airplanes]                           # NEW: Minimum fuel percentage
        ], dtype=np.float32)


        self.normalization_state_max = np.array([
            *[world_x_length for _ in self._airplanes],          # x position range in nautical miles
            *[world_y_length for _ in self._airplanes],          # y position range in nautical miles
            *[airplane.h_max for airplane in self._airplanes],   # maximum altitude range in feet
            *[360 for _ in self._airplanes],                     # heading range in degrees
            *[airplane.v_max - airplane.v_min for airplane in self._airplanes],       # speed range in knots
            *[airplane.h_max for airplane in self._airplanes],                        # maximum height above MVA in feet
            *[airplane.h_max for airplane in self._airplanes],                  # maximum on glidepath altitude in feet
            *[self._world_max_distance for _ in self._d_fafs],                        # maximum distance to FAF in nautical miles
            *[360 for _ in self._phi_rel_fafs],                   # maximum relative angle to FAF in degrees
            *[360 for _ in self._phi_rel_runways],                 # maximum relative angle to runway in degrees
            *[100 for _ in self._airplanes]                     # NEW: Maximum fuel percentage
        ], dtype=np.float32)
        
        # Define the observation space: x, y, h, phi, v, h-mva, on_gp, d_faf, phi_rel_faf, phi_rel_runway, fuel
        # All values normalized to [-1, 1]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.normalization_state_min),))
        self.reward_range = (-3000.0, 23000.0)  # Define the min/max possible rewards

        print("Number of airplanes: ", len(self._airplanes))
        print("Observation space: ", len(self.normalization_state_min))
        print("Action space: ", len(self.normalization_action_offset))

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
        reward = -0.05 * self._sim_parameters.timestep

        dones = [False] * len(self._airplanes)
        out_of_fuel = [False] * len(self._airplanes)
        
        # Apply each action component and accumulate rewards
        c = 0
        for airplane in self._airplanes:
            action = action_array[c * 3: (c + 1) * 3] # for airplane 0: 0-2, for airplane 1: 3-5
            last_action = self.last_action[c * 3: (c + 1) * 3]

            rewa, updated_last_action = self._action_with_reward(airplane.action_v, action, last_action, 0)    # Speed
            reward += rewa
            self.last_action[(c * 3) + 0] = updated_last_action[0]
            self.last_action[(c * 3) + 1] = updated_last_action[1]
            self.last_action[(c * 3) + 2] = updated_last_action[2]
            rewa, updated_last_action = self._action_with_reward(airplane.action_h, action, last_action, 1)    # Altitude
            reward += rewa
            self.last_action[(c * 3) + 0] = updated_last_action[0]
            self.last_action[(c * 3) + 1] = updated_last_action[1]
            self.last_action[(c * 3) + 2] = updated_last_action[2]
            rewa, updated_last_action = self._action_with_reward(airplane.action_phi, action, last_action, 2)  # Heading
            reward += rewa
            self.last_action[(c * 3) + 0] = updated_last_action[0]
            self.last_action[(c * 3) + 1] = updated_last_action[1]
            self.last_action[(c * 3) + 2] = updated_last_action[2]


            # Update the airplane position based on its current state
            has_fuel = airplane.step()
            
            # Check if airplane is out of fuel
            if not has_fuel:
                out_of_fuel[c] = True
                # Small penalty for running out of fuel
                reward -= 10
                print(f"Aircraft {airplane.name} is out of fuel!")

            # Check if airplane is above the MVA (minimum vectoring altitude)
            try:
                mva = self._airspace.get_mva_height(airplane.x, airplane.y)

                if airplane.h < mva:
                    # Airplane has descended below minimum safe altitude - failure
                    self._win_buffer.append(0)
                    reward = -200  # Large negative reward
                    dones[c] = True
                    print(f"Aircraft {airplane.name} has descended below MVA!")
            except ValueError:
                # Airplane has left the defined airspace - failure
                self._win_buffer.append(0)
                dones[c] = True
                reward = -50  # Negative reward
                mva = 0  # Dummy MVA value for the final state
                print(f"Aircraft {airplane.name} has left the airspace!")

            # Check if airplane has successfully reached the final approach corridor
            if self._runway.inside_corridor(airplane.x, airplane.y, airplane.h, airplane.phi):
                # GAME WON! Aircraft successfully guided to final approach
                self._win_buffer.append(1)
                # Large positive reward plus bonus for finishing quickly and fuel efficiency
                fuel_bonus = airplane.fuel_remaining_pct / 10  # Up to 10 points for fuel efficiency
                reward = 10000 + max((self.timestep_limit - self.timesteps) * 5, 0) + fuel_bonus
                dones[c] = True
                print(f"Aircraft {airplane.name} has successfully reached the approach corridor!")

            # Apply additional reward shaping to guide learning
            if self._sim_parameters.reward_shaping:
                # Reward for being in a good approach position
                app_position_reward = self._reward_approach_position(
                    self._d_fafs[c], self._runway.phi_to_runway,
                    self._phi_rel_fafs[c], self._world_max_distance
                )
                reward += app_position_reward
                
                # Reward for correct approach angle
                reward += self._reward_approach_angle(
                    self._runway.phi_to_runway,
                    self._phi_rel_fafs[c], airplane.phi, app_position_reward
                )
                
                # Reward for being on the correct glideslope
                reward += self._reward_glideslope(
                    airplane.h, self._on_gp_altitudes[c], app_position_reward
                )
                
                # NEW: Small reward for fuel efficiency
                reward += 0.01 * airplane.fuel_remaining_pct

            c += 1

        # NEW: Check if all aircraft are out of fuel
        if all(out_of_fuel):
            self.done = True
            reward = -100  # Penalty for all aircraft running out of fuel
            print("All aircraft are out of fuel!")
        else:
            self.done = all(dones)

        # Check if time limit exceeded
        if self.timesteps > self.timestep_limit:
            reward = -200  # Negative reward for timeout
            self.done = True
            print("Time limit exceeded!")

        # Get the current observation
        state = self._get_obs(mva)
        self.state = state

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
    # have to remove this
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
        except ValueError:
            # Invalid action (outside permissible range)
            print(f"Warning invalid action: {action_to_take} for index: {index}")
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
        for airplane in self._airplanes:
            self._render_airplane(airplane) # Aircraft symbols
        self._render_all_aircraft_info_panel(self._airplanes) # Aircraft information panel
        self._render_reward()      # Reward information

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

        # also add an arrow to show the heading
        # Create arrow to show heading from center of diamond
        arrow_length = render_size * 6  # Fixed length
        arrow_vector = np.array([[arrow_length], [0]])  # Start with horizontal vector
        # Rotate arrow by airplane heading
        rotated_arrow = np.dot(model.rot_matrix(airplane.phi), arrow_vector)
        # Draw arrow from center of diamond
        arrow = rendering.Line(
            (vector[0][0], vector[1][0]),  # Start at diamond center
            (vector[0][0] + rotated_arrow[0][0], vector[1][0] + rotated_arrow[1][0])  # End at rotated point
        )
        arrow.set_color(*ColorScheme.airplane)
        self.viewer.add_onetime(arrow)
        
        # Create the airplane symbol
        symbol = rendering.PolyLine([
            (corner_top_right[0][0], corner_top_right[1][0]),
            (corner_bottom_right[0][0], corner_bottom_right[1][0]),
            (corner_bottom_left[0][0], corner_bottom_left[1][0]),
            (corner_top_left[0][0], corner_top_left[1][0])
        ], True)
        
        # Color the airplane based on fuel level
        if airplane.fuel_remaining_pct > 66:
            symbol.set_color(*ColorScheme.airplane)  # Normal color
        elif airplane.fuel_remaining_pct > 33:
            symbol.set_color(240, 240, 0)  # Yellow for medium fuel
        else:
            symbol.set_color(240, 0, 0)   # Red for low fuel
            
        self.viewer.add_onetime(symbol)

        # Create labels with aircraft information
        label_pos = np.dot(model.rot_matrix(135), 2 * corner_vector) + vector
        render_altitude = round(airplane.h)  # Flight level
        render_speed = round(airplane.v)      # Speed in knots
        render_text = f"{render_altitude}' {render_speed}kt"
        
        # Add aircraft callsign and details labels
        label_name = Label(airplane.name, x=label_pos[0][0], y=label_pos[1][0])
        label_details = Label(render_text, x=label_pos[0][0], y=label_pos[1][0] - 15)
        self.viewer.add_onetime(label_name)
        self.viewer.add_onetime(label_details)
        
        # Add a small fuel gauge near the airplane
        # DEBUG Only
        # fuel_gauge = FuelGauge(
        #     x=label_pos[0][0], 
        #     y=label_pos[1][0] - 40,
        #     width=30,
        #     height=5,
        #     fuel_percentage=airplane.fuel_remaining_pct,
        # )
        # self.viewer.add_onetime(fuel_gauge)

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
                
    def _render_all_aircraft_info_panel(self, airplanes: list[model.Airplane]):
        """
        Render aircraft parameters as text labels in the corner of the screen
        """
        # Position for the labels - bottom right corner with padding
        x_pos = self.viewer.width - 230  # Right side with margin
        y_pos = self.viewer.height - 10  # Top side with margin

        # Create labels for aircraft parameters
        title = Label("AIRCRAFT PARAMETERS", x_pos, y_pos, bold=True)
        self.viewer.add_onetime(title)
        
        for airplane_index, airplane in enumerate(airplanes):
            
            # Get current MVA with error handling
            try:
                current_mva = self._airspace.get_mva_height(airplane.x, airplane.y)
                height_above_mva = airplane.h - current_mva
            except ValueError:
                height_above_mva = 0
            
            # Create parameter strings
            params = [
                f"Flight: {airplane.name}",
                f"Position: ({airplane.x:.1f}, {airplane.y:.1f})",
                f"Altitude: {airplane.h:.0f} ft",
                f"Heading: {airplane.phi:.1f}°",
                f"Speed: {airplane.v:.0f} knots",
                f"Ground Speed: {airplane.ground_speed:.0f} knots",
                f"Track: {airplane.track:.1f}°",
                f"Distance to FAF: {self._d_fafs[airplane_index]:.1f} nm",
                f"Height above MVA: {height_above_mva:.0f} ft",
                f"Fuel remaining: {airplane.fuel_remaining_pct:.1f}%"
            ]
            
            # Add parameter labels starting from the bottom
            for i, param in enumerate(params):
                y = y_pos - 20 * (i + 1)
                param_label = Label(param, x_pos, y, bold=False)
                self.viewer.add_onetime(param_label)

            # Add fuel gauge below parameters
            fuel_gauge = FuelGauge(
                x=x_pos,
                y=y_pos - 20 * (len(params) + 2),
                width=200,
                height=15,
                fuel_percentage=airplane.fuel_remaining_pct,
            )
            self.viewer.add_onetime(fuel_gauge)

            y_pos -= 20 * (len(params) + 3)  # Move up for next aircraft with space for fuel gauge
        
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

        for mva in self._mvas:
            # add label on edge of the mva indicating its FL
            label_pos = transform_world_to_screen(mva.area.centroid.coords)[0]
            label = Label(f"{mva.height}ft", label_pos[0], label_pos[1], bold=False)
            self.viewer.add_geom(label)

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

        # Calculate the center of the airspace for a safer starting position
        center_x = (self._world_x_min + self._world_x_max) / 2
        center_y = (self._world_y_min + self._world_y_max) / 2
        
        self._airplanes = []
        self._d_fafs = []
        self._phi_rel_fafs = []
        self._phi_rel_runways = []
        self._on_gp_altitudes = []

        for i in range(self._airplane_count):
            
            # Choose a random entry point from the scenario
            entry_point = random.choice(self._scenario.entrypoints)
            
            # Place the airplane at a position 25% of the way from the center to the entry point
            # This ensures it starts well within the airspace
            x = center_x + 0.25 * (entry_point.x - center_x)
            y = center_y + 0.25 * (entry_point.y - center_y)
            
            # Create new airplane instances for the simulation
            self._airplanes.append(
                model.Airplane(
                    self._sim_parameters,
                    f"FLT{i+1:03}",                                    # Flight identifier
                    x, y,                                       # Position
                    random.choice(entry_point.levels) * 100,    # Altitude
                    entry_point.phi,                            # Heading
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