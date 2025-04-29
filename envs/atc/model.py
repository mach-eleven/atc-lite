# minimal-atc-rl/envs/atc/model.py
import math
import random
from typing import List

import numpy as np
import shapely.geometry as geom
import shapely.ops
from numba import jit  # Numba provides just-in-time compilation for faster execution

# Conversion constant: 1 nautical mile = 6076 feet. I know it's distance now, fkn Pranjal.
nautical_miles_to_feet = 6076  # ft/nm

from enum import Enum

import logging 
logger = logging.getLogger("train.model")
logger.setLevel(logging.INFO)


class MvaType(Enum):
    """Enumeration of different types of Minimum Vectoring Altitude areas."""
    GENERIC = "Generic"
    MOUNTAINOUS = "Mountainous" 
    WEATHER = "Weather"
    OCEANIC = "Oceanic"


# Function to get wind speed at a given location and altitude
def get_wind_speed(x, y, h, mva_type=None, badness=5, wind_dirn=270):
    """
    Get wind speed at a given location and altitude based on terrain type.
    
    Args:
        x, y: Position in nautical miles
        h: Altitude in feet
        mva_type: Type of Minimum Vectoring Altitude area (affects wind pattern)
        badness: Wind intensity factor (0-10)
        
    Returns:
        Tuple (wind_x, wind_y) in knots
    """
    # Scale factor based on badness parameter (0-10)
    badness_factor = badness / 5.0  # Scale to make 5 the "normal" intensity
    
    # Base wind - predominately from west (270°) - no randomization for determinism
    base_wind_speed = 5 * badness_factor
    
    # Wind increases with altitude (logarithmic increase)
    altitude_factor = 1.0 + 0.1 * math.log(max(1, h / 1000))
    
    # Deterministic positional variations using sine/cosine functions
    # These create a gradient across the airspace
    x_variation = math.sin(x * 0.1) * 2 * badness_factor
    y_variation = math.cos(y * 0.1) * 2 * badness_factor
    
    wind_speed = base_wind_speed * altitude_factor + x_variation 
    wind_direction = wind_dirn + y_variation # Default wind from west
    
    # print(wind_dirn, wind_direction)
    # Apply terrain-specific wind adjustments based on MVA type
    if mva_type:
        if mva_type == MvaType.MOUNTAINOUS:
            # Mountains cause updrafts, turbulence and flow distortion
            # Wind tends to flow around mountains and accelerate through passes
            mountain_effect = math.sin(x * 0.3 + y * 0.2) * 3.0 * badness_factor
            wind_speed += mountain_effect
            # Wind direction shifts more dramatically near mountains
            wind_direction += math.sin(x * 0.2) * 15 * badness_factor
            
            # Add vertical wind component (useful for future expansions)
            # Positive values near the upwind side, negative on the downwind side
            vertical_flow = math.sin(x * 0.5 + y * 0.3) * 5 * badness_factor
            
        elif mva_type == MvaType.OCEANIC:
            # Ocean winds tend to be steadier but can still have patterns
            # Reduced small-scale turbulence
            ocean_effect = math.sin(x * 0.05 + y * 0.05) * 1.5 * badness_factor
            wind_speed += ocean_effect
            # More consistent direction over ocean
            wind_direction += math.sin(y * 0.1) * 5 * badness_factor
            
        elif mva_type == MvaType.WEATHER:
            # Weather systems create more chaotic, stronger winds
            weather_effect = (math.sin(x * 0.25) * math.cos(y * 0.25)) * 4.0 * badness_factor
            wind_speed += weather_effect
            # Wind direction shifts more in weather systems
            wind_direction += math.sin(x * 0.1 + y * 0.1) * 20 * badness_factor
            
            # Simulate gusts with deterministic pseudo-random pattern
            gust_factor = math.sin(x * 0.7 + y * 0.9) * 3.0 * badness_factor
            wind_speed += max(0, gust_factor)
            
        elif mva_type == MvaType.GENERIC:
            # Generic terrain has moderate effects
            generic_effect = math.sin(x * 0.15 + y * 0.15) * 1.0 * badness_factor
            wind_speed += generic_effect
            wind_direction += math.sin(y * 0.05) * 10 * badness_factor
    
    # Ensure wind speed is non-negative
    wind_speed = max(0, wind_speed)
    
    # Convert to vector components (wind from direction)
    wind_direction_rad = math.radians(wind_direction)
    wind_x = -wind_speed * math.sin(wind_direction_rad)  # Negative because wind from this direction
    wind_y = -wind_speed * math.cos(wind_direction_rad)
    
    return (wind_x, wind_y)

class Airplane:
    def __init__(self, sim_parameters, name, x, y, h, phi, v, h_min=0, h_max=38000, v_min=100, v_max=300, starting_fuel=None):
        """
        Represents an aircraft in the simulation with its physical properties and constraints.
        
        :param sim_parameters: Object containing general simulation parameters
        :param name: Identifier for the flight/airplane (e.g., "FLT01")
        :param x: X-coordinate in nautical miles
        :param y: Y-coordinate in nautical miles
        :param h: Altitude in feet
        :param phi: Heading angle in degrees (0-360)
        :param v: Speed in knots
        :param v_min: Minimum allowed speed in knots
        :param v_max: Maximum allowed speed in knots
        :param h_min: Minimum allowed altitude in feet
        :param h_max: Maximum allowed altitude in feet
        :param starting_fuel: Initial fuel amount in kg (defaults to 10000 if None)
        """
        self.sim_parameters = sim_parameters
        self.name = name
        self.x = x
        self.y = y
        self.h = h
        # Validate that altitude is within allowed range
        if (h < h_min) or (h > h_max):
            raise ValueError("invalid altitude")
        self.v = v
        # Validate that speed is within allowed range
        if (v < v_min) or (v > v_max):
            raise ValueError("invalid velocity")
        self.phi = phi
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max
        # Maximum descent rate in feet per second
        self.h_dot_min = -41
        # Maximum climb rate in feet per second
        self.h_dot_max = 15
        # Maximum acceleration in knots per second
        self.a_max = 5
        # Maximum deceleration in knots per second
        self.a_min = -5
        # Maximum turn rate in degrees per second (right turn)
        self.phi_dot_max = 3
        # Maximum turn rate in degrees per second (left turn)
        self.phi_dot_min = -3
        # Stores previous positions for tracking/visualization
        self.position_history = []
        # Random identifier for this airplane instance
        self.id = random.randint(0, 32767)
        
        # NEW: Add fuel and mass properties
        self.empty_mass = 40000  # kg (aircraft without fuel)
        self.fuel_mass = starting_fuel if starting_fuel is not None else 10000  # kg (initial fuel)
        self.max_fuel = 10000    # kg (fuel capacity)
        
        # Fuel consumption rates (kg/s)
        self.cruise_consumption = 0.5   # Base fuel flow at cruise
        self.fuel_remaining_pct = 100.0  # Percentage of fuel remaining
        
        # Wind components (will be updated from function)
        self.wind_x = 0.0  # Wind x-component in knots
        self.wind_y = 0.0  # Wind y-component in knots
        self.headwind = 0.0  # Headwind component
        
        # Track vs heading (with wind)
        self.ground_speed = v  # Initially same as airspeed
        self.track = phi      # Initially same as heading
        
        # For tracking vertical speed
        self.prev_h = h
        self.vertical_speed = 0  # feet per minute
        
        # Flag to enable autopilot heading correction
        self.autopilot_enabled = True
        # Maximum rate at which autopilot can correct heading (degrees per second)
        self.autopilot_max_correction = 1.0

    def above_mva(self, mvas):
        """
        Checks if the aircraft is above the Minimum Vectoring Altitude (MVA) for its current position.
        
        :param mvas: List of MVA areas
        :return: True if aircraft is above the MVA, False otherwise
        :raises ValueError: If aircraft is outside the defined airspace
        """
        for mva in mvas:
            if mva.area.contains(geom.Point(self.x, self.y)):
                return self.h >= mva.height
        # If no MVA contains the aircraft's position, it's outside the airspace
        raise ValueError('Outside of airspace')

    def action_v(self, action_v):
        """
        Updates the aircraft's speed based on ATC command.
        
        The target speed will be constrained by the aircraft's performance limits:
        - Speed must be between v_min and v_max
        - Acceleration/deceleration cannot exceed a_max/a_min per timestep
        
        :param action_v: Target speed in knots
        :raises ValueError: If requested speed is outside allowed range
        """
        if action_v < self.v_min:
            raise ValueError("invalid speed")
        if action_v > self.v_max:
            raise ValueError("invalid speed")
            
        # Calculate requested speed change
        delta_v = action_v - self.v
        
        # Limit acceleration to aircraft performance constraints
        delta_v = min(delta_v, self.a_max * self.sim_parameters.timestep)
        
        # Limit deceleration to aircraft performance constraints
        delta_v = max(delta_v, self.a_min * self.sim_parameters.timestep)
        
        # Apply the constrained speed change
        self.v = self.v + delta_v

    def action_h(self, action_h):
        """
        Updates the aircraft's altitude based on ATC command.
        
        The target altitude will be constrained by the aircraft's performance limits:
        - Altitude must be between h_min and h_max
        - Climb/descent rate cannot exceed h_dot_max/h_dot_min per timestep
        
        :param action_h: Target altitude in feet
        :raises ValueError: If requested altitude is outside allowed range
        """
        if action_h < self.h_min:
            raise ValueError("invalid altitude")
        if action_h > self.h_max:
            raise ValueError("invalid altitude")
            
        # Calculate requested altitude change
        delta_h = action_h - self.h
        
        # Limit climb rate to aircraft performance constraints
        delta_h = min(delta_h, self.h_dot_max * self.sim_parameters.timestep)
        
        # Limit descent rate to aircraft performance constraints
        delta_h = max(delta_h, self.h_dot_min * self.sim_parameters.timestep)
        
        # Apply the constrained altitude change
        self.h = self.h + delta_h

    def action_phi(self, action_phi):
        """
        Updates the aircraft's heading based on ATC command.
        
        The turn rate will be constrained by the aircraft's performance limits:
        - Turn rate cannot exceed phi_dot_max/phi_dot_min per timestep
        
        :param action_phi: Target heading in degrees (0-360)
        """
        # Calculate the relative angle between current and target heading
        delta_phi = relative_angle(self.phi, action_phi)
        
        # Limit turn rate to aircraft performance constraints (right turn)
        delta_phi = min(delta_phi, self.phi_dot_max * self.sim_parameters.timestep)
        
        # Limit turn rate to aircraft performance constraints (left turn)
        delta_phi = max(delta_phi, self.phi_dot_min * self.sim_parameters.timestep)
        
        # Apply the constrained heading change, keeping within 0-360 range
        self.phi = (self.phi + delta_phi) % 360

    def update_wind(self, airspace=None, wind_badness=5, wind_dirn=270):
        """
        Update wind components based on current position and terrain type.
        
        Args:
            airspace: Airspace object containing MVAs
            wind_badness: Wind intensity factor (0-10)
        """
        mva_type = None
        
        # If we have airspace info, determine the MVA type at the current position
        if airspace:
            try:
                # Find which MVA we're in and get its type
                mva = airspace.find_mva(self.x, self.y)
                mva_type = mva.mva_type
            except ValueError:
                # Outside airspace, use generic wind
                mva_type = MvaType.GENERIC
        
        # Call the enhanced wind function with MVA type and badness
        wind_vector = get_wind_speed(self.x, self.y, self.h, mva_type, wind_badness, wind_dirn)
        self.wind_x = wind_vector[0]
        self.wind_y = wind_vector[1]
        
        # Recalculate ground speed and track
        self._calculate_ground_vector()

    def _calculate_ground_vector(self):
        """Calculate ground speed and track based on airspeed, heading, and wind."""
        # Convert heading to radians
        heading_rad = math.radians(self.phi)
        
        # Calculate true airspeed components
        air_x = self.v * math.sin(heading_rad)
        air_y = self.v * math.cos(heading_rad)
        
        # Calculate ground vector by adding wind components
        ground_x = air_x + self.wind_x
        ground_y = air_y + self.wind_y
        
        # Calculate new ground speed
        self.ground_speed = math.sqrt(ground_x**2 + ground_y**2)
        
        # Calculate new track
        self.track = (math.degrees(math.atan2(ground_x, ground_y)) + 360) % 360
        
        # Calculate headwind component for fuel calculations
        self.headwind = -self.wind_x * math.sin(heading_rad) - self.wind_y * math.cos(heading_rad)
        
        # logger.debug(f"{self.name}: Airspeed={self.v:.1f}kt, Heading={self.phi:.1f}°, Ground Speed={self.ground_speed:.1f}kt, Track={self.track:.1f}°, Headwind={self.headwind:.1f}kt")

    def _autopilot_heading_correction(self):
        """
        Apply an autopilot heading correction to align heading more with track.
        This is a simplified implementation of how real aircraft autopilots work.
        """
        if not self.autopilot_enabled:
            return
            
        # Calculate the difference between track and heading
        # This is the crab angle that needs to be maintained for the wind
        delta_angle = relative_angle(self.track, self.phi)
        
        # If the difference is very small, don't bother correcting
        if abs(delta_angle) < 0.5:
            return
            
        # Determine the direction to adjust heading (positive = right turn, negative = left turn)
        correction_sign = -1 if delta_angle > 0 else 1
        
        # Apply a gradual correction, limited by the maximum correction rate
        correction = min(abs(delta_angle) * 0.1, self.autopilot_max_correction * self.sim_parameters.timestep)
        correction *= correction_sign
        
        # Apply the heading correction
        self.phi = (self.phi + correction) % 360

    def update_fuel(self):
        """Update fuel quantity based on consumption."""
        # Calculate vertical speed in feet per minute
        self.vertical_speed = (self.h - self.prev_h) * (60 / self.sim_parameters.timestep)
        self.prev_h = self.h
        
        # Base consumption varies with speed
        speed_factor = (self.v - self.v_min) / (self.v_max - self.v_min)
        base_consumption = self.cruise_consumption * (0.8 + 0.4 * speed_factor**2)
        
        # Adjust for climb/descent
        climb_factor = 1.0
        if abs(self.vertical_speed) > 100:
            if self.vertical_speed > 0:
                # Climbing uses more fuel
                climb_factor = 2.0
            else:
                # Descending uses less fuel
                climb_factor = 0.7
        
        # Adjust for headwind (headwind increases fuel consumption)
        wind_factor = 1.0
        if abs(self.headwind) > 0:
            wind_factor = 1.0 + 0.2 * min(abs(self.headwind) / max(1.0, self.v), 0.5)
        
        # Calculate final fuel consumption
        consumption_rate = base_consumption * climb_factor * wind_factor
        
        # Calculate fuel burned this time step
        fuel_burned = consumption_rate * self.sim_parameters.timestep
        
        # Update fuel mass
        self.fuel_mass = max(0, self.fuel_mass - fuel_burned)
        self.fuel_remaining_pct = (self.fuel_mass / self.max_fuel) * 100
        

        # Return fuel status
        return self.fuel_mass > 0

    def step(self, airspace=None, wind_badness=5, wind_dirn=270):
        """Updates the aircraft's position based on wind-affected ground speed and track.
        
        Args:
            airspace: Airspace object containing MVAs for terrain-based wind effects
            wind_badness: How strong and turbulent the wind should be (0-10)
        
        Returns:
            Boolean indicating if the aircraft still has fuel
        """
        # Record the current position in history
        self.position_history.append((self.x, self.y))
        
        # Update wind at current position with terrain information
        self.update_wind(airspace, wind_badness, wind_dirn)
        
        # Apply heading correction to better align with track (simulates autopilot)
        self._autopilot_heading_correction()
        
        # Update fuel consumption
        has_fuel = self.update_fuel()
        
        # If out of fuel, reduce performance
        if not has_fuel:
            # Start descent if no fuel (simplified glide)
            self.h = max(0, self.h - 500 * self.sim_parameters.timestep)
            # Reduce speed (simplified glide)
            self.v = max(self.v_min, self.v * 0.98)
            logger.debug(f"WARNING: {self.name} is out of fuel! Gliding at {self.v:.1f}kt, descending at 500ft/min")
        
        # Use track and ground speed for position update instead of heading and airspeed
        track_rad = math.radians(self.track)
        
        # Calculate distance traveled this timestep
        distance = (self.ground_speed / 3600) * self.sim_parameters.timestep
        
        # Update position
        self.x += distance * math.sin(track_rad)
        self.y += distance * math.cos(track_rad)
        
        return has_fuel

class SimParameters:
    def __init__(self, timestep: float, precision: float = 0.5, reward_shaping: bool = True,
                 normalize_state: bool = True, discrete_action_space: bool = False):
        """
        Contains parameters that configure the simulation behavior.
        
        :param timestep: Simulation time increment in seconds
        :param precision: Epsilon value for floating-point comparisons
        :param reward_shaping: Whether to use reward shaping techniques for reinforcement learning
        :param normalize_state: Whether to normalize state values for reinforcement learning algorithms
        :param discrete_action_space: Whether to use discrete or continuous action space
        """
        self.timestep = timestep
        self.precision = precision
        self.reward_shaping = reward_shaping
        self.normalize_state = normalize_state
        self.discrete_action_space = discrete_action_space

class Corridor:
    def __init__(self, x: int, y: int, h: int, phi_from_runway: int):
        """
        Defines the approach corridor for an aircraft to land.
        This is a 3D space where aircraft must be positioned to properly approach the runway.
        
        :param x: X-coordinate of the runway threshold
        :param y: Y-coordinate of the runway threshold
        :param h: Altitude of the runway threshold
        :param phi_from_runway: Heading from the runway (opposite of landing direction)
        """
        self.x = x
        self.y = y
        self.h = h
        
        # Direction flying away from the runway
        self.phi_from_runway = phi_from_runway
        
        # Direction flying toward the runway (landing direction)
        self.phi_to_runway = (phi_from_runway + 180) % 360
        
        # Unit vector perpendicular to the runway direction
        self._faf_iaf_normal = np.dot(rot_matrix(self.phi_from_runway), np.array([[0], [1]]))

        # Distance from threshold to FAF (Final Approach Fix) in nautical miles
        faf_threshold_distance = 2
        
        # Angle defining the corridor width (degrees from centerline)
        faf_angle = 45
        self.faf_angle = faf_angle
        
        # Distance from FAF to IAF (Initial Approach Fix) in nautical miles
        faf_iaf_distance = 3
        
        # Calculate distance to corridor corners accounting for the corridor angle
        faf_iaf_distance_corner = faf_iaf_distance / math.cos(math.radians(faf_angle))
        
        # Calculate FAF position (vector from runway threshold along approach path)
        self.faf = np.array([[x], [y]]) + np.dot(rot_matrix(phi_from_runway),
                                                 np.array([[0], [faf_threshold_distance]]))
        
        # Calculate positions of corridor corners at FAF
        self.corner1 = np.dot(rot_matrix(faf_angle),
                              np.dot(rot_matrix(phi_from_runway), [[0], [faf_iaf_distance_corner]])) + self.faf
        self.corner2 = np.dot(rot_matrix(-faf_angle),
                              np.dot(rot_matrix(phi_from_runway), [[0], [faf_iaf_distance_corner]])) + self.faf
        
        # Define the horizontal corridor area at FAF
        self.corridor_horizontal = geom.Polygon([
            (self.faf[0][0], self.faf[1][0]), 
            (self.corner1[0][0], self.corner1[1][0]), 
            (self.corner2[0][0], self.corner2[1][0])
        ])
        
        # Calculate IAF position (extends further from FAF)
        self.iaf = np.array([[x], [y]]) + np.dot(rot_matrix(phi_from_runway),
                                                 np.array([[0], [faf_threshold_distance + faf_iaf_distance]]))
        
        # Define the two sides of the approach corridor
        self.corridor1 = geom.Polygon([
            (self.faf[0][0], self.faf[1][0]), 
            (self.corner1[0][0], self.corner1[1][0]), 
            (self.iaf[0][0], self.iaf[1][0])
        ])
        self.corridor2 = geom.Polygon([
            (self.faf[0][0], self.faf[1][0]), 
            (self.corner2[0][0], self.corner2[1][0]), 
            (self.iaf[0][0], self.iaf[1][0])
        ])

        # Convert polygon coordinates to arrays for faster checking with ray tracing
        self.corridor_horizontal_list = np.array(list(self.corridor_horizontal.exterior.coords))
        self.corridor1_list = np.array(list(self.corridor1.exterior.coords))
        self.corridor2_list = np.array(list(self.corridor2.exterior.coords))

    def inside_corridor(self, x, y, h, phi):
        """
        Checks if an aircraft at a given position and heading is properly positioned
        within the approach corridor.
        
        :param x: X-coordinate of the aircraft
        :param y: Y-coordinate of the aircraft
        :param h: Altitude of the aircraft in feet
        :param phi: Heading of the aircraft in degrees
        :return: True if aircraft is in the corridor, False otherwise
        """
        # First check if aircraft is horizontally within the corridor
        if not ray_tracing(x, y, self.corridor_horizontal_list):
            return False

        # Calculate the vertical constraints (glideslope)
        # Find the projection of aircraft position onto the approach path
        p = np.array([[x, y]])
        t = np.dot(p - np.transpose(self.faf), self._faf_iaf_normal)
        projection_on_faf_iaf = self.faf + t * self._faf_iaf_normal
        
        # Calculate maximum allowed altitude at current distance
        # Uses a 3-degree glideslope angle
        h_max_on_projection = np.linalg.norm(projection_on_faf_iaf - np.array([[self.x], [self.y]])) * \
                              math.tan(3 * math.pi / 180) * nautical_miles_to_feet + self.h

        # Check if aircraft is below the glideslope
        if not h <= h_max_on_projection:
            return False

        # Check if aircraft has the correct heading for the approach
        return self._inside_corridor_angle(x, y, phi)

    def _inside_corridor_angle(self, x, y, phi):
        """
        Checks if an aircraft has the correct heading based on which side of the corridor it's in.
        
        :param x: X-coordinate of the aircraft
        :param y: Y-coordinate of the aircraft
        :param phi: Heading of the aircraft in degrees
        :return: True if aircraft has a valid heading for its position, False otherwise
        """
        direction_correct = False
        to_runway = self.phi_to_runway
        
        # Calculate the angle between the aircraft's heading and the runway direction
        # This is complex geometry to determine acceptable approach angles
        beta = self.faf_angle - np.arccos(
            np.dot(
                np.transpose(np.dot(rot_matrix(to_runway), np.array([[0], [1]]))),
                np.dot(rot_matrix(phi), np.array([[0], [1]]))
            )
        )[0][0]
        min_angle = self.faf_angle - beta
        
        # Check if aircraft is in corridor1 (right side) with correct heading
        if ray_tracing(x, y, self.corridor1_list) and min_angle <= relative_angle(to_runway, phi) <= self.faf_angle:
            direction_correct = True
        # Check if aircraft is in corridor2 (left side) with correct heading    
        elif ray_tracing(x, y, self.corridor2_list) and min_angle <= relative_angle(phi, to_runway) <= self.faf_angle:
            direction_correct = True

        return direction_correct

class Runway:
    def __init__(self, x, y, h, phi):
        """
        Represents a runway in the simulation.
        
        :param x: X-coordinate of the runway threshold
        :param y: Y-coordinate of the runway threshold
        :param h: Altitude of the runway threshold in feet
        :param phi: Heading from the runway (opposite of landing direction)
        """
        self.x = x
        self.y = y
        self.h = h
        self.phi_orig = phi
        self.phi_from_runway = phi
        # Landing direction (opposite of phi_from_runway)
        self.phi_to_runway = (phi + 180) % 360
        # Create the approach corridor for this runway
        self.corridor = Corridor(x, y, h, phi)

    def inside_corridor(self, x: int, y: int, h: int, phi: int):
        """
        Checks if an aircraft at a given position and heading is properly positioned
        within this runway's approach corridor.
        
        :param x: X-coordinate of the aircraft
        :param y: Y-coordinate of the aircraft
        :param h: Altitude of the aircraft in feet
        :param phi: Heading of the aircraft in degrees
        :return: True if aircraft is in the corridor, False otherwise
        """
        return self.corridor.inside_corridor(x, y, h, phi)

class MinimumVectoringAltitude:
    def __init__(self, area: geom.Polygon, height: int, mva_type: MvaType = None):
        """
        Defines a Minimum Vectoring Altitude area - the lowest altitude
        an aircraft can safely fly in a specific region.
        
        :param area: Polygon defining the horizontal boundaries of the MVA
        :param height: Minimum safe altitude in feet for this area
        """
        self.area = area
        self.height = height
        # Convert to array for faster checking with ray tracing
        self.area_as_list = np.array(list(area.exterior.coords))
        # Extract bounding box for quicker preliminary position checks
        self.outer_bounds = area.bounds

        self.mva_type = mva_type if mva_type else MvaType.GENERIC

class Airspace:
    def __init__(self, mvas: List[MinimumVectoringAltitude], runway: Runway):
        """
        Represents the complete airspace with multiple MVA areas and a runway.
        
        :param mvas: List of MinimumVectoringAltitude objects defining safe altitudes
        :param runway: The runway object for this airspace
        """
        self.mvas = mvas
        self.runway = runway

    def find_mva(self, x, y):
        """
        Finds which MVA area contains the given coordinates.
        
        :param x: X-coordinate to check
        :param y: Y-coordinate to check
        :return: The MinimumVectoringAltitude object containing these coordinates
        :raises ValueError: If coordinates are outside all defined MVA areas
        """
        for mva in self.mvas:
            bounds = mva.outer_bounds
            # First do a quick check against the MVA's bounding box
            # bounds is a tuple with (minx, miny, maxx, maxy)
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                # Then do a precise check using ray tracing
                if ray_tracing(x, y, mva.area_as_list):
                    return mva
        # If no MVA contains the point, it's outside the airspace
        raise ValueError('Outside of airspace')

    def get_mva_height(self, x, y):
        point = geom.Point(x, y)
        for mva in self.mvas:
            # Debug print for inclusion check
            inside = mva.area.covers(point)
            if inside:
                return mva.height
        raise ValueError(f"Point ({x}, {y}) is outside the defined airspace!")

    def get_bounding_box(self):
        """
        Calculates the overall bounding box of the entire airspace.
        
        :return: Tuple with (minx, miny, maxx, maxy)
        """
        combined_poly = self.get_outline_polygon()
        return combined_poly.bounds

    def get_outline_polygon(self):
        """
        Combines all MVA areas into a single polygon representing the entire airspace.
        
        :return: A shapely Polygon object representing the combined airspace
        """
        polys = [mva.area for mva in self.mvas]
        logger.debug(f"Number of polygons: {len(polys)}")
        # check valid polygons
        index = 0
        for poly in polys:
            if not poly.is_valid:
                raise ValueError(f"Invalid polygon at index {index}: {poly}") 
            
            index += 1
        
        combined_poly = shapely.ops.unary_union(polys)
        return combined_poly

class Wind:
    """
    - Maintains a 2D grid of wind vectors covering a given bounding box.
    - Currently ignores altitude, so the wind is identical at all altitudes.
    - Generates wind vectors by placing up to 3 swirl "centers" in the domain
      (replace or augment with Perlin noise if you prefer).
    """

    def __init__(self, 
                 bounding_box,      # (min_x, max_x, min_y, max_y) - region in FAF-as-origin coordinates 
                                    # (can just be set as (max_x, min_x, 0, 0) acc to my understanding of our implementation)
                 resolution=1.0,    # this is the spacing between points in our wind grid
                 seed=0,            # for rng
                 num_centers=3,     # how many hotspots to create
                 swirl_scale=2.0):  # REDUCED from 10.0 to 5.0: strength of the swirl effect for each center
       
        self.min_x, self.max_x, self.min_y, self.max_y = bounding_box
        self.resolution = resolution

        # Number of grid cells in x / y directions
        self.width  = int((self.max_x - self.min_x) / self.resolution) + 5
        self.height = int((self.max_y - self.min_y) / self.resolution) + 5
        
        if self.width < 0 or self.height < 0:
            raise ValueError("Likely Invalid bounding box! Check min/max x/y values and resolution.")

        # 3D array: (width, height, 2) => store (wind_x, wind_y)
        self.wind_field = np.zeros((self.width, self.height, 2), dtype=float)

        # Pre-generate wind vectors
        self._generate_wind_field(seed=seed, num_centers=num_centers, swirl_scale=swirl_scale)

    def _generate_wind_field(self, seed=0, num_centers=3, swirl_scale=5.0):
        """
        Generate a swirl-based wind field by placing random swirl "centers"
        in the bounding box. Each center contributes a swirl vector to each grid cell. (might be negligible)
        not using perlin noise yet, but can be added later.
        """
        rng = np.random.default_rng(seed)

        # Each center is (center_x, center_y, direction)
        # 'direction' ±1 just flips swirl orientation
        swirl_centers = []
        for _ in range(num_centers):
            cx = rng.uniform(self.min_x, self.max_x)
            cy = rng.uniform(self.min_y, self.max_y)
            direction = rng.choice([-1, 1])  # swirl clockwise vs counterclockwise
            swirl_centers.append((cx, cy, direction))

        # Fill each grid cell with the sum of swirl vectors from all centers
        for i in range(self.width):
            for j in range(self.height):
                # Convert grid indices back to "world" coordinates
                world_x = self.min_x + i * self.resolution
                world_y = self.min_y + j * self.resolution

                vx_total, vy_total = 0.0, 0.0
                for (cx, cy, d) in swirl_centers:
                    dx = world_x - cx
                    dy = world_y - cy
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-6  # avoid div by zero

                    # A swirl vector can be computed as a tangent to the radial vector:
                    #   radial:    (dx, dy)
                    #   tangent:   (-dy, dx) or (dy, -dx)
                    # We'll pick (-dy, dx) and multiply by direction d to flip sign if needed
                    tx = -dy
                    ty =  dx

                    # The swirl magnitude falls off with distance. Tweak as you like:
                    swirl_strength = swirl_scale / dist

                    vx_total += d * tx * swirl_strength
                    vy_total += d * ty * swirl_strength

                self.wind_field[i, j, 0] = vx_total
                self.wind_field[i, j, 1] = vy_total

    def get_wind_speed(self, x, y, h=None):
        """
        Lookup the wind vector at (x,y). Ignores altitude (h).

        :return: (wind_x, wind_y) in knots (or whatever unit you used).
        """
        # Clamp (x,y) to bounding box so we don't go out of array bounds.
        x_clamped = max(self.min_x, min(self.max_x, x))
        y_clamped = max(self.min_y, min(self.max_y, y))

        # Find nearest grid cell
        i = int((x_clamped - self.min_x) / self.resolution)
        j = int((y_clamped - self.min_y) / self.resolution)

        # If you want bilinear interpolation, you'd do that here. For simplicity:
        wind_x = self.wind_field[i, j, 0]
        wind_y = self.wind_field[i, j, 1]

        return (wind_x, wind_y)

class EntryPoint:
    def __init__(self, x: float, y: float, phi: int, levels: List[int]):
        """
        Defines an entry point for aircraft entering the airspace.
        
        :param x: X-coordinate of the entry point
        :param y: Y-coordinate of the entry point
        :param phi: Initial heading for aircraft at this entry point
        :param levels: List of possible flight levels (in hundreds of feet) for aircraft at this entry point
        """
        self.x = x
        self.y = y
        self.phi = phi
        self.levels = levels

    def __iter__(self):
        return iter([(self.x, self.y), self.phi, self.levels])
    
    def __str__(self):
        return f"Entry(x={self.x}, y={self.y}, phi={self.phi}, levels={self.levels})"
    def __repr__(self):
        return f"Entry(x={self.x}, y={self.y}, phi={self.phi}, levels={self.levels})"
        
@jit(nopython=True)
def ray_tracing(x, y, poly):
    """
    Uses ray tracing algorithm to determine if a point is inside a polygon.
    JIT-compiled for performance. CLAUDE TOLD ME THIS, OKAY FINE. (lol - shivangi)
    
    :param x: X-coordinate of the point
    :param y: Y-coordinate of the point
    :param poly: Array of polygon vertices
    :return: True if point is inside polygon, False otherwise
    """
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

@jit(nopython=True)
def relative_angle(angle1, angle2):
    """
    Calculates the smallest angle between two headings.
    JIT-compiled for performance.
    
    :param angle1: First angle in degrees
    :param angle2: Second angle in degrees
    :return: Relative angle between -180 and 180 degrees
    """
    return (angle2 - angle1 + 180) % 360 - 180

@jit(nopython=True)
def rot_matrix(phi):
    """
    Creates a 2D rotation matrix for the given angle.
    JIT-compiled for performance.
    
    :param phi: Angle in degrees
    :return: 2x2 rotation matrix
    """
    phi = math.radians(phi)
    return np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])