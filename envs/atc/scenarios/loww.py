import math
from typing import List

import random
import numpy as np

from .. import model
from .scenarios import Scenario

import shapely.geometry as shape
from math import ceil

import logging 
logger = logging.getLogger("train.scenarios")
logger.setLevel(logging.INFO)

MvaType = model.MvaType

class LOWW(Scenario):
    def __init__(self, random_entrypoints=False, entry_point=None):
        super().__init__()

        self.mvas = [
            model.MinimumVectoringAltitude(shape.Polygon([
                (48.43, 2.09),
                (39.36, 4.22),
                (27.26, 20.01),
                (54.03, 12.95),
                (48.43, 2.09)
            ]), 4800),
            model.MinimumVectoringAltitude(shape.Polygon([
                (27.26, 20.01),
                (26.37, 21.35),
                (29.73, 26.39),
                (28.83, 31.09),
                (34.32, 25.55),
                (46.08, 22.36),
                (42.47, 16),
                (27.26, 20.01)
            ]), 3700, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (26.37, 21.35),
                (13.15, 38.60),
                (22.0, 36.13),
                (22.0, 30.65),
                (29.73, 26.39),
                (26.37, 21.35)
            ]), 5700, MvaType.MOUNTAINOUS),
            model.MinimumVectoringAltitude(shape.Polygon([
                (29.73, 26.39),
                (22.0, 30.65),
                (22.0, 36.13),
                (13.15, 38.60),
                (8, 45.68),
                (18.75, 44.98),
                (28.83, 31.09),
                (29.73, 26.39)
            ]), 4600, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (28.83, 31.09),
                (18.75, 44.98),
                (22.0, 45.68),
                (26.37, 43.08),
                (28.83, 31.09)
            ]), 4100, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (28.83, 31.09),
                (28.83, 33.45),
                (31.29, 34.12),
                (29.73, 41.29),
                (26.9, 40.47),
                (28.83, 31.09)
            ]), 4000, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (22.0, 45.68),
                (18.75, 44.98),
                (8, 45.68),
                (4.08, 50.25),
                (15.73, 76.12),
                (29.73, 80.71),
                (56.16, 82.05),
                (58.51, 69.84),
                (42.94, 71.97),
                (22.56, 65.36),
                (16.17, 50.25),
                (23.23, 49.01),
                (22.0, 45.68)
            ]), 3500, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (46.08, 22.36),
                (34.32, 25.55),
                (31.5, 28.4),
                (36.22, 35.46),
                (44.46, 31.76),
                (46.08, 22.36)
            ]), 3000, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (31.5, 28.4),
                (28.83, 31.09),
                (28.83, 33.45),
                (31.29, 34.12),
                (29.73, 41.29),
                (26.9, 40.47),
                (26.37, 43.08),
                (22.0, 45.68),
                (23.23, 49.01),
                (31.29, 48.01),
                (30.17, 45.71),
                (32.19, 44.98),
                (35.14, 41.62),
                (36.22, 42.29),
                (37.56, 36.69),
                (36.22, 35.46),
                (31.5, 28.4)
            ]), 3500, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (35.14, 41.62),
                (32.19, 44.98),
                (30.17, 45.71),
                (31.29, 48.01),
                (23.23, 49.01),
                (16.17, 50.25),
                (22.56, 65.36),
                (36.58, 69.91),
                (39.47, 60.55),
                (35.73, 59.13),
                (36.22, 56.18),
                (38.46, 53.72),
                (34.32, 45.68),
                (35.14, 41.62)
            ]), 3200, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (46.08, 22.36),
                (44.95, 28.91),
                (53.5, 31.43),
                (57.97, 41.89),
                (47.17, 55.97),
                (40.75, 53.72),
                (38.46, 53.72),
                (36.22, 56.18),
                (35.73, 59.13),
                (39.47, 60.55),
                (36.58, 69.91),
                (42.94, 71.97),
                (58.51, 69.84),
                (54.78, 60.01),
                (68.15, 38.6),
                (66.34, 36.85),
                (65.53, 30.62),
                (62.92, 29.97),
                (66.58, 20.58),
                (52.88, 18.68),
                (51.64, 21.35),
                (46.08, 22.36)
            ]), 2700, MvaType.OCEANIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (44.95, 28.91),
                (44.46, 31.76),
                (36.22, 35.46),
                (37.56, 36.69),
                (36.22, 42.29),
                (35.14, 41.62),
                (34.32, 45.68),
                (38.46, 53.72),
                (40.75, 53.72),
                (47.17, 55.97),
                (57.97, 41.89),
                (53.5, 31.43),
                (44.95, 28.91)
            ]), 2600, MvaType.OCEANIC)]

        self.runway = model.Runway(45.16, 43.26, 586, 160)

        self.airspace = model.Airspace(self.mvas, self.runway)

        if entry_point is not None:
            # For curriculum learning with a specific entry point
            self.entrypoints = [entry_point]
        elif random_entrypoints:
            self.entrypoints = [
                model.EntryPoint(10, 51, 90, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(17, 74.6, 120, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(19.0, 34.0, 45, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(29.8, 79.4, 170, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(54.0, 80.5, 230, [140, 160, 180, 200, 220, 240]),
                model.EntryPoint(53.0, 60.0, 260, [140, 160, 180, 200, 220, 240]),
                model.EntryPoint(66.0, 39.0, 290, [140, 160, 180, 200, 220]),
                model.EntryPoint(64.4, 22.0, 320, [140, 160, 180, 200, 220]),
                model.EntryPoint(46.0, 7.0, 320, [140, 160, 180, 200, 220, 240, 260])
            ]
        else:
            # Default entry points for the scenario
            self.entrypoints = [
                model.EntryPoint(10, 51, 90, [150]),
                model.EntryPoint(54.0, 80.5, 230, [150])  # Second entry point for two-airplane scenario
            ]
        
        minx, miny, maxx, maxy = self.airspace.get_bounding_box()
        self.wind = model.Wind((ceil(minx), ceil(maxx), ceil(miny), ceil(maxy)))

    def generate_last_bound_entry_points(self, num_entrypoints=5):
        """
        Generate 5 entry points for randomised training in the environment. They are sampled from all points within the MVA region.
        """

        # sample num_entrypoints points from the region, not the bounding box

        # Get the bounding box of the airspace
        minx, miny, maxx, maxy = self.airspace.get_bounding_box()
        # Get the MVA polygons
        mva_polygons = [mva.area for mva in self.mvas]
        # Create a list to store the entry points
        entry_points = []

        # potential_
        # Generate random points within the bounding box
        for _ in range(num_entrypoints):
            # Generate a random point within the bounding box
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            # Check if the point is inside any of the MVA polygons
            for polygon in mva_polygons:
                if polygon.contains(shape.Point(x, y)):
                    # If it is, create an entry point and add it to the list
                    entry_points.append(model.EntryPoint(x, y, random.choice([50, 100, 80, 200, 180]), [290]))
                    break
        # If we don't have enough entry points, generate more
        while len(entry_points) < num_entrypoints:
            # Generate a random point within the bounding box
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            # Check if the point is inside any of the MVA polygons
            for polygon in mva_polygons:
                if polygon.contains(shape.Point(x, y)):
                    # If it is, create an entry point and add it to the list
                    entry_points.append(model.EntryPoint(x, y, random.choice([50, 100, 80, 200, 180]), [290]))
                    break
        
        # If we still don't have enough entry points, just return what we have
        if len(entry_points) < num_entrypoints:
            logger.warning(f"Only generated {len(entry_points)} entry points, not enough for {num_entrypoints}")
        # Return the entry points
        return entry_points[:num_entrypoints]
    


    def generate_aircraft(self, count, bounds):
        """Generate aircraft with safer initial positions"""
        import numpy as np
        from envs.atc.model import Aircraft
        
        # Make aircraft start well within bounds
        padding = 10000  # meters from edge
        min_x, max_x = bounds[0] + padding, bounds[1] - padding
        min_y, max_y = bounds[2] + padding, bounds[3] - padding
        
        # Safe altitude range
        min_alt, max_alt = 3000, 8000  # meters
        
        aircraft = []
        for i in range(count):
            # Start in a safer region
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            # Start with altitude that gives room to maneuver
            altitude = np.random.uniform(min_alt, max_alt)
            # Random heading, but start pointing inward if near edges
            heading = np.random.uniform(0, 360)
            
            # Adjust heading to point more toward center if near edge
            center_x = (bounds[0] + bounds[1]) / 2
            center_y = (bounds[2] + bounds[3]) / 2
            
            # Simple vector toward center
            dx = center_x - x
            dy = center_y - y
            center_angle = np.degrees(np.arctan2(dy, dx)) % 360
            
            # Blend between random and center-pointing heading
            edge_proximity = min(
                (x - min_x) / padding,
                (max_x - x) / padding,
                (y - min_y) / padding,
                (max_y - y) / padding
            )
            # Closer to edge = more influence from center heading
            blend_factor = 1 - min(1, edge_proximity)
            heading = heading * (1-blend_factor) + center_angle * blend_factor
            
            aircraft.append(Aircraft(f"FLT{i+1:03d}", x, y, altitude, heading))
        
        return aircraft

    def generate_curriculum_entrypoints(self, num_entrypoints: int) -> List[model.EntryPoint]:
        """
        Generate curriculum entry points for training with the LOWW scenario.
        Entry points maintain the same heading toward the runway but vary in distance.
        
        For a two-airplane scenario, we generate two sets of entry points with 
        different approach paths to avoid collisions.
        """
        # We have two planes with different entry points
        # Create paths from each entry point to runway, maintaining heading
        runway_threshold_coords = self.runway.x, self.runway.y
        
        # Calculate runway approach direction (opposite of runway heading)
        approach_direction = (self.runway.phi_from_runway + 180) % 360
        
        # Define two distinct entry point positions with headings pointing to runway
        entry_points = [
            # First plane - Northeast approach
            model.EntryPoint(54.0, 80.5, 225, [150]),  
            # Second plane - West approach 
            model.EntryPoint(10, 51, 135, [150]) 
        ]
        
        # Calculate paths by extending along same heading from these points
        curriculum_paths = []
        
        # Store approach vectors for both entry points (for later adjustment)
        approach_vectors = []
        
        for base_entry in entry_points:
            entry_coords = base_entry.x, base_entry.y
            entry_heading = base_entry.phi
            
            # Calculate the heading from runway to entry point (in radians)
            dx = entry_coords[0] - runway_threshold_coords[0]
            dy = entry_coords[1] - runway_threshold_coords[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Store the unit vector from runway to entry point (for path adjustments)
            if distance > 0:
                approach_vectors.append((dx/distance, dy/distance))
            else:
                approach_vectors.append((0, 0)) # Failsafe
            
            # Generate curriculum entry points along this path
            path_points = []
            step_size = distance / (num_entrypoints + 1)
            
            for i in range(num_entrypoints):
                # Calculate position along the line from runway to entry point
                ratio = i / (num_entrypoints - 1) if num_entrypoints > 1 else 0  # 0 to 1
                # We want to start closer to runway and move outward
                new_x = runway_threshold_coords[0] + dx * ratio
                new_y = runway_threshold_coords[1] + dy * ratio
                
                # Create new entry point with same heading
                new_entry = model.EntryPoint(new_x, new_y, entry_heading, base_entry.levels)
                path_points.append(new_entry)
            
            curriculum_paths.append(path_points)
        
        # Ensure planes are at least 5 nautical miles 
        min_separation = 5  # dont change this
        curriculum_entries = []
        
        # Check and adjust entry points to maintain minimum separation
        for i in range(min(len(curriculum_paths[0]), len(curriculum_paths[1]))):
            entry1 = curriculum_paths[0][i]
            entry2 = curriculum_paths[1][i]
            
            # Calculate distance between the two entry points
            dx = entry1.x - entry2.x
            dy = entry1.y - entry2.y
            distance = math.sqrt(dx**2 + dy**2)
            
            # If too close, adjust positions
            if distance < min_separation:
                # Calculate additional distance needed along each approach path
                adjustment_needed = min_separation - distance
                
                # We'll move both aircraft further out along their approach paths
                # This maintains their heading to the runway while ensuring separation
                
                # Get the approach vectors (unit vectors from runway to entry point)
                approach_vector1 = approach_vectors[0]
                approach_vector2 = approach_vectors[1]
                
                # Calculate how far to move each aircraft
                # We'll move them both by the same amount to maintain relative positions
                adjustment_distance = adjustment_needed * 0.75  # Move a bit more to ensure separation
                
                # Move both entry points outward along their approach paths
                entry1_adjusted = model.EntryPoint(
                    entry1.x + approach_vector1[0] * adjustment_distance,
                    entry1.y + approach_vector1[1] * adjustment_distance,
                    entry1.phi,
                    entry1.levels
                )
                
                entry2_adjusted = model.EntryPoint(
                    entry2.x + approach_vector2[0] * adjustment_distance,
                    entry2.y + approach_vector2[1] * adjustment_distance,
                    entry2.phi,
                    entry2.levels
                )
                
                # Check if the adjusted points are within the airspace bounds
                minx, miny, maxx, maxy = self.airspace.get_bounding_box()
                
                # Verify new points are within bounds and properly separated
                within_bounds = (
                    entry1_adjusted.x >= minx and entry1_adjusted.x <= maxx and
                    entry1_adjusted.y >= miny and entry1_adjusted.y <= maxy and
                    entry2_adjusted.x >= minx and entry2_adjusted.x <= maxx and
                    entry2_adjusted.y >= miny and entry2_adjusted.y <= maxy
                )
                
                # Calculate new distance between adjusted points
                new_dx = entry1_adjusted.x - entry2_adjusted.x
                new_dy = entry1_adjusted.y - entry2_adjusted.y
                new_distance = math.sqrt(new_dx**2 + new_dy**2)
                
                if within_bounds and new_distance >= min_separation:
                    # Use adjusted points
                    curriculum_entries.append([entry1_adjusted, entry2_adjusted])
                    
                    # Log the adjustment
                    logger.info(f"Adjusted entry points for stage {i} by moving along approach paths. "
                               f"Original distance: {distance/1852:.1f}NM, New distance: {new_distance/1852:.1f}NM")
                else:
                    # If adjustment didn't work properly, use original strategy as fallback
                    logger.warning(f"Could not adjust along approach paths for stage {i}, trying alternate strategy")
                    
                    # Move planes away from each other perpendicular to their connecting line
                    # Calculate perpendicular vectors to the line connecting the two aircraft
                    if distance > 0:
                        # Unit vector perpendicular to the line connecting the two aircraft
                        perp_ux = -dy / distance
                        perp_uy = dx / distance
                        
                        # Apply perpendicular adjustment
                        half_adjustment = adjustment_needed / 1.5  # Adjust for safety margin
                        
                        entry1_adjusted = model.EntryPoint(
                            entry1.x + perp_ux * half_adjustment,
                            entry1.y + perp_uy * half_adjustment,
                            entry1.phi,
                            entry1.levels
                        )
                        
                        entry2_adjusted = model.EntryPoint(
                            entry2.x - perp_ux * half_adjustment,
                            entry2.y - perp_uy * half_adjustment,
                            entry2.phi,
                            entry2.levels
                        )
                        
                        # Check if adjusted points are within bounds
                        if (entry1_adjusted.x >= minx and entry1_adjusted.x <= maxx and
                            entry1_adjusted.y >= miny and entry1_adjusted.y <= maxy and
                            entry2_adjusted.x >= minx and entry2_adjusted.x <= maxx and
                            entry2_adjusted.y >= miny and entry2_adjusted.y <= maxy):
                            
                            curriculum_entries.append([entry1_adjusted, entry2_adjusted])
                            logger.info(f"Adjusted entry points for stage {i} using perpendicular movement. "
                                      f"New distance: {min_separation/1852:.1f}NM")
                        else:
                            logger.warning(f"Could not maintain separation for curriculum stage {i} "
                                         f"without moving points outside airspace. Using original points.")
                            curriculum_entries.append([entry1, entry2])
                    else:
                        # Points at same location (unlikely), just use original
                        logger.warning(f"Entry points for stage {i} are at the same location. Using original.")
                        curriculum_entries.append([entry1, entry2])
            else:
                # Already sufficiently separated
                curriculum_entries.append([entry1, entry2])
                
        # Return curriculum points as pairs of entry points
        return curriculum_entries

    def generate_curriculum_entrypoint_but_many(self, num_entrypoints: int, n_per_circle: int = 5) -> list[list[model.EntryPoint]]:
        """
        Lets say we have a circle of entry points around the runway.
        Based on num_entrypoints, we will generate N circles. The circles are distributed 
        between runway_threshold_coords and the entry point.
        The entry points are distributed evenly on the circle, and there are n_per_circle entry points per circle.

        The entry points are generated in a way that they are not too close to each other.

        If n_per_circle is 5, we will have 5 entry points on the circle. But actually, each entry_point will have 2 versions, with 2 different headings.
        So we will have 10 entry points on the circle.

        This function is only made for 1 airplane scenario, so it needs to return only a set of entry points for a singular plane.

        So, it returns a list of lists. Each list inside the list is a set of entry points for a single airplane on a circle. This is so that any point 
        can then be randomly selected from the list of entry points.
        """
        # Get runway coordinates
        runway_threshold_coords = self.runway.x, self.runway.y
        
        # Calculate approach direction (opposite of runway heading)
        approach_direction = (self.runway.phi_from_runway + 180) % 360
        
        # Define a reference point for distance calculation (using one of the default entry points)
        reference_entry = model.EntryPoint(54.0, 80.5, 50, [150])

        ref_x, ref_y = reference_entry.x, reference_entry.y
        
        # Calculate maximum distance from runway to reference point
        dx_max = ref_x - runway_threshold_coords[0]
        dy_max = ref_y - runway_threshold_coords[1]
        max_distance = math.sqrt(dx_max**2 + dy_max**2)
        
        # Default flight level if we can't determine MVA
        default_altitude = 15000  # feet - corresponds to FL150
        default_flight_level = 150  # Flight level 150 = 15,000 feet
        
        # Create circles with increasing radii from runway
        circles = []
        for i in range(num_entrypoints):
            # Calculate radius for this circle (increasing from runway)
            # Start with a small radius and increase gradually
            radius_ratio = (i + 1) / num_entrypoints
            circle_radius = max_distance * radius_ratio
            
            # Generate entry points around this circle
            circle_entries = []
            for j in range(n_per_circle):
                # Calculate angle for this point (evenly distributed around circle)
                angle_degrees = j * (360 / n_per_circle)
                angle_radians = math.radians(angle_degrees)
                
                # Calculate position on circle
                x = runway_threshold_coords[0] + circle_radius * math.cos(angle_radians)
                y = runway_threshold_coords[1] + circle_radius * math.sin(angle_radians)
                
                # Calculate heading toward runway
                dx_to_runway = runway_threshold_coords[0] - x
                dy_to_runway = runway_threshold_coords[1] - y
                heading_to_runway = (math.degrees(math.atan2(dy_to_runway, dx_to_runway)) + 90) % 360
                
                # Get MVA at this position to ensure we start at a safe altitude
                try:
                    # Find the minimum vectoring altitude at this position
                    mva_height = self.airspace.get_mva_height(x, y)
                    # Add a safety margin (e.g., 500 feet above MVA)
                    safe_altitude = mva_height + 50
                    # Calculate flight level (divide by 100)
                    safe_flight_level = math.ceil(safe_altitude / 100) * 10
                except ValueError:
                    # If point is outside defined airspace, use default altitude
                    logger.warning(f"Entry point ({x}, {y}) is outside airspace, using default altitude")
                    safe_flight_level = default_flight_level
                
                if safe_flight_level > 380:
                    safe_flight_level = 350
                # Create entry point with heading toward runway and safe altitude
                runway_heading_entry = model.EntryPoint(x, y, heading_to_runway, [safe_flight_level])
                
                # Create second version with a slightly offset heading (e.g., +30 degrees)
                offset_angle = 30
                offset_heading = (heading_to_runway + offset_angle) % 360
                offset_heading_entry = model.EntryPoint(x, y, offset_heading, [safe_flight_level])
                
                # Add both entry points to the circle
                circle_entries.append(runway_heading_entry)
                circle_entries.append(offset_heading_entry)
            
            circles.append(circle_entries)
        # logger.info([x.levels for x in [entry for circle in circles for entry in circle]])
        return circles

    def get_high_safe_entrypoint(self):
        # User-specified entrypoint coordinates and heading
        x, y, heading = 29.009196288983958, 77.00541385328415, 270
        safe_flight_level = 295
        return model.EntryPoint(x, y, heading, [safe_flight_level])

class ModifiedLOWW(LOWW):
    def __init__(self, random_entrypoints=False, entry_point=None):
        super().__init__(random_entrypoints=random_entrypoints, entry_point=entry_point)
        # Modify the MVAs for the
        self.mvas = [
            model.MinimumVectoringAltitude(shape.Polygon([
                (48.43, 2.09),
                (39.36, 4.22),
                (27.26, 20.01),
                (54.03, 12.95),
                (48.43, 2.09)
            ]), 14800, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (27.26, 20.01),
                (26.37, 21.35),
                (29.73, 26.39),
                (28.83, 31.09),
                (34.32, 25.55),
                (46.08, 22.36),
                (42.47, 16),
                (27.26, 20.01)
            ]), 9700, MvaType.OCEANIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (26.37, 21.35),
                (13.15, 38.60),
                (22.0, 36.13),
                (22.0, 30.65),
                (29.73, 26.39),
                (26.37, 21.35)
            ]), 15700, MvaType.MOUNTAINOUS),
            model.MinimumVectoringAltitude(shape.Polygon([
                (29.73, 26.39),
                (22.0, 30.65),
                (22.0, 36.13),
                (13.15, 38.60),
                (8, 45.68),
                (18.75, 44.98),
                (28.83, 31.09),
                (29.73, 26.39)
            ]), 32600, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (28.83, 31.09),
                (18.75, 44.98),
                (22.0, 45.68),
                (26.37, 43.08),
                (28.83, 31.09)
            ]), 4100, MvaType.WEATHER),
            model.MinimumVectoringAltitude(shape.Polygon([
                (28.83, 31.09),
                (28.83, 33.45),
                (31.29, 34.12),
                (29.73, 41.29),
                (26.9, 40.47),
                (28.83, 31.09)
            ]), 3000, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (22.0, 45.68),
                (18.75, 44.98),
                (8, 45.68),
                (4.08, 50.25),
                (15.73, 76.12),
                (29.73, 80.71),
                (56.16, 82.05),
                (58.51, 69.84),
                (42.94, 71.97),
                (22.56, 65.36),
                (16.17, 50.25),
                (23.23, 49.01),
                (22.0, 45.68)
            ]), 23500, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (46.08, 22.36),
                (34.32, 25.55),
                (31.5, 28.4),
                (36.22, 35.46),
                (44.46, 31.76),
                (46.08, 22.36)
            ]), 3800, MvaType.GENERIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (31.5, 28.4),
                (28.83, 31.09),
                (28.83, 33.45),
                (31.29, 34.12),
                (29.73, 41.29),
                (26.9, 40.47),
                (26.37, 43.08),
                (22.0, 45.68),
                (23.23, 49.01),
                (31.29, 48.01),
                (30.17, 45.71),
                (32.19, 44.98),
                (35.14, 41.62),
                (36.22, 42.29),
                (37.56, 36.69),
                (36.22, 35.46),
                (31.5, 28.4)
            ]), 3500, MvaType.OCEANIC),
            model.MinimumVectoringAltitude(shape.Polygon([
                (35.14, 41.62),
                (32.19, 44.98),
                (30.17, 45.71),
                (31.29, 48.01),
                (23.23, 49.01),
                (16.17, 50.25),
                (22.56, 65.36),
                (36.58, 69.91),
                (39.47, 60.55),
                (35.73, 59.13),
                (36.22, 56.18),
                (38.46, 53.72),
                (34.32, 45.68),
                (35.14, 41.62)
            ]), 18000, MvaType.MOUNTAINOUS),
            model.MinimumVectoringAltitude(shape.Polygon([
                (46.08, 22.36),
                (44.95, 28.91),
                (53.5, 31.43),
                (57.97, 41.89),
                (47.17, 55.97),
                (40.75, 53.72),
                (38.46, 53.72),
                (36.22, 56.18),
                (35.73, 59.13),
                (39.47, 60.55),
                (36.58, 69.91),
                (42.94, 71.97),
                (58.51, 69.84),
                (54.78, 60.01),
                (68.15, 38.6),
                (66.34, 36.85),
                (65.53, 30.62),
                (62.92, 29.97),
                (66.58, 20.58),
                (52.88, 18.68),
                (51.64, 21.35),
                (46.08, 22.36)
            ]), 20000, MvaType.MOUNTAINOUS),
            model.MinimumVectoringAltitude(shape.Polygon([
                (44.95, 28.91),
                (44.46, 31.76),
                (36.22, 35.46),
                (37.56, 36.69),
                (36.22, 42.29),
                (35.14, 41.62),
                (34.32, 45.68),
                (38.46, 53.72),
                (40.75, 53.72),
                (47.17, 55.97),
                (57.97, 41.89),
                (53.5, 31.43),
                (44.95, 28.91)
            ]), 2500, MvaType.GENERIC)]

        self.runway = model.Runway(45.16, 43.26, 586, 160)

        self.airspace = model.Airspace(self.mvas, self.runway)
        self.entrypoints = self.generate_last_bound_entry_points(num_entrypoints=1)
        self.entrypoints = [self.get_high_safe_entrypoint()]
        self.entrypoints[0].phi = 270
        
        logger.info(f"Generated entry point: {self.entrypoints[0].x}, {self.entrypoints[0].y}, {self.entrypoints[0].phi}")