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
            model.EntryPoint(54.0, 80.5, 50, [150]),  
            # Second plane - West approach 
            model.EntryPoint(10, 51, 270, [150])
        ]
        
        # Calculate paths by extending along same heading from these points
        curriculum_paths = []
        
        for base_entry in entry_points:
            entry_coords = base_entry.x, base_entry.y
            entry_heading = base_entry.phi
            
            # Calculate the heading from entry point to runway (in radians)
            dx = runway_threshold_coords[0] - entry_coords[0]
            dy = runway_threshold_coords[1] - entry_coords[1]
            heading_to_runway = math.atan2(dy, dx)  # in radians, corrected order of dy, dx
            
            # Calculate distance between entry point and runway
            distance = math.sqrt(dx**2 + dy**2)
            
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
        
        # Ensure planes are at least 5 nautical miles apart (9260 meters)
        min_separation = 5.0  # 5 nautical miles in meters
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
                # Calculate adjustment vector
                adjustment_needed = min_separation - distance
                if distance > 0:  # Avoid division by zero
                    # Unit vector from entry2 to entry1
                    ux = dx / distance
                    uy = dy / distance
                    
                    # Apply half of the adjustment to each point in opposite directions
                    half_adjustment = adjustment_needed / 2
                    
                    # Adjusted positions
                    entry1_adjusted = model.EntryPoint(
                        entry1.x + ux * half_adjustment,
                        entry1.y + uy * half_adjustment,
                        entry1.phi,
                        entry1.levels
                    )
                    
                    entry2_adjusted = model.EntryPoint(
                        entry2.x - ux * half_adjustment,
                        entry2.y - uy * half_adjustment,
                        entry2.phi,
                        entry2.levels
                    )
                    
                    # Check if the adjusted points are within the airspace bounds
                    minx, miny, maxx, maxy = self.airspace.get_bounding_box()
                    
                    # If either point is outside the airspace, use the original points but log a warning
                    if (entry1_adjusted.x < minx or entry1_adjusted.x > maxx or 
                        entry1_adjusted.y < miny or entry1_adjusted.y > maxy or
                        entry2_adjusted.x < minx or entry2_adjusted.x > maxx or
                        entry2_adjusted.y < miny or entry2_adjusted.y > maxy):
                        logger.warning(f"Could not maintain 5NM separation for curriculum stage {i} "
                                      f"without moving points outside airspace. Distance: {distance/1852:.1f}NM")
                        curriculum_entries.append([entry1, entry2])
                    else:
                        # Use adjusted points
                        curriculum_entries.append([entry1_adjusted, entry2_adjusted])
                        
                        # Log the adjustment
                        logger.info(f"Adjusted entry points for stage {i} to maintain 5NM separation. "
                                   f"Original distance: {distance/1852:.1f}NM, New distance: {min_separation/1852:.1f}NM")
                else:
                    # If points are at the same location (unlikely), offset one arbitrarily
                    logger.warning(f"Entry points for stage {i} are at the same location. Applying arbitrary offset.")
                    entry2_adjusted = model.EntryPoint(
                        entry2.x + min_separation,
                        entry2.y,
                        entry2.phi,
                        entry2.levels
                    )
                    curriculum_entries.append([entry1, entry2_adjusted])
            else:
                # Already sufficiently separated
                curriculum_entries.append([entry1, entry2])
                
        # Return curriculum points as pairs of entry points
        return curriculum_entries
