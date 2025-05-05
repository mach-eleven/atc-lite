import shapely.geometry as shape
from .. import model
from .scenarios import Scenario
import math
import logging
import random
from typing import List

logger = logging.getLogger("train.scenarios")
logger.setLevel(logging.INFO)

class MvaGoAroundScenario(Scenario):
    """
    Scenario where the aircraft must go around a high MVA region to reach the runway.
    The MVA is placed directly between the entry point and the runway, forcing a detour.
    """
    def __init__(self, entry_points: List[model.EntryPoint] = None):
        super().__init__()
        # Central high MVA obstacle
        mva_obstacle = model.MinimumVectoringAltitude(
            shape.Polygon([
                (20, 16), (25, 16), (25, 24), (20, 24), (20, 16)
            ]), 180, model.MvaType.MOUNTAINOUS
        )
        # Low MVAs covering the rest of the airspace
        mva_low = model.MinimumVectoringAltitude(
            shape.Polygon([
                (0, 0), (40, 0), (40, 40), (0, 40), (0, 0)
            ]), 100
        )
        mva_low2 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (0, 30), (10, 30), (10, 40), (0, 40), (0, 30)
            ]), 200
        )
        mva_low3 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (30, 0), (40, 0), (40, 10), (30, 10), (30, 0)
            ]), 250 # 250
        )
        mva_low4 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (15, 30), (25, 30), (25, 40), (15, 40), (15, 30)
            ]), 250 # 250
        )
        # Additional low MVAs for more coverage
        mva_low5 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (0, 15), (10, 15), (10, 25), (0, 25), (0, 15)
            ]), 150 # 150
        )
        mva_low6 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (30, 30), (40, 30), (40, 40), (30, 40), (30, 30)
            ]), 120 # 120
        )
        mva_low7 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (0, 0), (10, 0), (10, 10), (0, 10), (0, 0)
            ]), 130 # 130
        )
        mva_low8 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (30, 15), (40, 15), (40, 25), (30, 25), (30, 15)
            ]), 170
        )
        mva_low9 = model.MinimumVectoringAltitude(
            shape.Polygon([
                (15, 0), (25, 0), (25, 10), (15, 10), (15, 0)
            ]), 900 # 140
        )
        # mva_low10 = model.MinimumVectoringAltitude(
        #     shape.Polygon([
        #         (15, 15), (25, 15), (25, 25), (15, 25), (15, 15)
        #     ]), 1000 # 160
        # )
        self.mvas = [
            mva_obstacle, mva_low, mva_low2, mva_low3, mva_low4, mva_low6,
            mva_low5, mva_low7, mva_low8, mva_low9
        ]

        # Runway at (35, 20)
        self.runway = model.Runway(35, 20, 0, 270)

        # Airspace
        self.airspace = model.Airspace(self.mvas, self.runway)

        # Wind: moderate, with some swirl
        self.wind = model.Wind((0, 40, 0, 40), swirl_scale=2.0)

        # Fixed entry points for consistency across scenarios
        self.entrypoints = [
            model.EntryPoint(2, 2, 45, [150]),   # Southwest corner, heading NE
            model.EntryPoint(33, 38, 225, [150]) # Northeast corner, heading SW
        ] if entry_points is None else entry_points


    def generate_curriculum_entrypoints(self, num_entrypoints: int) -> List[List[model.EntryPoint]]:
        """
        Generate curriculum entry points for training with two planes.
        
        The entry points for both planes will form paths that start near the runway and 
        gradually move outward to their respective starting positions, forcing them to 
        navigate around the central MVA obstacle.
        
        Returns:
            A list of lists where each inner list contains two entry points, 
            one for each plane at a particular curriculum stage.
        """
        # Get runway coordinates
        runway_threshold_coords = self.runway.x, self.runway.y
        
        # Define two distinct entry point positions with headings pointing to runway
        # We specifically choose points that require going around the central obstacle
        entry_points = [
            # First plane - Left approach
            model.EntryPoint(5, 20, 90, [150]),
            # Second plane - Bottom approach 
            model.EntryPoint(20, 5, 0, [150])
        ]
        
        # Calculate paths by extending from runway along same heading to the entry points
        curriculum_paths = []
        
        # Store approach vectors for both entry points (for later adjustment)
        approach_vectors = []
        
        for base_entry in entry_points:
            entry_coords = base_entry.x, base_entry.y
            entry_heading = base_entry.phi
            
            # Calculate the heading from runway to entry point
            dx = entry_coords[0] - runway_threshold_coords[0]
            dy = entry_coords[1] - runway_threshold_coords[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Store the unit vector from runway to entry point (for path adjustments)
            if distance > 0:
                approach_vectors.append((dx/distance, dy/distance))
            else:
                approach_vectors.append((0, 0))  # Failsafe
            
            # Generate curriculum entry points along this path
            path_points = []
            
            for i in range(num_entrypoints):
                # Calculate position along the line from runway to entry point
                # We want to start closer to runway and move outward
                ratio = i / (num_entrypoints - 1) if num_entrypoints > 1 else 0  # 0 to 1
                new_x = runway_threshold_coords[0] + dx * ratio
                new_y = runway_threshold_coords[1] + dy * ratio
                
                # Create new entry point with same heading
                new_entry = model.EntryPoint(new_x, new_y, entry_heading, base_entry.levels)
                path_points.append(new_entry)
            
            curriculum_paths.append(path_points)
        
        # Ensure planes are at least 5 units apart for safety
        min_separation = 5  # Minimum separation distance
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
                # Calculate additional distance needed
                adjustment_needed = min_separation - distance
                
                # We'll move both aircraft further out along their approach paths
                # This maintains their heading to the runway while ensuring separation
                
                # Get the approach vectors
                approach_vector1 = approach_vectors[0]
                approach_vector2 = approach_vectors[1]
                
                # Add a safety margin
                adjustment_distance = adjustment_needed * 0.75
                
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
                minx, miny, maxx, maxy = 0, 0, 40, 40  # Our airspace is 40x40
                
                # Verify new points are within bounds
                within_bounds = (
                    minx <= entry1_adjusted.x <= maxx and miny <= entry1_adjusted.y <= maxy and
                    minx <= entry2_adjusted.x <= maxx and miny <= entry2_adjusted.y <= maxy
                )
                
                if within_bounds:
                    # Use the adjusted entry points
                    curriculum_entries.append([entry1_adjusted, entry2_adjusted])
                    # Print adjusted entry points for debugging
                    logger.info(f"Stage {i+1}: Adjusted entry points for separation - Entry1: ({entry1_adjusted.x:.2f}, {entry1_adjusted.y:.2f}) Entry2: ({entry2_adjusted.x:.2f}, {entry2_adjusted.y:.2f})")
                else:
                    # Use the original entry points if adjustment would push them out of bounds
                    curriculum_entries.append([entry1, entry2])
                    logger.info(f"Stage {i+1}: Using original entry points - Entry1: ({entry1.x:.2f}, {entry1.y:.2f}) Entry2: ({entry2.x:.2f}, {entry2.y:.2f})")
            else:
                # No adjustment needed, use the original entry points
                curriculum_entries.append([entry1, entry2])
                logger.info(f"Stage {i+1}: Entry points with sufficient separation - Entry1: ({entry1.x:.2f}, {entry1.y:.2f}) Entry2: ({entry2.x:.2f}, {entry2.y:.2f})")
        
        # Debug info - print all curriculum entries
        logger.info(f"Generated {len(curriculum_entries)} curriculum stages for MvaGoAroundScenario")
        
        return curriculum_entries