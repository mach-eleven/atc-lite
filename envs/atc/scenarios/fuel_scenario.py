import random
import numpy as np
import shapely.geometry as shape
from .. import model
from .scenarios import Scenario

import logging
logger = logging.getLogger("train.scenarios")
logger.setLevel(logging.INFO)

MvaType = model.MvaType

class FuelScenario(Scenario):
    """
    A small airspace scenario designed for fuel optimization challenges.
    
    This scenario includes:
    - 4 non-overlapping MVA regions with varied minimum altitudes and types
    - A centrally located runway
    - Entry points positioned strategically inside the MVA airspace
    """
    def __init__(self, random_entrypoints=False):
        """
        Initialize the FuelScenario with predefined airspace elements.
        
        :param random_entrypoints: Flag to enable random entry points
        """
        # Create a small airspace (20x20 nautical miles)
        # Using distinct, non-overlapping MVAs
        
        # MVA region 1: Southwest quadrant with 2500ft minimum (standard clearance)
        mva_1 = model.MinimumVectoringAltitude(
            shape.Polygon([(5, 5), (10, 5), (10, 10), (5, 10), (5, 5)]), 
            2500,
            mva_type=MvaType.GENERIC
        )
        
        # MVA region 2: Southeast quadrant with 3000ft minimum (oceanic)
        mva_2 = model.MinimumVectoringAltitude(
            shape.Polygon([(10, 5), (15, 5), (15, 10), (10, 10), (10, 5)]), 
            3000,
            mva_type=MvaType.OCEANIC
        )
        
        # MVA region 3: Northeast quadrant with 3500ft minimum (mountains)
        mva_3 = model.MinimumVectoringAltitude(
            shape.Polygon([(10, 10), (15, 10), (15, 15), (10, 15), (10, 10)]), 
            3500,
            mva_type=MvaType.MOUNTAINOUS
        )
        
        # MVA region 4: Northwest quadrant with 2000ft minimum (oceanic)
        mva_4 = model.MinimumVectoringAltitude(
            shape.Polygon([(5, 10), (10, 10), (10, 15), (5, 15), (5, 10)]), 
            2000,
            mva_type=MvaType.GENERIC
        )
        
        # Combine all MVAs into a list - removed overlapping central MVA
        self.mvas = [mva_1, mva_2, mva_3, mva_4]

        # Define the runway parameters - centered in the airspace
        x = 10  # X-coordinate in nautical miles (center of the airspace)
        y = 10  # Y-coordinate in nautical miles (center of the airspace)
        h = 0   # Runway altitude (at sea level)
        phi = 90  # Runway heading east-west
        
        # Create the runway object with the specified parameters
        self.runway = model.Runway(x, y, h, phi)
        
        # Create the airspace object by combining MVAs and runway
        self.airspace = model.Airspace(self.mvas, self.runway)

        # Create the wind object with moderate effects
        self.wind = model.Wind((5, 15, 5, 15), swirl_scale=0.3)
        
        # Define entry points where aircraft enter the simulation
        if random_entrypoints:
            self.entrypoints = self._generate_random_entrypoints(6)
        else:
            # Fixed entry points positioned INSIDE the MVA areas, not on edges
            self.entrypoints = [
                model.EntryPoint(2, 2, 45, [150]),   # Southwest corner, heading NE
                model.EntryPoint(33, 38, 225, [150]) # Northeast corner, heading SW
            ]

    def _generate_random_entrypoints(self, num_points=6):
        """Generate random entry points INSIDE the MVA regions, not on edges."""
        entrypoints = []
        
        # Create a list of MVA regions and their corresponding bounding coordinates
        mva_regions = [
            # (MVA index, min_x, max_x, min_y, max_y)
            (0, 6, 9, 6, 9),     # Southwest MVA (mva_1) - slightly inset from borders
            (1, 11, 14, 6, 9),   # Southeast MVA (mva_2)
            (2, 11, 14, 11, 14), # Northeast MVA (mva_3)
            (3, 6, 9, 11, 14)    # Northwest MVA (mva_4)
        ]
        
        # Ensure at least one point in each MVA region
        for i, (mva_idx, min_x, max_x, min_y, max_y) in enumerate(mva_regions):
            if i < num_points:  # Make sure we don't request more points than MVA regions
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                
                # Choose heading based on general direction toward runway
                if mva_idx == 0:  # Southwest
                    heading = random.choice([45, 0, 90])
                elif mva_idx == 1:  # Southeast
                    heading = random.choice([315, 0, 270])
                elif mva_idx == 2:  # Northeast
                    heading = random.choice([225, 180, 270])
                else:  # Northwest
                    heading = random.choice([135, 180, 90])
                
                # Create entry point with altitude appropriate for the MVA
                if mva_idx == 0:
                    altitude = random.randint(110, 130)  # Lower altitude for 2500ft MVA
                elif mva_idx == 1:
                    altitude = random.randint(120, 140)  # Medium altitude for 3000ft MVA
                elif mva_idx == 2:
                    altitude = random.randint(140, 160)  # Higher altitude for 3500ft MVA
                else:
                    altitude = random.randint(100, 120)  # Lowest altitude for 2000ft MVA
                
                entrypoints.append(model.EntryPoint(x, y, heading, [altitude]))
        
        # Add any remaining random points
        remaining_points = num_points - len(mva_regions)
        for _ in range(remaining_points):
            # Select a random MVA region
            mva_idx, min_x, max_x, min_y, max_y = random.choice(mva_regions)
            
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            
            # Choose heading based on general direction toward runway
            heading = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            
            # Create entry point with appropriate altitude for the MVA
            altitude = random.randint(100, 160)
            entrypoints.append(model.EntryPoint(x, y, heading, [altitude]))
            
        return entrypoints