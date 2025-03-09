# minimal-atc-rl/envs/atc/scenarios.py
import shapely.geometry as shape  # For geometric operations and polygon definitions
from typing import List  # For type hinting

# Import the model module which contains class definitions for airspace components
from . import model
import random

class Scenario:
    """
    Base class for air traffic control scenarios.
    
    A scenario defines a complete ATC environment including:
    - A runway with approach corridor
    - Multiple Minimum Vectoring Altitude (MVA) areas
    - The complete airspace combining all MVAs
    - Entry points where aircraft enter the simulation
    """
    runway: model.Runway  # The runway where aircraft will land
    mvas: List[model.MinimumVectoringAltitude]  # List of MVA regions with minimum safe altitudes
    airspace: model.Airspace  # The complete airspace combining all elements
    entrypoints: List[model.EntryPoint]  # Points where aircraft can enter the simulation
    wind: model.Wind  # Wind conditions in the airspace

class SimpleScenario(Scenario):
    """
    A basic ATC scenario with a simplified airspace design.
    
    This scenario includes:
    - 5 MVA regions with different minimum altitudes
    - A centrally located runway
    - A single entry point for incoming aircraft
    """
    def __init__(self, random_entrypoints=False):
        """
        Initialize the SimpleScenario with predefined airspace elements.
        
        :param random_entrypoints: Flag to enable random entry points (not implemented in this class)
        """
        # Create five minimum vectoring altitude regions with different height restrictions
        # Each MVA is defined by a polygon (coordinates in nautical miles) and a minimum altitude (in feet)
        
        # MVA region 1: Southeast region with 3500ft minimum
        mva_1 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 0), (35, 26)]), 3500) #PRANJAL WANTS TO DRAW THIS SHIT
        
        # MVA region 2: Central-east region with 2400ft minimum (lower, likely closer to airport)
        mva_2 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 26), (35, 30), (15, 30), (15, 27.8)]), 2400)
        
        # MVA region 3: Northeast region with 4000ft minimum
        mva_3 = model.MinimumVectoringAltitude(shape.Polygon([(15, 30), (35, 30), (35, 40), (15, 40)]), 4000)
        
        # MVA region 4: Southwest region with 8000ft minimum (highest, likely mountainous terrain)
        mva_4 = model.MinimumVectoringAltitude(shape.Polygon([(0, 10), (15, 0), (15, 28.7), (0, 17)]), 8000)
        
        # MVA region 5: Northwest region with 6500ft minimum
        mva_5 = model.MinimumVectoringAltitude(shape.Polygon([(0, 17), (15, 28.7), (15, 40), (0, 32)]), 6500)
        

        # Combine all MVAs into a list
        self.mvas = [mva_1, mva_2, mva_3, mva_4, mva_5]

        # Define the runway parameters
        x = 20  # X-coordinate in nautical miles (center of the airspace)
        y = 20  # Y-coordinate in nautical miles (center of the airspace)
        h = 0   # Runway altitude (at sea level)
        phi = 130  # Runway heading (direction from which aircraft depart)
                   # The landing direction would be phi+180 = 310 degrees
        
        # Create the runway object with the specified parameters
        self.runway = model.Runway(x, y, h, phi)
        
        # Create the airspace object by combining MVAs and runway
        self.airspace = model.Airspace(self.mvas, self.runway)

        # Create the wind object 
        self.wind = model.Wind((35, 0, 0, 40))
        # Define a single entry point where aircraft enter the simulation
        # Parameters: (x, y, initial heading, list of possible altitudes in 100s of feet)
        # Create multiple random entry points around the edges of the airspace
        self.entrypoints = []
        
        # Number of entry points to create
        num_points = 20
        
        for _ in range(num_points):
            # Randomly choose which edge to place the entry point
            edge = random.randint(0, 3)

            # this is not random entry point!!!!! - disappointing
            if edge == 0:  # Top edge
                x = random.uniform(0, 35)
                y = 40
                heading = random.choice([180, 225, 270])
            elif edge == 1:  # Right edge
                x = 35
                y = random.uniform(0, 40)
                heading = random.choice([180, 225, 270])
            elif edge == 2:  # Bottom edge
                x = random.uniform(0, 35)
                y = 0
                heading = random.choice([0, 45, 90])
            else:  # Left edge
                x = 0
                y = random.uniform(0, 40)
                heading = random.choice([45, 90, 135])
            
            # Create entry point with random altitude between 100-200 hundred feet
            altitude = random.randint(100, 200)
            self.entrypoints.append(model.EntryPoint(x, y, heading, [altitude]))
