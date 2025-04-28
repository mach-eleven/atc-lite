# minimal-atc-rl/envs/atc/scenarios.py
from typing import List  # For type hinting
# Import the model module which contains class definitions for airspace components
from .. import model

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