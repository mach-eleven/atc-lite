import shapely.geometry as shape
from .. import model
from .scenarios import Scenario

class MvaGoAroundScenario(Scenario):
    """
    Scenario where the aircraft must go around a high MVA region to reach the runway.
    The MVA is placed directly between the entry point and the runway, forcing a detour.
    """
    def __init__(self):
        super().__init__()
        # Define the MVA obstacle (move further back, away from the runway)
        mva_obstacle = model.MinimumVectoringAltitude(
            shape.Polygon([
                (20, 16), (25, 16), (25, 24), (20, 24), (20, 16)
            ]), 800000  # High enough to force a detour
        )
        # Add a lower MVA for the rest of the airspace
        mva_low = model.MinimumVectoringAltitude(
            shape.Polygon([
                (0, 0), (40, 0), (40, 40), (0, 40), (0, 0)
            ]), 100
        )
        self.mvas = [mva_obstacle, mva_low]

        # Runway at (35, 20)
        self.runway = model.Runway(35, 20, 0, 270)

        # Airspace
        self.airspace = model.Airspace(self.mvas, self.runway)

        # Wind (optional: set to zero for simplicity)
        self.wind = model.Wind((0, 40, 0, 40), swirl_scale=0.0)

        # Entry point on the left, heading toward the runway but blocked by the MVA
        self.entrypoints = [
            model.EntryPoint(5, 20, 90, [150])
        ]
