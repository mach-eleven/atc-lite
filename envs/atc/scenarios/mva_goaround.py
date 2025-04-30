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
        # Central high MVA obstacle
        mva_obstacle = model.MinimumVectoringAltitude(
            shape.Polygon([
                (20, 16), (25, 16), (25, 24), (20, 24), (20, 16)
            ]), 800000, model.MvaType.MOUNTAINOUS
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

        # Entry point on the left, heading toward the runway but blocked by the MVA
        self.entrypoints = [
            model.EntryPoint(5, 20, 90, [150])
        ]
