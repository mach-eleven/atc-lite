import random

import numpy as np

from .. import model
from .scenarios import Scenario

import shapely.geometry as shape
from math import ceil

import random

import logging 
logger = logging.getLogger("train.scenarios")
logger.setLevel(logging.INFO)

MvaType = model.MvaType

class CurriculumTrainingScenario(Scenario):
    def __init__(self, entry_xy=(10, 10), entry_heading=45):
        super().__init__()
        self.mvas = [
            model.MinimumVectoringAltitude(
                shape.Polygon([(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]), 3000
            ),
            model.MinimumVectoringAltitude(
                shape.Polygon([(20, 0), (40, 0), (40, 20), (20, 20), (20, 0)]), 4000
            ),
            model.MinimumVectoringAltitude(
                shape.Polygon([(0, 20), (20, 20), (20, 40), (0, 40), (0, 20)]), 3500
            ),
            model.MinimumVectoringAltitude(
                shape.Polygon([(20, 20), (40, 20), (40, 40), (20, 40), (20, 20)]), 2500
            )
        ]
        self.runway = model.Runway(20, 20, 0, 90)
        self.airspace = model.Airspace(self.mvas, self.runway)
        self.wind = model.Wind((0, 40, 0, 40), swirl_scale=0.0)
        self.entrypoints = [model.EntryPoint(entry_xy[0], entry_xy[1], entry_heading, [150])]