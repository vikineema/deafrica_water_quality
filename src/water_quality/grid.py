from odc.geo import XY, Resolution
from odc.geo.gridspec import GridSpec


class WaterbodiesGrid:
    def __init__(self):
        self.gridname = "waterbodies_grid"
        self.gridspec = GridSpec(
            crs="EPSG:6933",
            tile_shape=XY(y=3200, x=3200),
            resolution=Resolution(y=-30, x=30),
            origin=XY(y=-7392000, x=-17376000),
        )
