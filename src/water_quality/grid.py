from odc.geo import XY, Resolution
from odc.geo.gridspec import GridSpec


class WaterbodiesGrid:
    def __init__(self):
        self.gridname = "waterbodies_grid"
        self.resolution = Resolution(y=-10, x=10)
        # Match the tile size for DE Africa GeoMADs tiles
        self.tile_size = XY(y=96000, x=96000)
        self.gridspec = GridSpec(
            crs="EPSG:6933",
            tile_shape=XY(
                y=self.tile_size.y / abs(self.resolution.y),
                x=self.tile_size.x / abs(self.resolution.x),
            ),
            resolution=self.resolution,
            origin=XY(y=-7392000, x=-17376000),
        )
