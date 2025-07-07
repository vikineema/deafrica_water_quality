from odc.geo import XY, Resolution
from odc.geo.gridspec import GridSpec


def get_waterbodies_grid(resolution: int) -> GridSpec:
    """
    Get the grid to be used for tiling in the DE Africa
    Water Quality continental workflow.

    Parameters
    ----------
    resolution : int
        X and Y resolution as an absolute integer.

    Returns
    -------
    GridSpec
        Gridspec to be used for tiling.
    """
    resolution = abs(resolution)
    tile_size = 96000
    tile_shape = tile_size / resolution

    gridspec = GridSpec(
        crs="EPSG:6933",
        tile_shape=XY(
            y=tile_shape,
            x=tile_shape,
        ),
        resolution=Resolution(y=-resolution, x=resolution),
        origin=XY(y=-7392000, x=-17376000),
    )
    return gridspec
