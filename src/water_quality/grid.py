from odc.geo import XY, Resolution
from odc.geo.gridspec import GridSpec


def get_waterbodies_grid(resolution_m: int) -> GridSpec:
    """
    Get the grid to be used for tiling in the DE Africa
    Water Quality continental workflow. The resulting tiles
    cover the same (regions) as the DE Africa GeoMAD products.

    Parameters
    ----------
    resolution : int
        Pixel resolution in meters.

    Returns
    -------
    GridSpec
        Gridspec to be used for tiling.
    """
    if resolution_m < 0:
        raise ValueError(
            f"Expecting positive value for resolution not {resolution_m}"
        )
    if not isinstance(resolution_m, int):
        raise ValueError(
            f"Expecting resolution to be an integer not {resolution_m}"
        )

    # To match the tile size for DE Africa GeoMAD products.
    tile_size = 96000
    tile_shape = tile_size / resolution_m

    gridspec = GridSpec(
        crs="EPSG:6933",
        tile_shape=XY(
            y=tile_shape,
            x=tile_shape,
        ),
        resolution=Resolution(y=-resolution_m, x=resolution_m),
        origin=XY(y=-7392000, x=-17376000),
    )
    return gridspec
