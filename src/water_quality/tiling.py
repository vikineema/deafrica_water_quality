"""
This module provides functions to create and process tiles used
as building blocks in the DE Africa Water Quality workflow.
"""

import logging
from typing import Iterator

import geopandas as gpd
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from water_quality.grid import WaterbodiesGrid
from water_quality.utils import AFRICA_EXTENT_URL

log = logging.getLogger(__name__)


def get_aoi_tiles(aoi_geom: Geometry) -> Iterator[tuple[tuple[int, int], GeoBox]]:
    """
    Get the tiles covering an area of interest defined by the input polygon.

    Parameters
    ----------
    aoi_geom : Geometry
        Polygon defining the area of interest.

    Returns
    -------
    Iterator[tuple[tuple[int, int], GeoBox]]
        Output is a sequence of tile_index, odc.geo.geobox.GeoBox tuples.
    """
    # Tiles to match the DE Africa Landsat GeoMAD products tiles.
    gridspec = WaterbodiesGrid().gridspec
    aoi_geom = aoi_geom.to_crs(gridspec.crs)

    tiles = gridspec.tiles_from_geopolygon(aoi_geom)

    return tiles


def get_region_code(tile_id: tuple[int, int]) -> str:
    """Get the tile ID for a tile in the region code
    format "x{x:02d}y{y:02d}" .

    Parameters
    ----------
    tile_id : tuple[int, int]
        Tile ID (x,y) for a tile

    Returns
    -------
    str
        Tile ID in the region code format string "x{x:02d}y{y:02d}".
    """
    region_code_format = "x{x:02d}y{y:02d}"

    x, y = tile_id
    region_code = region_code_format.format(x=x, y=y)
    return region_code


def get_tile_region_codes(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
) -> list[str]:
    """
    Get the region codes for a list of tiles.

    Parameters
    ----------
    tiles : Iterator[tuple[tuple[int, int], GeoBox]] | list[tuple[tuple[int, int], GeoBox]]
        Tiles to get the region codes for.

    Returns
    -------
    list[str]
        List of region codes for the tiles.
    """
    if not isinstance(tiles, list):
        tiles = list(tiles)

    tile_ids = []
    for tile in tiles:
        tile_id = tile[0]
        tile_ids.append(get_region_code(tile_id))
    return tile_ids


def get_tile_extents(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
) -> list[Geometry]:
    """
    Get the extent geometry of each tile in a list of tiles.

    Parameters
    ----------
    tiles : Iterator[tuple[tuple[int, int], GeoBox]] | list[tuple[tuple[int, int], GeoBox]]
        Tiles to get the extents for.

    Returns
    -------
    list[Geometry]
        List of the tile extent Geometries for each tile in the input tile list.
    """
    if not isinstance(tiles, list):
        tiles = list(tiles)

    tile_extents = []
    for tile in tiles:
        tile_geobox = tile[-1]
        tile_extent = tile_geobox.extent
        tile_extents.append(tile_extent)

    # Check if all extents have the same crs
    crs_list = [i.crs for i in tile_extents]
    crs_list = list(set(crs_list))
    try:
        assert len(crs_list) == 1
    except AssertionError:
        raise ValueError(
            "List of input tiles contains tiles with "
            f"different CRS: {', '.join(crs_list)}"
        )
    else:
        return tile_extents


def tiles_to_gdf(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
) -> gpd.GeoDataFrame:
    if not isinstance(tiles, list):
        tiles = list(tiles)

    region_codes = get_tile_region_codes(tiles)
    tile_extents = get_tile_extents(tiles)
    crs = tile_extents[0].crs

    tiles_extents_gdf = gpd.GeoDataFrame(
        data={"region_code": region_codes},
        geometry=tile_extents,
        crs=crs,
    )
    return tiles_extents_gdf


def get_africa_tiles(
    save_to_disk: bool = False,
) -> Iterator[tuple[tuple[int, int], GeoBox]]:
    """
    Get tiles over Africa's extent.

    Returns
    -------
    Iterator[tuple[tuple[int, int], GeoBox]]
        Output is a sequence of tile_index, odc.geo.geobox.GeoBox tuples.
    """

    # Get the tiles over Africa
    africa_extent = gpd.read_file(AFRICA_EXTENT_URL)
    africa_extent_geom = Geometry(
        geom=africa_extent.iloc[0].geometry, crs=africa_extent.crs
    )
    tiles = get_aoi_tiles(africa_extent_geom)
    if save_to_disk is True:
        tiles_gdf = tiles_to_gdf(tiles)
        output_fp = "water_quality_regions.parquet"
        tiles_gdf.to_parquet(output_fp)
        log.info(f"Regions saved to {output_fp}")
    return tiles
