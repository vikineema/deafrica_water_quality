"""
This module provides functions to create and process tiles used
as building blocks in the DE Africa Water Quality workflow.
"""

import logging
import re
from typing import Iterator

import geopandas as gpd
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.stats._text import split_and_check

from water_quality.africa_extent import AFRICA_EXTENT_URL
from water_quality.grid import WaterbodiesGrid

log = logging.getLogger(__name__)


def get_aoi_tiles(
    aoi_geom: Geometry,
) -> Iterator[tuple[tuple[int, int], GeoBox]]:
    """
    Get the tiles covering an area of interest defined by the input
    polygon.

    Parameters
    ----------
    aoi_geom : Geometry
        Polygon defining the area of interest.

    Returns
    -------
    Iterator[tuple[tuple[int, int], GeoBox]]
        Output is a sequence of tile_index, odc.geo.geobox.GeoBox
        tuples.
    """
    # Tiles to match the DE Africa Landsat GeoMAD products tiles.
    gridspec = WaterbodiesGrid().gridspec
    aoi_geom = aoi_geom.to_crs(gridspec.crs)

    tiles = gridspec.tiles_from_geopolygon(aoi_geom)

    return tiles


def get_tile_region_codes(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
    sep: str = "",
) -> list[str]:
    """
    Get the region codes for a list of tiles.

    Parameters
    ----------
    tiles : Iterator[tuple[tuple[int, int], GeoBox]] | \
            list[tuple[tuple[int, int], GeoBox]]
        Tiles to get the region codes for.
    sep : str, optional
        Seperator between the x and y parts of the region code,
        by default ""
    Returns
    -------
    list[str]
        List of region codes for the input tiles.
    """
    if not isinstance(tiles, list):
        tiles = list(tiles)

    region_codes = []
    for tile in tiles:
        tile_id = tile[0]
        region_codes.append(get_region_code(tile_id, sep))
    return region_codes


def get_tile_extents(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
) -> list[Geometry]:
    """
    Get the extent geometry of each tile in a list of tiles.

    Parameters
    ----------
    tiles : Iterator[tuple[tuple[int, int], GeoBox]] | \
            list[tuple[tuple[int, int], GeoBox]]
        Tiles to get the extents for.

    Returns
    -------
    list[Geometry]
        List of the tile extent Geometries for each tile in the input 
        tile list.
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
            "List of input tiles contains tiles with different CRS: "
            f"{', '.join(crs_list)}"
        )
    else:
        return tile_extents


def tiles_to_gdf(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
) -> gpd.GeoDataFrame:
    """
    Get the tile extents for a list of tiles into a GeoDataFrame.

    Parameters
    ----------
    tiles : Iterator[tuple[tuple[int, int], GeoBox]] |
            list[tuple[tuple[int, int], GeoBox]]
        Tiles to get the extent Geometries for

    Returns
    -------
    gpd.GeoDataFrame
        Table containing the region codes and extent Geometries
        for a list
    """
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

    Parameters
    ----------
    save_to_disk : bool
        If True write the tile extents for the tiles to a parquet file.

    Returns
    -------
    Iterator[tuple[tuple[int, int], GeoBox]]
        Output is a sequence of tile_index, odc.geo.geobox.GeoBox
        tuples.
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


def get_region_code(tile_id: tuple[int, int], sep: str = "") -> str:
    """
    Get the region code for a tile from its tile ID in the format
    format "x{x:02d}{sep}y{y:02d}".

    Parameters
    ----------
    tile_id : tuple[int, int]
        Tile ID for the tile.
    sep : str, optional
        Seperator between the x and y parts of the region code, by
        default ""

    Returns
    -------
    str
        Region code for the input tile ID.
    """
    x, y = tile_id
    region_code_format = "x{x:02d}{sep}y{y:02d}"
    region_code = region_code_format.format(x=x, y=y, sep=sep)
    return region_code


def parse_region_code(region_code: str) -> tuple[int, int]:
    """
    Parse a tile id in the string format "x{x:02d}{sep}y{y:02d}", into
    the a tuple of integers (x, y).

    Parameters
    ----------
    region_code : str
        Tile id in string format "x{x:02d}{sep}y{y:02d}".

    Returns
    -------
    tuple[int, int]
        Tile  id as a tuple of integers (x, y).
    """

    x_pattern = re.compile(r"x\d{3}")
    y_pattern = re.compile(r"y\d{3}")

    tile_id_x_str = re.search(x_pattern, region_code).group(0)
    tile_id_y_str = re.search(y_pattern, region_code).group(0)

    tile_id_x = int(tile_id_x_str.lstrip("x"))
    tile_id_y = int(tile_id_y_str.lstrip("y"))

    tile_id = (tile_id_x, tile_id_y)

    return tile_id


def create_task_id(year: str | int, tile_id: tuple[int, int] | str) -> str:
    """Create  a task given a year and a tile id.

    Parameters
    ----------
    year : str | int
        Year to create the task for.
    tile_id : tuple[int, int] | str
        Tile ID for the tile to create the task for.

    Returns
    -------
    str
        Task ID
    """
    if isinstance(year, int):
        year = str(year)
    # task id format "{year}/x{x:02d}/y{y:02d}"
    region_code = get_region_code(tile_id, sep="/")
    task_id = f"{year}/{region_code}"
    return task_id


def parse_task_id(task_id: str) -> tuple[int, tuple[int, int]]:
    """
    Parse a task ID into the year and tile ID it was created from.

    Parameters
    ----------
    task_id : str
        Task ID to parse.

    Returns
    -------
    tuple[int, tuple[int, int]]
        Year and tile ID components of the task.
    """
    # Check Task id has only 3 parts
    sep = "/" # based on seperator used in create_task_id
    _ = split_and_check(task_id, sep, 3)

    # Get the tile ID 
    tile_id = parse_region_code(task_id)
    
    # Get the year
    year_pattern = re.compile(r"\d{4}")
    year_str = re.search(year_pattern, task_id).group(0) 
    year = int(year_str)
    return year, tile_id
