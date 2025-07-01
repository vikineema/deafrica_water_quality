import logging
import os
import posixpath
import re
from typing import Iterator

import geopandas as gpd
import pandas as pd
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from water_quality.grid import WaterbodiesGrid
from water_quality.io import is_local_path
from water_quality.utils import AFRICA_EXTENT_URL

log = logging.getLogger(__name__)


def get_tile_index_str_tuple(string_: str) -> tuple[str]:
    """Get the tile index in the string
    format "{x:03d}","{y:03d}" from a string.

    Parameters
    ----------
    string_ : str
        String to search the tile index from.

    Returns
    -------
    tuple[str]
        Tile index in the string format "{x:03d}","{y:03d}" .
    """
    x_pattern = re.compile(r"x\d{3}")
    y_pattern = re.compile(r"y\d{3}")

    tile_index_x_str = re.search(x_pattern, string_).group(0)
    tile_index_y_str = re.search(y_pattern, string_).group(0)

    return tile_index_x_str, tile_index_y_str


def get_tile_index_int_tuple(string_: str) -> tuple[int, int]:
    """
    Get the tile index in the format (x,y) from a string
    where x and y are integers.

    Parameters
    ----------
    string_ : str
        String to search the tile index from.

    Returns
    -------
    tuple[int, int]
        Tile index in the format (x,y) where x and y are integers.
    """

    tile_index_x_str, tile_index_y_str = get_tile_index_str_tuple(string_)

    tile_index_x = int(tile_index_x_str.lstrip("x"))
    tile_index_y = int(tile_index_y_str.lstrip("y"))

    tile_index = (tile_index_x, tile_index_y)

    return tile_index


def get_tile_index_str(tile_index_tuple: tuple[int, int]) -> str:
    """
    Convert a tile index tuple (x,y) into the string format "{x:03d}","{y:03d}" .

    Parameters
    ----------
    tile_index_tuple : tuple[int, int]
        Tile index tuple to convert to string.

    Returns
    -------
    str
        Tile index in string format "{x:03d}","{y:03d}" .
    """

    tile_index_x, tile_index_y = tile_index_tuple

    tile_index_str = f"x{tile_index_x:03d}y{tile_index_y:03d}"

    return tile_index_str


def get_tile_index_tuple_from_filename(file_path: str) -> tuple[int, int]:
    """
    Search for a tile index in the base name of a file.

    Parameters
    ----------
    file_path : str
        File path to search tile index in.

    Returns
    -------
    tuple[int, int]
        Found tile index (x,y).
    """
    if is_local_path(file_path):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
    else:
        file_name = os.path.splitext(posixpath.basename(file_path))[0]

    tile_id = get_tile_index_int_tuple(file_name)

    return tile_id


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
        gdf = tiles_to_gdf(tiles)
        output_fp = "water_quality_regions.parquet"
        gdf.to_parquet(output_fp)
        log.info(f"Regions saved to {output_fp}")
    return tiles


def tiles_to_gdf(
    tiles: Iterator[tuple[tuple[int, int], GeoBox]]
    | list[tuple[tuple[int, int], GeoBox]],
) -> gpd.GeoDataFrame:
    if not isinstance(tiles, list):
        tiles = list(tiles)

    tiles_list = []
    for tile in tiles:
        tile_id = tile[0]
        tile_id_str = get_tile_index_str(tile_id)

        tile_geobox = tile[-1]
        tile_extent = tile_geobox.extent

        tile_gdf = gpd.GeoDataFrame(
            data={"region_code": [tile_id_str]},
            geometry=[tile_extent],
            crs=tile_extent.crs,
        )
        tiles_list.append(tile_gdf)

    gdf_all = gpd.GeoDataFrame(
        pd.concat(tiles_list, ignore_index=True), geometry="geometry"
    )
    return gdf_all
