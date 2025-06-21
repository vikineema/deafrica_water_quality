import json
import os
from importlib.resources import files

import click
import geopandas as gpd
import numpy as np
from odc.geo.geom import Geometry

from water_quality.io import check_directory_exists, get_filesystem
from water_quality.logs import setup_logging
from water_quality.tiling import get_aoi_tiles
from water_quality.utils import AFRICA_EXTENT_URL
from water_quality.tiling import get_tile_index_str

@click.command(
    name="generate-tasks",
    no_args_is_help=True,
)
@click.option(
    "--place-name",
    type=str,
    help="Optional name of a test area to generate tiles for. "
    "To view the names of these predefined test areas, run the command `list-test-areas`.",
)
@click.argument(
    "max-parallel-steps",
    type=int,
)
def cli(place_name: str, max_parallel_steps: int):
    """
    Get the list of tiles to run the DE Africa Water Quality workflow on,
    split the list of tiles into at most MAX_PARALLEL_STEPS lists and write the lists
    to the json file `/tmp/tasks_chunks`.

    Arguments:
        max-parallel-steps

    """
    _log = setup_logging()

    places_fp = files("water_quality.data").joinpath("places.parquet")
    places_gdf = gpd.read_parquet(places_fp)
    place_name_list = places_gdf["name"].to_list()
    if place_name:
        if place_name not in place_name_list:
            raise ValueError(
                f"{place_name} not in found in test areas file. Expected names include {' ,'.join(place_name_list)}"
            )
        else:
            _log.info(f"Getting tiles for test area {place_name}")
            place = places_gdf[places_gdf["name"].isin([place_name])]
            aoi_geom = Geometry(geom=place.iloc[0].geometry, crs=place.crs)
    else:
        _log.info("Getting tiles for Africa for continental run")
        africa_extent_gdf = gpd.read_file(AFRICA_EXTENT_URL)
        aoi_geom = Geometry(
            geom=africa_extent_gdf.iloc[0].geometry, crs=africa_extent_gdf.crs
        )

    tiles = get_aoi_tiles(aoi_geom)
    tiles = list(tiles)
    tile_ids = [tile[0] for tile in tiles]
    tile_ids =  [get_tile_index_str(tile_id) for tile_id in tile_ids]
    _log.info(f"Tiles found: {', '.join(tile_ids)}")
    _log.info(f"Total number of tiles: {len(tile_ids)}")

    tile_ids.sort()

    # Split the list of tiles
    task_chunks = np.array_split(np.array(tile_ids), max_parallel_steps)
    task_chunks = [chunk.tolist() for chunk in task_chunks]
    task_chunks = list(filter(None, task_chunks))
    task_chunks_count = str(len(task_chunks))
    _log.info(f"{len(tiles)} tile(s) chunked into {task_chunks_count} chunks")
    task_chunks_json_array = json.dumps(task_chunks)

    tasks_directory = "/tmp/"
    tasks_output_file = os.path.join(tasks_directory, "tasks_chunks")
    tasks_count_file = os.path.join(tasks_directory, "tasks_chunks_count")

    fs = get_filesystem(path=tasks_directory)

    if not check_directory_exists(path=tasks_directory):
        fs.mkdirs(path=tasks_directory, exist_ok=True)
        _log.info(f"Created directory {tasks_directory}")

    with fs.open(tasks_output_file, "w") as file:
        file.write(task_chunks_json_array)
    _log.info(f"Tasks chunks written to {tasks_output_file}")

    with fs.open(tasks_count_file, "w") as file:
        file.write(task_chunks_count)
    _log.info(f"Number of tasks chunks written to {tasks_count_file}")
