from importlib.resources import files

import click
import geopandas as gpd
from odc.geo.geom import Geometry
from odc.stats.model import DateTimeRange

from water_quality.dates import get_temporal_ids
from water_quality.io import (
    check_directory_exists,
    check_file_exists,
    get_filesystem,
)
from water_quality.logs import setup_logging
from water_quality.tasks import create_task_id
from water_quality.tiling import (
    get_africa_tiles,
    get_aoi_tiles,
    parse_region_code,
)

SUPPORTED_FREQUENCY = [
    "annual",
    "semiannual",
    "monthly",
    "fortnightly",
    "weekly",
]


@click.command(
    name="generate-tasks",
    no_args_is_help=True,
)
@click.option(
    "--tile-ids",
    help="Optional list of comma separated tile IDs in the format "
    "x{x:02d}/y{y:02d} to generate tasks for. For example "
    "`x188/y109,x178/y095,x199y/100`",
)
@click.option(
    "--tile-ids-file",
    help="Optional path to text file containing the tile IDs to "
    "generate tasks for. This file can be generated using the command "
    "`wq-generate-tiles`.",
)
@click.option(
    "--place-name",
    type=str,
    help="Optional name of a test area to get the tile IDs to generate "
    "tasks for. To view the names of these predefined test areas, run "
    "the command `wq-list-test-areas`.",
)
@click.argument(
    "temporal-range",
    type=str,
)
@click.argument(
    "frequency",
)
@click.argument(
    "output-file",
    type=str,
)
def cli(
    tile_ids: str,
    tile_ids_file: str,
    place_name: str,
    temporal_range: str,
    frequency: str,
    output_file: str,
):
    """
    Prepare tasks for the time range TEMPORAL_RANGE,
    for example '2020-05--P1M' for the month of May 2020, and temporal
    binning FREQUENCY for running the DE Africa Water Quality continental
    workflow on. If no tile IDs are specified, the tasks will be generated
    for all the tiles across Africa. The tasks will be written to the file
    OUTPUT_FILE.

    **Note**: If the temporal range does not completely cover a temporal
    bin that it falls in, all data available for the complete temporal
    bin it falls into would be loaded. For example the temporal range
    '2025-08-13--P3D' with the frequency 'weekly', all data available for
    the temporal bin '2025-08-13--P1W' will be loaded. For the temporal
    range '2025-08-13--P1M' with the frequency monthly, data will be
    loaded for the following temporal bins '2025-08--P1M' and
    '2025-09--P1M'.
    """
    log = setup_logging()

    if frequency not in SUPPORTED_FREQUENCY:
        e = ValueError(
            f"Frequency must be one of {'|'.join(SUPPORTED_FREQUENCY)} and not "
            f"{frequency}"
        )
        log.error(e)
        raise e

    try:
        temporal_range = DateTimeRange(temporal_range)
    except ValueError:
        e = ValueError(
            f"Failed to parse supplied temporal_range: '{temporal_range}'",
        )
        log.error(e)
        raise e

    if tile_ids and tile_ids_file and place_name:
        raise click.UsageError(
            "Specify exactly one of --tile-ids or --tile-ids-file "
            "or --place-name."
        )

    temporal_ids = get_temporal_ids(temporal_range, frequency)

    if tile_ids:
        region_codes = tile_ids.split(",")
        region_codes = [i.strip() for i in region_codes]
        tile_ids_list = [parse_region_code(i) for i in region_codes]
    elif tile_ids_file:
        # Assumption here is the file is public-read.
        if not check_file_exists(tile_ids_file):
            raise FileNotFoundError(f"{tile_ids_file} does not exist!")
        else:
            fs = get_filesystem(tile_ids_file, anon=True)
            with fs.open(tile_ids_file, "r") as f:
                region_codes = f.readlines()
                region_codes = [i.strip() for i in region_codes]
                tile_ids_list = [parse_region_code(i) for i in region_codes]
    else:
        if place_name:
            places_fp = files("water_quality.data").joinpath("places.parquet")
            places_gdf = gpd.read_parquet(places_fp)
            place_name_list = places_gdf["name"].to_list()
            if place_name not in place_name_list:
                raise ValueError(
                    f"{place_name} not in found in test areas file. "
                    f"Expected names include {' ,'.join(place_name_list)}"
                )
            else:
                log.info(f"Getting tiles for test area {place_name}")
                place = places_gdf[places_gdf["name"].isin([place_name])]
                aoi_geom = Geometry(geom=place.iloc[0].geometry, crs=place.crs)
                tiles = get_aoi_tiles(aoi_geom)
        else:
            log.info("Getting tiles for all of Africa for continental run")
            tiles = get_africa_tiles(save_to_disk=False)

        tiles = list(tiles)
        tile_ids_list = [tile[0] for tile in tiles]

    tasks = []
    for temporal_id in temporal_ids:
        for tile_id in tile_ids_list:
            task_id = create_task_id(temporal_id, tile_id)
            tasks.append(task_id)

    tasks.sort()

    log.info(f"Total tasks: {len(tasks)}")

    fs = get_filesystem(path=output_file, anon=False)
    parent_dir = fs._parent(output_file)
    if not check_directory_exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(output_file, "w") as file:
        for task_id in tasks:
            file.write(str(task_id) + "\n")
    log.info(f"{len(tasks)} tasks written to {output_file}")
