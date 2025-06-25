from importlib.resources import files

import click
import geopandas as gpd
from odc.geo.geom import Geometry

from water_quality.io import check_directory_exists, get_filesystem
from water_quality.logs import setup_logging
from water_quality.tiling import get_aoi_tiles, get_tile_index_str
from water_quality.utils import AFRICA_EXTENT_URL


@click.command(
    name="generate-tasks",
    no_args_is_help=True,
)
@click.option(
    "--place-name",
    type=str,
    help="Optional name of a test area to generate tiles for. "
    "To view the names of these predefined test areas, run the command `wq-list-test-areas`.",
)
@click.argument(
    "output-file",
    type=str,
)
def cli(place_name: str, output_file: str):
    """
    Get the list of tiles to run the DE Africa Water Quality workflow on
    and write the tile IDs to the file OUTPUT_FILE.

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
    tile_ids = [get_tile_index_str(tile_id) for tile_id in tile_ids]
    _log.info(f"Tiles found: {', '.join(tile_ids)}")
    _log.info(f"Total number of tiles: {len(tile_ids)}")

    tile_ids.sort()

    fs = get_filesystem(path=output_file, anon=False)
    tasks_directory = fs._parent(output_file)
    if not check_directory_exists(tasks_directory):
        fs.makedirs(tasks_directory, exist_ok=True)

    with fs.open(output_file, "w") as file:
        for item in tile_ids:
            file.write(str(item) + "\n")

    _log.info(f"Tasks written to {output_file}")


if __name__ == "__main__":
    cli()
