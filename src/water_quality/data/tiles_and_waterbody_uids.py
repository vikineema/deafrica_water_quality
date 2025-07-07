"""
This module contains functions to generate a table matching tile IDs
(region codes) for DE Africa Water Quality tiles to DE Africa
Historical Extent waterbodies polygon uids that intersect with each
the tile.
"""

from pathlib import Path

import geopandas as gpd
from deafrica_tools.waterbodies import (
    get_waterbodies as get_deafrica_waterbodies,
)
from tqdm import tqdm

from water_quality.io import join_url
from water_quality.logs import setup_logging
from water_quality.tiling import get_africa_tiles, tiles_to_gdf


def get_tiles_and_waterbody_uids():
    log = setup_logging()

    data_dir = Path(__file__).resolve().parent
    output_file = join_url(
        str(data_dir), "wq_tile_ids_and_waterbodies_uids.parquet"
    )

    # Any resolution can be used here as
    # the tile extents will be the same.
    tiles = get_africa_tiles(resolution_m=30)
    tiles = list(tiles)
    log.info(f"Total number of tiles: {len(tiles)}")

    tiles_gdf = tiles_to_gdf(tiles)
    tiles_gdf = tiles_gdf.to_crs("EPSG:4326")

    for i in tqdm(range(len(tiles_gdf))):
        tile_geometry = tiles_gdf.iloc[i].geometry
        tile_waterbodies = get_deafrica_waterbodies(
            tuple(tile_geometry.bounds), crs="EPSG:4326"
        )

        if tile_waterbodies.empty:
            tiles_gdf.at[i, "waterbodies_uids"] = ""
        else:
            tile_geometry_gdf = gpd.GeoDataFrame(
                geometry=[tile_geometry], crs="EPSG:4326"
            )
            intersecting_ids = (
                tile_waterbodies.sjoin(
                    tile_geometry_gdf, how="inner", predicate="intersects"
                )["uid"]
                .unique()
                .tolist()
            )
            tiles_gdf.at[i, "waterbodies_uids"] = ", ".join(intersecting_ids)

    tiles_gdf[["region_code", "waterbodies_uids"]].to_parquet(output_file)
    log.info(f"Table saved to {output_file}")


if __name__ == "__main__":
    get_tiles_and_waterbody_uids()
