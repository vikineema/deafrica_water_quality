import logging
from collections import defaultdict
from importlib.resources import files

import numpy as np
import pandas as pd
import rioxarray
import toolz
import xarray as xr
from deafrica_tools.waterbodies import get_waterbody
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import assign_crs

from water_quality.dates import year_to_dc_datetime
from water_quality.grid import get_waterbodies_grid
from water_quality.io import (
    find_geotiff_files,
    get_wq_csv_url,
    join_url,
    parse_wq_cog_url,
)
from water_quality.tiling import (
    get_aoi_tiles,
    get_region_code,
    get_tile_region_codes,
)

log = logging.getLogger(__name__)

NORMALISATION_PARAMETERS = {
    "ndssi_rg_msi_agm": {"scale": 83.711, "offset": 56.756},
    "ndssi_rg_oli_agm": {"scale": 45.669, "offset": 45.669},
    "ndssi_rg_tm_agm": {"scale": 149.21, "offset": 57.073},
    "ndssi_bnir_oli_agm": {"scale": 37.125, "offset": 37.125},
    "ti_yu_oli_agm": {"scale": 6.656, "offset": 36.395},
    "ti_yu_tm_agm": {"scale": 8.064, "offset": 42.562},
    "tsm_lym_oli_agm": {"scale": 1.0, "offset": 0.0},
    "tsm_lym_msi_agm": {"scale": 14.819, "offset": -118.137},
    "tsm_lym_tm_agm": {"scale": 1.184, "offset": -2.387},
    "tss_zhang_msi_agm": {"scale": 18.04, "offset": 0.0},
    "tss_zhang_oli_agm": {"scale": 10.032, "offset": 0.0},
    "spm_qiu_oli_agm": {"scale": 1.687, "offset": -0.322},
    "spm_qiu_tm_agm": {"scale": 2.156, "offset": -16.863},
    "spm_qiu_msi_agm": {"scale": 2.491, "offset": -4.112},
    "ndci_msi54_agm": {"scale": 131.579, "offset": 21.737},
    "ndci_msi64_agm": {"scale": 33.153, "offset": 33.153},
    "ndci_msi74_agm": {"scale": 33.516, "offset": 33.516},
    "ndci_tm43_agm": {"scale": 53.157, "offset": 28.088},
    "ndci_oli54_agm": {"scale": 38.619, "offset": 29.327},
    "chla_meris2b_msi_agm": {"scale": 1.148, "offset": -36.394},
    "chla_modis2b_msi_agm": {"scale": 0.22, "offset": 7.139},
    "chla_modis2b_tm_agm": {"scale": 1.209, "offset": -63.141},
    "ndssi_bnir_tm_agm": {"scale": 37.41, "offset": 37.41},
}


def load_all_waterbodies_uids():
    file_path = files("water_quality.data").joinpath("waterbodies_uids.txt")
    with open(file_path, "r") as f:
        waterbodies_uids = [line.strip() for line in f]
    return waterbodies_uids


def get_waterbody_geom(waterbody_uid: str) -> Geometry:
    gridspec = get_waterbodies_grid()
    waterbody_gdf = get_waterbody(waterbody_uid)
    waterbody_geom = Geometry(
        geom=waterbody_gdf.iloc[0].geometry, crs=waterbody_gdf.crs
    )
    waterbody_geom = waterbody_geom.to_crs(gridspec.crs)
    return waterbody_geom


def get_waterbody_geobox(waterbody_uid: str) -> GeoBox:
    waterbody_geom = get_waterbody_geom(waterbody_uid)
    gridspec = get_waterbodies_grid()
    waterbody_geobox = GeoBox.from_geopolygon(
        geopolygon=waterbody_geom,
        resolution=gridspec.resolution,
        crs=gridspec.crs,
    )
    return waterbody_geobox


def get_waterbody_tiles(waterbody_uid) -> list[tuple[tuple[int, int], GeoBox]]:
    all_waterbodies_uids = load_all_waterbodies_uids()
    if waterbody_uid not in all_waterbodies_uids:
        raise ValueError(f"Waterbody {waterbody_uid} not found")
    else:
        waterbody_geom = get_waterbody_geom(waterbody_uid)
        tiles = get_aoi_tiles(waterbody_geom)
        tiles = list(tiles)
    return tiles


def _get_cog_year(cog_url: str) -> str:
    _, _, year = parse_wq_cog_url(cog_url)
    return year


def get_wq_measures_cogs(
    waterbody_uid: str, wq_measures_dir: str
) -> dict[str, dict[tuple[int, int], list]]:
    tiles = get_waterbody_tiles(waterbody_uid)
    region_codes = get_tile_region_codes(tiles, sep="")
    log.info(
        f"Found {len(tiles)} tiles covering "
        f"waterbody {waterbody_uid}: {', '.join(region_codes)}"
    )
    grouped_by_year_and_tile = defaultdict(lambda: defaultdict(list))
    for tile_id, _ in tiles:
        region_code = get_region_code(tile_id, "/")
        tile_cogs_dir = join_url(wq_measures_dir, region_code)
        all_tile_cog_urls = find_geotiff_files(tile_cogs_dir)
        for year, cog_urls in toolz.groupby(
            _get_cog_year, all_tile_cog_urls
        ).items():
            grouped_by_year_and_tile[year][tile_id].extend(cog_urls)
    return grouped_by_year_and_tile


def load_wq_measurements_table(wq_parameters_csv_url: str) -> pd.DataFrame:
    wq_parameters_df = pd.read_csv(wq_parameters_csv_url)
    assert all(
        col in wq_parameters_df.columns
        for col in ["tss_measure", "chla_measure"]
    ), "Required columns 'tss_measure' and/or 'chla_measure' are missing."
    return wq_parameters_df


def get_bands_to_load(wq_parameters_csv_url: str) -> list[str]:
    wq_parameters_df = load_wq_measurements_table(wq_parameters_csv_url)
    bands_to_load = []
    for col in wq_parameters_df:
        bands = wq_parameters_df[col].dropna().to_list()
        bands_to_load.extend(bands)
    return bands_to_load


def create_ds_from_cogs(
    cog_urls: str, bands_to_load: list[str], waterbody_uid: str
) -> xr.Dataset:
    waterbody_geom = get_waterbody_geom(waterbody_uid)

    data_vars = {}
    for cog_url in cog_urls:
        band_name, _, year = parse_wq_cog_url(cog_url)
        if band_name in bands_to_load:
            da = rioxarray.open_rasterio(
                cog_url, chunks={"x": 300, "y": 300}
            ).squeeze()
            if "band" in da.coords:
                da = da.drop_vars("band")
            da = assign_crs(da, da.rio.crs)
            da = da.odc.crop(waterbody_geom)
            time_coords = np.array(
                [year_to_dc_datetime(int(year))], dtype="datetime64[ns]"
            )
            da = da.expand_dims(time=time_coords)
            data_vars[band_name] = da
    ds = xr.Dataset(data_vars)
    return ds


def normalise_water_quality_measures(
    ds: xr.Dataset, wq_parameters_csv_url: str
) -> xr.Dataset:
    wq_parameters_df = load_wq_measurements_table(wq_parameters_csv_url)

    for col in wq_parameters_df.columns:
        bands = wq_parameters_df[col].dropna().to_list()
        ds = ds.assign_coords({col: bands})

        var_name = col.replace("_measure", "")

        da_dims = ["time", "x", "y", col]
        da_coords = {dim: ds.coords[dim] for dim in da_dims}
        da_data_shape = tuple(len(da_coords[dim]) for dim in da_dims)
        da = xr.DataArray(
            data=np.zeros(da_data_shape, dtype=np.float32),
            coords=da_coords,
            dims=da_dims,
            name=var_name,
        )
        log.info(
            f"Applying scale and offset to {var_name} water quality variables"
        )
        for band in bands:
            log.debug(
                f"Applying scale and offset to {var_name} "
                f"water quality variable {band}"
            )
            scale = NORMALISATION_PARAMETERS[band]["scale"]
            offset = NORMALISATION_PARAMETERS[band]["offset"]
            da.sel({col: band})[:] = ds[band] * scale + offset

        ds[var_name] = da
        ds = ds.drop_vars(bands, errors="ignore")

    return ds


def load_water_quality_measures(
    waterbody_uid: str, wq_measures_dir: str
) -> xr.Dataset:
    log.info(f"Loading data for waterbody {waterbody_uid}")
    grouped_by_year_and_tile = get_wq_measures_cogs(
        waterbody_uid, wq_measures_dir
    )
    per_year_ds = []
    for year, grouped_by_tile in grouped_by_year_and_tile.items():
        log.info(f"Loading data for year {year}")
        per_tile_ds = []
        for tile_id, cog_urls in grouped_by_tile.items():
            log.info(f"Loading data for tile {get_region_code(tile_id)}")
            wq_parameters_csv_url = get_wq_csv_url(
                output_directory=wq_measures_dir, tile_id=tile_id, year=year
            )
            bands_to_load = get_bands_to_load(wq_parameters_csv_url)

            ds = create_ds_from_cogs(cog_urls, bands_to_load, waterbody_uid)
            ds = ds.compute()
            ds = normalise_water_quality_measures(ds, wq_parameters_csv_url)
            per_tile_ds.append(ds)
        ds = xr.merge(per_tile_ds)
        per_year_ds.append(ds)

    ds = xr.concat(per_year_ds, dim="time")
    return ds
