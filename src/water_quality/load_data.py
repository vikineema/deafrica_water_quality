import logging
from typing import Any

import xarray as xr
from datacube import Datacube
from odc.geo.geobox import GeoBox

from water_quality.instruments import INSTRUMENTS_MEASUREMENTS

log = logging.getLogger(__name__)

INSTRUMENTS_PRODUCTS = {
    "tm_agm": ["gm_ls5_ls7_annual"],
    "oli_agm": ["gm_ls8_annual", "gm_ls8_ls9_annual"],
    "msi_agm": ["gm_s2_annual"],
    "tirs": ["ls5_st", "ls7_st", "ls8_st", "ls9_st"],
    "wofs_ann": ["wofs_ls_summary_annual"],
    "wofs_all": ["wofs_ls_summary_alltime"],
}


def get_dc_products(instrument_name: str) -> list[str]:
    """
    Get the datacube products to load for a given instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instruments

    Returns
    -------
    list[str]
        Datacube products to load for an instrument

    """
    dc_products = INSTRUMENTS_PRODUCTS.get(instrument_name, None)
    if dc_products is None:
        raise NotImplementedError(
            f"Datacube products for the instrument {instrument_name} are not defined."
        )
    else:
        return dc_products


def get_dc_measurements(instrument_name: str) -> list[str]:
    """
    Get the datacube measurements to load for a given instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument

    Returns
    -------
    list[str]
        Datacube measurements to load for the instrument

    """
    measurements = INSTRUMENTS_MEASUREMENTS.get(instrument_name, None)
    if measurements is None:
        raise NotImplementedError(
            f"Datacube measurements for the instrument {instrument_name} are not defined."
        )
    else:
        dc_measurements: list[str] = []
        for measurement_name, measurement_info in measurements.items():
            is_required = measurement_info["parameters"][0]
            assert isinstance(is_required, bool)
            if is_required is True:
                dc_measurements.append(measurement_name)
            else:
                continue
        return dc_measurements


def get_measurements_name_dict(instrument_name: str) -> dict[str, tuple[str]]:
    """
    Get the dictionary for re-naming measurements to have unique dataset variable name
    for the loaded data for an instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument

    Returns
    -------
    dict[str, tuple[str]]
        Dictionary whose keys are the datacube measurements loaded for the instrument
        and whose values are the desired names for the measurements

    """
    measurements = INSTRUMENTS_MEASUREMENTS.get(instrument_name, None)
    if measurements is None:
        raise NotImplementedError(
            f"Datacube measurements for the instrument {instrument_name} are not defined."
        )
    else:
        measurements_name_dict: dict[str, tuple[str]] = {}
        for measurement_name, measurement_info in measurements.items():
            new_measurement_name: tuple[str] = measurement_info["varname"]
            is_required = measurement_info["parameters"][0]
            assert isinstance(is_required, bool)
            if is_required is True:
                measurements_name_dict[measurement_name] = new_measurement_name
            else:
                continue
        return measurements_name_dict


def build_dc_queries(
    instruments_to_use: dict[str, dict[str, bool]],
    tile_geobox: GeoBox,
    start_date: str,
    end_date: str,
    resampling: str = "bilinear",
) -> dict[str, dict[str, Any]]:
    """
    Build a reusable datacube query for each instrument to load
    data for.

    Parameters
    ----------
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the selected instruments to use for the analysis.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of data,
        including it’s crs.
    start_date : str
        The start of the time range to load data for.
    end_date : str
        The end of the time range to load data for.
    resampling : str, optional
        Resampling method to use, by default "bilinear".

    Returns
    -------
    dict[str, dict[str, Any]]
        Datacube query for each instrument.
    """
    dc_queries = {}
    for instrument_name, usage in instruments_to_use.items():
        if usage["use"] is True:
            dc_products = get_dc_products(instrument_name)
            dc_measurements = get_dc_measurements(instrument_name)
            dc_query = dict(
                product=dc_products,
                measurements=dc_measurements,
                like=tile_geobox,
                time=(start_date, end_date),
                resampling=resampling,
                # align=(0, 0), not supported when using like
            )
            dc_queries[instrument_name] = dc_query
    return dc_queries


def process_st_data_to_annnual():
    pass


def build_wq_dataset(
    dc_queries: dict[str, dict[str, Any]], dc: Datacube = None
) -> xr.Dataset:
    """Build a combined dataset from loading data
    for each instrument using the datacube queries provided.

    Parameters
    ----------
    dc_queries : dict[str, dict[str, Any]]
        Datacube query to use to load data for each instrument.

    Returns
    -------
    xr.Dataset
        A single dataset containing all the data found for each instrument
        in the datacube.
    """
    if dc is None:
        dc = Datacube()

    loaded_data = {}
    for instrument_name, dc_query in dc_queries.items():
        ds = dc.load(**dc_query)
        """
        if instrument_name == "wofs_all":
            ds = ds.squeeze(dim="time", drop=True)
        """
        ds = ds.rename(get_measurements_name_dict(instrument_name))
        loaded_data[instrument_name] = ds
    """
    combined_ds = xr.merge(
        list(loaded_data.values()),
        compat="no_conflicts",
        join="outer",
        combine_attrs="override",
    )
    """
    return loaded_data
