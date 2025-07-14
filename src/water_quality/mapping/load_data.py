import logging
from typing import Any

import dask
import numpy as np
import xarray as xr
from datacube import Datacube
from odc.geo.geobox import GeoBox

from water_quality.dates import year_to_dc_datetime
from water_quality.mapping.instruments import INSTRUMENTS_MEASUREMENTS

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
            f"Datacube products for the instrument {instrument_name} "
            "are not defined."
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
            f"Datacube measurements for the instrument {instrument_name} "
            "are not defined."
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
    Get the dictionary for re-naming measurements to have unique
    dataset variable names for the loaded data for an instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument

    Returns
    -------
    dict[str, tuple[str]]
        Dictionary whose keys are the datacube measurements loaded for
        the instrument and whose values are the desired names for the
        measurements

    """
    measurements = INSTRUMENTS_MEASUREMENTS.get(instrument_name, None)
    if measurements is None:
        raise NotImplementedError(
            f"Datacube measurements for the instrument {instrument_name} "
            "are not defined."
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
    dask_chunks: dict[str, int] = {},
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
    dask_chunks:  dict[str, int]
        Number of chunks for dask arrays, by default {}
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
                dask_chunks=dask_chunks,
                # align=(0, 0), not supported when using like
            )
            dc_queries[instrument_name] = dc_query
    return dc_queries


def process_st_data_to_annnual(
    ds_tirs: xr.Dataset, ds_wofs_ann: xr.Dataset
) -> xr.Dataset:
    """
    Processes daily surface temperature data into an annual time series,
    masked by water extent derived from WOfS annual statistics.

    The function performs the following steps:
    - Converts raw surface temperature values to degrees Celsius.
    - Filters data by temperature validity, quality assurance threshold,
      and high surface emissivity.
    - Aggregates daily temperature data into annual median, 10th
        percentile (min), and 90th percentile (max) statistics.
    - Masks aggregated temperature data to include only pixels with
      consistent water presence based on the WOfS annual frequency
      threshold.

    Parameters
    ----------
    ds_tirs : xr.Dataset
        An xarray Dataset containing daily surface temperature data
        with bands:
            - 'tirs_st'
            - 'tirs_st_qa'
            - 'tirs_emis'

    ds_wofs_ann : xr.Dataset
        An xarray Dataset of Water Observations from Space (WOfS) annual
        water frequency statistics. It must include the band:
        'wofs_ann_freq'

    Returns
    -------
    xr.Dataset
        An annual xarray Dataset with the following data variables:
            - 'tirs_st_ann_med': Median annual surface temperature (°C).
            - 'tirs_st_ann_min': 10th percentile annual temperature (°C).
            - 'tirs_st_ann_max': 90th percentile annual temperature (°C).

        All layers are masked to include only pixels where the WOfS
        annual water frequency exceeds 0.5.
    """
    # Rescale the daily timeseries to centigrade, remove outliers,
    # apply quality filter and also filter on emissivity > 0.95.
    tirs_st = (ds_tirs["tirs_st"] * 0.00341802 + 149.0) - 273.15
    # -- uncertainty in kelvin
    tirs_st_qa = ds_tirs["tirs_st_qa"] * 0.01
    # -- emissivity fraction
    tirs_emis = ds_tirs["tirs_emis"] * 0.0001

    valid_mask = (tirs_st > 0) & (tirs_st_qa < 5) & (tirs_emis > 0.95)
    ds_tirs["tirs_st"] = tirs_st.where(valid_mask)

    # Drop intermediate arrays to reduce memory
    del tirs_st, tirs_st_qa, tirs_emis, valid_mask

    # If dask backed
    if ds_tirs.chunks is not None:
        # Rechunk so the time dimension has only one chunk
        # for the quantile commputation
        ds_tirs = ds_tirs.chunk({"time": ds_tirs.sizes["time"]})

    # Create an empty annual dataset
    annual_ds_tirs = xr.Dataset()

    # Average the temperatures up to years - min, max and mean
    group = ds_tirs["tirs_st"].groupby("time.year")
    annual_ds_tirs["tirs_st_ann_med"] = group.median(dim="time")
    annual_ds_tirs["tirs_st_ann_min"] = group.quantile(0.1, dim="time")
    annual_ds_tirs["tirs_st_ann_max"] = group.quantile(0.9, dim="time")

    # Replace the year coordinate with datetime64[ns] time coordinate
    annual_ds_tirs = annual_ds_tirs.rename({"year": "time"})
    time_values = np.array(
        [year_to_dc_datetime(i) for i in annual_ds_tirs.time.values],
        dtype="datetime64[ns]",
    )
    annual_ds_tirs = annual_ds_tirs.assign_coords(time=time_values)

    # Restrict values to areas of water
    water_frequency_threshold = 0.5
    water_mask = (
        ds_wofs_ann["wofs_ann_freq"].sel(time=time_values)
        > water_frequency_threshold
    )

    for var in ["tirs_st_ann_med", "tirs_st_ann_min", "tirs_st_ann_max"]:
        annual_ds_tirs[var] = annual_ds_tirs[var].where(water_mask)

    # Clean up
    # annual_ds_tirs = annual_ds_tirs.compute()
    # annual_ds_tirs = annual_ds_tirs.drop_vars("quantile")
    return annual_ds_tirs


def fix_wofs_all_time(ds: xr.Dataset) -> xr.Dataset:
    """
    This is a work around to get data for the `wofs_all` instrument,
    i.e. data loaded from the DE Africa `wofs_ls_summary_alltime`
    product, to have the same time dimension (year) as data loaded from
    all other instruments in a dataset.
    > This only works if data loaded for all other instruments apart from
    `wofs_all` was loaded for **one specific year only** using
    `build_wq_agm_dataset`.


    Parameters
    ----------
    ds : xr.Dataset
        Dataset built using `build_wq_agm_dataset`.

    Returns
    -------
    xr.Dataset
        Input dataset with updated time dimension.
    """
    time_values = list(ds.time.values)
    count = len(time_values)
    if count < 1 or count > 2:
        raise ValueError(
            f"Expecting data for a single year. Found data for {time_values}"
        )
    else:
        if count == 2:
            wofs_all_vars = list(
                get_measurements_name_dict("wofs_all").values()
            )
            # Recombine datasets with the correct time dimensions
            ds_list = []
            for var in list(ds.data_vars):
                da = ds[var]
                all_nan = da.isnull().all(dim=["y", "x"])
                # Remove empty time steps.
                da = da.sel(time=~all_nan)
                if var in wofs_all_vars:
                    new_wofs_all_time = [
                        i for i in time_values if i != da.time.values
                    ][0]
                    da = da.assign_coords(time=[new_wofs_all_time])
                ds_list.append(da)
            ds = xr.merge(ds_list)
        return ds


def build_wq_agm_dataset(
    dc_queries: dict[str, dict[str, Any]],
    dc: Datacube = None,
    single_year: bool = False,
) -> xr.Dataset:
    """Build a combined annual dataset from loading data
    for each instrument using the datacube queries provided.

    Parameters
    ----------
    dc_queries : dict[str, dict[str, Any]]
        Datacube query to use to load data for each instrument.

    single_year : bool
        Specify if data is being loaded for **one specific year only**.
        If it is and data for the `wofs_all` instrument is loaded, the
        `wofs_all` DataArrays will be assigned the time value matching
        other annual datasets.

    Returns
    -------
    xr.Dataset
        A single dataset containing all the data found for each instrument
        in the datacube.
    """
    if dc is None:
        dc = Datacube()

    loaded_data: dict[str, xr.Dataset] = {}
    for instrument_name, dc_query in dc_queries.items():
        ds = dc.load(**dc_query)
        ds = ds.rename(get_measurements_name_dict(instrument_name))
        loaded_data[instrument_name] = ds

    # Process temperature data to an annual timeseries
    if "tirs" in loaded_data.keys():
        if "wofs_ann" not in loaded_data.keys():
            raise ValueError(
                "Data for the wofs_ann instrument is required to process "
                "daily surface temperature data for the tirs instrument into "
                "an annual timeseries."
            )
        else:
            loaded_data["tirs"] = process_st_data_to_annnual(
                ds_tirs=loaded_data["tirs"],
                ds_wofs_ann=loaded_data["wofs_ann"],
            )

    # All datasets expect those for the instrument wofs_all
    # are expected to have the same time dimensions.
    results = dask.compute(*list(loaded_data.values()))
    combined = xr.merge(results)
    combined = combined.drop_vars("quantile", errors="ignore")

    if single_year:
        combined = fix_wofs_all_time(combined)

    return combined
