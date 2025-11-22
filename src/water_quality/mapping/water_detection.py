"""
This module provides functions to implement water detection using DE
Africa Water Observations from Space (WOfS) products.
"""

import logging

import dask.array
import numpy as np
import xarray as xr
from datacube import Datacube
from datacube.model import Dataset
from odc.geo.geobox import GeoBox

from water_quality.mapping.load_data import get_measurements_name_dict

log = logging.getLogger(__name__)


def _sum_over_time(da: xr.DataArray) -> xr.DataArray:
    """Helper function for `load_5year_water_mask` to sum
    xr.DataArray over time dimension while handling nodata values."""
    nodata_val = da.attrs.get("nodata", None)

    if nodata_val is not None:
        da_masked = da.where(da != nodata_val, np.nan).astype("float32")
        sum_over_time = da_masked.sum(dim="time")
    else:
        sum_over_time = da.sum(dim="time")
    return sum_over_time


def load_5year_water_mask(
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    compute: bool = True,
    dc: Datacube = None,
) -> xr.DataArray:
    """Load and process 5 years data for the `wofs_ann` instrument to
    generate a water mask.

    Parameters
    ----------
    dss: dict[str, list[Dataset]]
        Mapping of instrument name to list of datasets to load.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data, by default None.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the water mask derived from the
        wofs_ann data.
    """
    log.info("Generating 5 year water mask ...")

    inst = "wofs_ann"
    if inst not in list(dss.keys()):
        error = (
            f"No datasets found for instrument '{inst}'.",
            "Returning empty array.",
        )
        log.error(error)
        return xr.DataArray(data=[], dims=["time"], coords={"time": []})
    else:
        datasets = dss[inst]
        # TODO: Set a global dask chunk size configuration
        # Expected tile size is 9600 x 9 600 at 10 m resolution
        dask_chunks = {"x": 4800, "y": 4800}
        measurements = get_measurements_name_dict(inst)
        # For int data nearest is preferred
        # bilinear for float data.
        resampling = "nearest"

        log.info(f"Loading data for the instrument {inst} ...")

        if dc is None:
            dc = Datacube(app=f"Load_{inst}")

        ds = dc.load(
            datasets=datasets,
            measurements=list(measurements.keys()),
            like=tile_geobox,
            resampling=resampling,
            dask_chunks=dask_chunks,
        )
        ds = ds.rename(measurements)

        # Calculate the ratio of clear wet observations to total clear
        # observations for each pixel over the 5 year period.
        clearcount_sum = _sum_over_time(ds["wofs_ann_clearcount"])
        wet_count_sum = _sum_over_time(ds["wofs_ann_wetcount"])

        frequency_np = dask.array.full_like(
            wet_count_sum, np.nan, dtype="float32", name="frequency"
        )
        dask.array.divide(
            wet_count_sum,
            clearcount_sum,
            out=frequency_np,
            where=clearcount_sum > 0,
        )

        log.info("Processing 5 year water mask ...")
        # Bool to float32 to allow for preserving no data pixels
        water_mask_np = (frequency_np > 0.45).astype("float32")
        water_mask_np = dask.array.where(
            ~np.isnan(frequency_np), water_mask_np, np.nan
        )

        water_mask_da = xr.DataArray(
            water_mask_np,
            coords=wet_count_sum.coords,
            dims=wet_count_sum.dims,
            name="water_mask",
        )
        water_mask_da.attrs = dict(
            nodata=np.nan,
            scales=1,
            offsets=0,
        )
        # Add a time coordinate for compatibility with other annual datasets.
        # Use the last year in the 5 year period
        # based on how the tasks were generated in `wq-generate-tasks`.
        water_mask_da = water_mask_da.expand_dims(
            time=[ds.isel(time=-1).time.values]
        )

        if compute:
            log.info("Computing water mask ...")
            water_mask_da = water_mask_da.compute()
            log.info("Done.")
        else:
            water_mask_da = water_mask_da.persist()
        del ds, clearcount_sum, wet_count_sum, frequency_np, water_mask_np
        log.info("Processing for 5 year water mask completed.")
        # gc.collect()
        return water_mask_da


def water_analysis(
    wofs_ann_ds: xr.Dataset,
    water_frequency_threshold: float = 0.5,
):
    """Performs water detection analysis on DE Africa WOfS Annual
    Summary data.

    Parameters
    ----------
    wofs_ann_ds : Dataset
        An xarray Dataset containing the processed data for the
        instrument `wofs_ann` for a single year.
    water_frequency_threshold : float, optional
        The frequency threshold above which a pixel is classified as
        general water, by default 0.5

    Returns
    -------
    xarray.Dataset
        The input Dataset with the following new data variables added:
        - `wofs_ann_watermask` (float): A mask showing where general water is detected.

    """

    # Standard deviation of the annual frequency at each pixel
    # should really be dividing by n-1 but then I would need to
    # change SC
    wofs_ann_freq_sigma = (
        (wofs_ann_ds.wofs_ann_freq * (1 - wofs_ann_ds.wofs_ann_freq))
        / wofs_ann_ds.wofs_ann_clearcount
    ) ** 0.5
    # A variable called watermask is used in places.
    # I set the value of the mask as sigma or nan
    # Renamed this from watermask to wofs_ann_watermask to prevent
    # confusion with the 5 year summary watermask
    wofs_ann_ds["wofs_ann_watermask"] = wofs_ann_freq_sigma.where(
        wofs_ann_ds["wofs_ann_freq"] > water_frequency_threshold
    )

    return wofs_ann_ds
