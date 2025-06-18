import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from datacube import Datacube
from odc.geo.geom import Geometry

LWQ_PRODUCTS = [
    "cgls_lwq300_2002_2012",
    "cgls_lwq300_2016_2024",
    "cgls_lwq300_2024_nrt",
]
LWQ_MEASUREMENTS = ["TSI", "turbidity"]
DEVIATION_CATEGORIES = {"low": 1, "medium": 2, "high": 3, "extreme": 4}


def get_monthly_timeseries(
    time_range: tuple[str], waterbody_geom: Geometry, dc: Datacube = None
):
    if dc is None:
        dc = Datacube()

    ds = dc.load(
        product=LWQ_PRODUCTS,
        measurements=LWQ_MEASUREMENTS,
        geopolygon=waterbody_geom,
        output_crs="EPSG:6933",
        resolution=(-300, 300),
        time=time_range,
    )

    # Mask the no data value
    for var in LWQ_MEASUREMENTS:
        ds[var] = ds[var].where(ds[var] != float(ds[var].nodata))

    # Resample decadal timeseries to monthly timeseries
    monthly_timeseries = ds.resample(time="1ME").mean()

    return monthly_timeseries


def get_monthly_multiannual_baseline(
    baseline_period: tuple[str], waterbody_geom: Geometry, dc: Datacube = None
) -> xr.Dataset:
    if dc is None:
        dc = Datacube()

    baseline_monthly_timeseries = get_monthly_timeseries(
        time_range=baseline_period, waterbody_geom=waterbody_geom, dc=dc
    )

    # Get the long term average for each month
    baseline_monthly_long_term_mean = baseline_monthly_timeseries.groupby(
        "time.month"
    ).mean()

    return baseline_monthly_long_term_mean


def get_target_years_monthly_avgs(
    target_period: tuple[str],
    waterbody_geom: Geometry,
    dc: Datacube = None,
):
    if dc is None:
        dc = Datacube()

    monthly_timeseries = get_monthly_timeseries(
        time_range=target_period, waterbody_geom=waterbody_geom, dc=dc
    )
    return monthly_timeseries


def get_monthly_deviations(
    target_years_monthly_avgs: xr.Dataset,
    monthly_multiannual_baseline: xr.Dataset,
) -> xr.Dataset:
    def _get_deviations(x):
        # This does not change the pixel values as
        # the data is a monthly timeseries already.
        # This is to get the coordinates of the dataset to
        # match coordinates in monthly_multiannual_baseline
        # i.e. replace the time coordinate with the month coordinate.
        monthly_averages = x.groupby("time.month").mean()
        # Calculate deviation
        monthly_deviations = (
            (monthly_averages - monthly_multiannual_baseline)
            / monthly_multiannual_baseline
        ) * 100
        return monthly_deviations

    monthly_deviations = target_years_monthly_avgs.groupby("time.year").map(
        _get_deviations
    )

    # Rework back to time coordinate instead of year and month
    # as seperate coordinates
    years = monthly_deviations.year.values
    months = monthly_deviations.month.values

    stack = []
    for year in years:
        for month in months:
            time_coord = np.datetime64(
                datetime(year, month, calendar.monthrange(year, month)[1]), "ns"
            )
            ds_sel = monthly_deviations.sel(year=year, month=month).drop_vars(
                ["year", "month"]
            )
            ds_sel = ds_sel.assign_coords(coords={"time": time_coord}).expand_dims(
                dim={"time": 1}
            )
            stack.append(ds_sel)

    monthly_deviations = xr.concat(stack, dim="time")

    return monthly_deviations


def get_deviation_categories(var: str, deviations_ds: xr.Dataset) -> np.ndarray:
    conditions = [
        (deviations_ds[var] <= 25),
        ((deviations_ds[var] > 25) & (deviations_ds[var] <= 50)),
        ((deviations_ds[var] > 50) & (deviations_ds[var] <= 100)),
        (deviations_ds[var] > 100),
    ]
    classes = list(DEVIATION_CATEGORIES.values())
    data = np.select(conditions, classes, default=np.nan)
    return data


def add_deviation_categories_attrs(ds: xr.Dataset) -> xr.Dataset:
    attrs = ds.attrs
    if attrs:
        if "deviation_categories" not in list(attrs.keys):
            attrs["deviation_categories"] = DEVIATION_CATEGORIES
    else:
        attrs = {"deviation_categories": DEVIATION_CATEGORIES}

    ds.attrs = attrs
    return ds


def classify_deviations(deviations_ds: xr.Dataset) -> xr.Dataset:
    deviation_classes = xr.full_like(deviations_ds, fill_value=np.nan)

    for var in LWQ_MEASUREMENTS:
        deviation_classes[var].data = get_deviation_categories(
            var=var, deviations_ds=deviations_ds
        )

    deviation_classes = add_deviation_categories_attrs(deviation_classes)

    return deviation_classes


def get_annual_deviations(monthly_deviations: xr.Dataset) -> xr.Dataset:
    annual_deviations = monthly_deviations.groupby("time.year").mean()
    return annual_deviations
