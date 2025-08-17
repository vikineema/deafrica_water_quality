import logging

import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# specify the variables to compare when calculating spectral angle and albedo divergences against the geomedian
# Spelling out all the bands to allow contingency here for comparison of tm with oli_agm (which allows us to
# extend the use of tm past 2013, esp to include L7 data   ---
# the 'noIRband' version remove the IR band from the assessment.
# The IR band picks up floating algae giving wide variations in spectral angle relative to the geomedian; these deviations are valid
# for wq purposes, but unusual enough to not be anticipated by the geomedian SMAD. This option allows those extremes to pass
# the QA test.

COMPARISON_TO_AGM_TYPES = {
    "tm-v-oli_agm": {
        "tm": ["tm01", "tm02", "tm03", "tm04", "tm05", "tm07"],
        "oli_agm": [
            "oli02_agm",
            "oli03_agm",
            "oli04_agm",
            "oli05_agm",
            "oli06_agm",
            "oli07_agm",
        ],
    },
    "tm-v-oli_agm-noIRband": {
        "tm": ["tm01", "tm02", "tm03", "tm05", "tm07"],
        "oli_agm": [
            "oli02_agm",
            "oli03_agm",
            "oli04_agm",
            "oli06_agm",
            "oli07_agm",
        ],
    },
    "msi-v-msi_agm": {
        "msi": ["msi02", "msi03", "msi04", "msi05", "msi06", "msi07"],
        "msi_agm": [
            "msi02_agm",
            "msi03_agm",
            "msi04_agm",
            "msi05_agm",
            "msi06_agm",
            "msi07_agm",
        ],
    },
    "msi-v-msi_agm-noIRband": {
        "msi": ["msi02", "msi03", "msi04", "msi05"],
        "msi_agm": ["msi02_agm", "msi03_agm", "msi04_agm", "msi05_agm"],
    },
    "msi-v-oli_agm-noIRband": {
        "msi": ["msi02", "msi03", "msi04", "msi05"],
        "oli_agm": [
            "oli02_agm",
            "oli03_agm",
            "oli04_agm",
            "oli05_agm",
        ],  # check that this is the right set.
    },
    "oli-v-oli_agm": {
        "oli": ["oli02", "oli03", "oli04", "oli05", "oli06", "oli07"],
        "oli_agm": [
            "oli02_agm",
            "oli03_agm",
            "oli04_agm",
            "oli05_agm",
            "oli06_agm",
            "oli07_agm",
        ],
    },
    "oli-v-oli_agm-noIRband": {
        "oli": ["oli02", "oli03", "oli04", "oli06", "oli07"],
        "oli_agm": [
            "oli02_agm",
            "oli03_agm",
            "oli04_agm",
            "oli06_agm",
            "oli07_agm",
        ],
    },
    "tm-v-tm_agm": {
        "tm": ["tm01", "tm02", "tm03", "tm04", "tm05", "tm07"],
        "tm_agm": [
            "tm01_agm",
            "tm02_agm",
            "tm03_agm",
            "tm04_agm",
            "tm05_agm",
            "tm07_agm",
        ],
    },
    "tm-v-tm_agm-noIRband": {
        "tm": ["tm01", "tm02", "tm03", "tm05", "tm07"],
        "tm_agm": ["tm01_agm", "tm02_agm", "tm03_agm", "tm05_agm", "tm07_agm"],
    },
}


def _convert_time_coord_to_year(
    x: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    x = x.assign_coords(time=x["time"].dt.year)
    x = x.rename({"time": "year"})
    return x


def _sum_variables_per_year(x: xr.Dataset) -> xr.Dataset:
    da = x.to_array().sum(dim="variable")
    da = _convert_time_coord_to_year(da)
    return da


def per_pixel_relative_albedo_deviation(
    single_day_instrument_ds: xr.Dataset,
    composite_instrument_ds: xr.Dataset,
    comparison_type_name: str,
    composite_scaling_band: str,
):
    """
    Per pixel relative albdedo deviation (RALB) is calculated by summing
    the band values and comparing with a geomedian. The `bcmad` geomedian
    band is used to normalize to relative values.
    RALB values can be negative.
    -n < |RALB| < n is taken as the default range of normal/typical,
    where n = 1.4
    """

    if comparison_type_name not in COMPARISON_TO_AGM_TYPES:
        raise ValueError(
            f"Select comparison type from {','.join(list(COMPARISON_TO_AGM_TYPES.keys()))}"
        )
    log.info(
        "Calculating relative albedo deviations (ralb) from the geomedian ... "
    )
    comparison_type = COMPARISON_TO_AGM_TYPES[comparison_type_name]

    composite_instrument = [
        i for i in list(comparison_type.keys()) if "_agm" in i
    ][0]
    composite_instrument_bands = comparison_type[composite_instrument]
    ds_agm = composite_instrument_ds[composite_instrument][
        composite_instrument_bands
    ]
    ds_agm = ds_agm.fillna(0)

    albedo_gm = ds_agm.groupby("time.year").apply(_sum_variables_per_year)

    albedo_divisor = (
        composite_instrument_ds[composite_instrument][[composite_scaling_band]]
        * 10000
    )  # this is the natural divisor to normalise the divergence
    albedo_divisor = albedo_divisor.fillna(0)
    albedo_divisor = _convert_time_coord_to_year(albedo_divisor)

    single_day_instrument = [
        i for i in list(comparison_type.keys()) if i != composite_instrument
    ][0]
    single_day_instrument_bands = comparison_type[single_day_instrument]
    ds = single_day_instrument_ds[single_day_instrument][
        single_day_instrument_bands
    ]
    ds = ds.fillna(0)
    albedo = ds.to_array().sum(dim="variable")

    def _per_timestep_relative_albedo(x, albedo_gm_ds, albedo_divisor_ds):
        year = pd.to_datetime(x.time.values.item()).year
        ra = (x - albedo_gm_ds.sel(year=year)) / albedo_divisor_ds.sel(
            year=year
        )
        return ra

    relative_albedo = albedo.groupby("time").map(
        _per_timestep_relative_albedo,
        albedo_gm_ds=albedo_gm,
        albedo_divisor_ds=albedo_divisor,
    )
    return relative_albedo, albedo
