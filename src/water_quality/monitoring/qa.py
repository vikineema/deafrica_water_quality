import logging

import numpy as np
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


def relative_albedo_deviation(
    single_day_instrument_data: dict[str, xr.Dataset],
    composite_instrument_data: dict[str, xr.Dataset],
    comparison_type_name: str,
    composite_scaling_band: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Per pixel relative albdedo deviation (RALB) is calculated by summing
    the band values and comparing with a geomedian. The `bcmad` geomedian
    band is used to normalize to relative values.
    RALB values can be negative.
    -n < |RALB| < n is taken as the default range of normal/typical,
    where n = 1.4

    Parameters
    ----------
    single_day_instrument_data : dict[str, xr.Dataset]
        _description_
    composite_instrument_data : dict[str, xr.Dataset]
        _description_
    comparison_type_name : str
        _description_
    composite_scaling_band : str
        _description_

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Relative albedo deviation and albedo
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
    ds_agm = composite_instrument_data[composite_instrument][
        composite_instrument_bands
    ]
    ds_agm = ds_agm.fillna(0)

    albedo_gm = ds_agm.groupby("time.year").apply(_sum_variables_per_year)

    albedo_divisor = (
        composite_instrument_data[composite_instrument][composite_scaling_band]
        * 10000
    )  # this is the natural divisor to normalise the divergence
    albedo_divisor = albedo_divisor.fillna(0)
    albedo_divisor = _convert_time_coord_to_year(albedo_divisor)

    single_day_instrument = [
        i for i in list(comparison_type.keys()) if i != composite_instrument
    ][0]
    single_day_instrument_bands = comparison_type[single_day_instrument]
    ds = single_day_instrument_data[single_day_instrument][
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


def _per_timestep_mulltiplication(
    x: xr.DataArray, annual_da: xr.DataArray
) -> xr.DataArray:
    year = pd.to_datetime(x.time.values.item()).year
    result = x * annual_da.sel(year=year)
    return result


def _per_timestep_division(
    x: xr.DataArray, annual_da: xr.DataArray
) -> xr.DataArray:
    year = pd.to_datetime(x.time.values.item()).year
    result = x / annual_da.sel(year=year)
    return result


def relative_spectral_angle_deviation(
    single_day_instruments_data: dict[str, xr.Dataset],
    composite_instruments_data: dict[str, xr.Dataset],
    comparison_type_name: str,
    composite_scaling_band: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Per pixel relative spectral angle deviation  (RSAD) is calculated as
    the spectral angle between the pixel and the geomedian and normalised
    using the smad geomedian band.
    RSAD < n is taken as the default range of normal/typical, where n = 1.4

    Parameters
    ----------
    single_day_instruments_data : dict[str, xr.Dataset]
        _description_
    composite_instruments_data : dict[str, xr.Dataset]
        _description_
    comparison_type_name : str
        _description_
    composite_scaling_band : str
        _description_

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Relative spectral angle deviation and spectral angle deviation.
    """

    if comparison_type_name not in COMPARISON_TO_AGM_TYPES:
        raise ValueError(
            f"Select comparison type from {','.join(list(COMPARISON_TO_AGM_TYPES.keys))}"
        )

    log.info(
        "Calculating relative spectral angle deviations (rsad) from the geomedian... "
    )
    comparison_type = COMPARISON_TO_AGM_TYPES[comparison_type_name]

    composite_instrument = [
        i for i in list(comparison_type.keys()) if "_agm" in i
    ][0]
    composite_instrument_bands = comparison_type[composite_instrument]
    ds_agm = composite_instruments_data[composite_instrument][
        composite_instrument_bands
    ]
    ds_agm = ds_agm.fillna(0)
    ds_agm = _convert_time_coord_to_year(ds_agm)
    gm_self_product = np.sqrt(np.square(ds_agm).to_array().sum(dim="variable"))

    single_day_instrument = [
        i for i in list(comparison_type.keys()) if i != composite_instrument
    ][0]
    single_day_instrument_bands = comparison_type[single_day_instrument]
    ds = single_day_instruments_data[single_day_instrument][
        single_day_instrument_bands
    ]
    ds = ds.fillna(0)
    self_product = np.sqrt(np.square(ds).to_array().sum(dim="variable"))

    # Check
    assert len(composite_instrument_bands) == len(single_day_instrument_bands)

    dot_product = 0
    for idx in range(0, len(composite_instrument_bands)):
        band = single_day_instrument_bands[idx]
        agm_band = composite_instrument_bands[idx]

        dot_product += (
            ds[band]
            .groupby("time")
            .map(_per_timestep_mulltiplication, annual_da=ds_agm[agm_band])
        )

    # relative spectral angle deviation is the 1-cosine of the angle,
    # divided by the smad
    cosdist = dot_product / self_product.groupby("time").map(
        _per_timestep_mulltiplication, annual_da=gm_self_product
    )
    sad = 1 - cosdist

    gm_divisor = composite_instruments_data[composite_instrument][
        composite_scaling_band
    ]
    gm_divisor = gm_divisor.fillna(0)
    gm_divisor = _convert_time_coord_to_year(gm_divisor)
    rsad = sad.groupby("time").map(
        _per_timestep_division, annual_da=gm_divisor
    )
    return rsad, sad


def calculate_qa_scores(
    single_day_instruments_data: dict[str, xr.Dataset],
    composite_instruments_data: dict[str, xr.Dataset],
    comparison_type_name: str,
    composite_scaling_band_alb: str,
    composite_scaling_band_sad: str,
    scaling_factor: float,
) -> tuple[xr.DataArray, ...]:
    """_summary_

    Parameters
    ----------
    single_day_instruments_data : dict[str, xr.Dataset]
        _description_
    composite_instruments_data : dict[str, xr.Dataset]
        _description_
    comparison_type_name : str
        _description_
    composite_scaling_band_alb : str
        _description_
    composite_scaling_band_sad : str
        _description_
    scaling_factor : float
        _description_

    Returns
    -------
    tuple[xr.DataArray, ...]
        The relative albedo, albedo, relative spectral deviation, spectral
        deviation, QA score and dataset level deviation.
    """
    # allow more variation in spectral angle than in brightness when
    # calculating the combined QA measure
    rsad_factor = 0.4

    ralb, alb = relative_albedo_deviation(
        single_day_instruments_data,
        composite_instruments_data,
        comparison_type_name,
        composite_scaling_band_alb,
    )
    # --- compensate for not including the IR bands ---
    ralb = scaling_factor * ralb
    alb = scaling_factor * alb

    rsad, sad = relative_spectral_angle_deviation(
        single_day_instruments_data,
        composite_instruments_data,
        comparison_type_name,
        composite_scaling_band_sad,
    )
    # --- compensate for not including the IR bands ---
    rsad = scaling_factor * rsad
    sad = scaling_factor * sad

    # Combine as the magnitude of the qa vector
    qa_score = np.square(ralb) + np.sqrt(np.square(rsad * rsad_factor))
    ci_score = (rsad * rsad_factor) - np.square(ralb)

    return ralb, alb, rsad, sad, qa_score, ci_score


def _per_timestep_watermasking(
    x: xr.DataArray, wofs_ann_freq: xr.DataArray
) -> xr.DataArray:
    year = pd.to_datetime(x.time.values.item()).year
    water_mask = _convert_time_coord_to_year(wofs_ann_freq).sel(year=year)
    result = x.where(water_mask > 0.5, np.nan)
    return result


def calculate_qa_scores_all_instruments(
    single_day_instruments_data: dict[str, xr.Dataset],
    composite_instruments_data: dict[str, xr.Dataset],
) -> xr.Dataset:
    if "msi" in single_day_instruments_data.keys():
        comparison_type_name = "msi-v-msi_agm-noIRband"
        composite_scaling_band_alb = "msi_agm_bcmad"
        composite_scaling_band_sad = "msi_agm_smad"
        scaling_factor = 6 / 4

        (
            msi_qa_ralb,
            msi_qa_alb,
            msi_qa_rsad,
            msi_qa_sad,
            msi_qa_score,
            msi_ci_score,
        ) = calculate_qa_scores(
            single_day_instruments_data,
            composite_instruments_data,
            comparison_type_name,
            composite_scaling_band_alb,
            composite_scaling_band_sad,
            scaling_factor,
        )
        single_day_instruments_data["msi"]["msi_qa_ralb"] = msi_qa_ralb
        single_day_instruments_data["msi"]["msi_qa_alb"] = msi_qa_alb
        single_day_instruments_data["msi"]["msi_qa_rsad"] = msi_qa_rsad
        single_day_instruments_data["msi"]["msi_qa_sad"] = msi_qa_sad
        single_day_instruments_data["msi"]["msi_qa_score"] = msi_qa_score
        single_day_instruments_data["msi"]["msi_ci_score"] = msi_ci_score

    if "oli" in single_day_instruments_data.keys():
        comparison_type_name = "oli-v-oli_agm-noIRband"
        composite_scaling_band_alb = "oli_agm_bcmad"
        composite_scaling_band_sad = "oli_agm_smad"
        scaling_factor = 6 / 5

        # ralb, alb, rsad, sad, qa_score, ci_score
        (
            oli_qa_ralb,
            oli_qa_alb,
            oli_qa_rsad,
            oli_qa_sad,
            oli_qa_score,
            oli_ci_score,
        ) = calculate_qa_scores(
            single_day_instruments_data,
            composite_instruments_data,
            comparison_type_name,
            composite_scaling_band_alb,
            composite_scaling_band_sad,
            scaling_factor,
        )
        single_day_instruments_data["oli"]["oli_qa_ralb"] = oli_qa_ralb
        single_day_instruments_data["oli"]["oli_qa_alb"] = oli_qa_alb
        single_day_instruments_data["oli"]["oli_qa_rsad"] = oli_qa_rsad
        single_day_instruments_data["oli"]["oli_qa_sad"] = oli_qa_sad
        single_day_instruments_data["oli"]["oli_qa_score"] = oli_qa_score
        single_day_instruments_data["oli"]["oli_ci_score"] = oli_ci_score

    if "tm" in single_day_instruments_data.keys():
        # ralb, alb, rsad, sad, qa_score, ci_score

        original_tm_data = single_day_instruments_data["tm"]
        pre_2013 = original_tm_data.where(
            original_tm_data.time.dt.year < 2013, drop=True
        )
        post_2012 = original_tm_data.where(
            original_tm_data.time.dt.year >= 2013, drop=True
        )

        collected = {
            "tm_qa_ralb": [],
            "tm_qa_alb": [],
            "tm_qa_rsad": [],
            "tm_qa_sad": [],
            "tm_qa_score": [],
            "tm_ci_score": [],
        }

        if pre_2013.to_array().size != 0:
            single_day_instruments_data["tm"] = pre_2013
            comparison_type_name = "tm-v-tm_agm-noIRband"
            composite_scaling_band_alb = "tm_agm_bcmad"
            composite_scaling_band_sad = "tm_agm_smad"
            scaling_factor = 6 / 5
            ralb, alb, rsad, sad, qa_score, ci_score = calculate_qa_scores(
                single_day_instruments_data,
                composite_instruments_data,
                comparison_type_name,
                composite_scaling_band_alb,
                composite_scaling_band_sad,
                scaling_factor,
            )
            collected["tm_qa_ralb"].append(ralb)
            collected["tm_qa_alb"].append(alb)
            collected["tm_qa_rsad"].append(rsad)
            collected["tm_qa_sad"].append(sad)
            collected["tm_qa_score"].append(qa_score)
            collected["tm_ci_score"].append(ci_score)

        if post_2012.to_array().size != 0:
            single_day_instruments_data["tm"] = post_2012
            comparison_type_name = "tm-v-oli_agm-noIRband"
            composite_scaling_band_alb = "oli_agm_bcmad"
            composite_scaling_band_sad = "oli_agm_smad"
            scaling_factor = 6 / 5
            ralb, alb, rsad, sad, qa_score, ci_score = calculate_qa_scores(
                single_day_instruments_data,
                composite_instruments_data,
                comparison_type_name,
                composite_scaling_band_alb,
                composite_scaling_band_sad,
                scaling_factor,
            )
            collected["tm_qa_ralb"].append(ralb)
            collected["tm_qa_alb"].append(alb)
            collected["tm_qa_rsad"].append(rsad)
            collected["tm_qa_sad"].append(sad)
            collected["tm_qa_score"].append(qa_score)
            collected["tm_ci_score"].append(ci_score)

        for key, da_list in collected.items():
            if da_list:
                collected[key] = xr.concat(da_list, dim="time")
            else:
                del collected[key]

        single_day_instruments_data["tm"] = original_tm_data.update(collected)

    combined = xr.merge(list(single_day_instruments_data.values()))

    # Trim to water areas
    combined = combined.groupby("time").map(
        _per_timestep_watermasking,
        wofs_ann_freq=composite_instruments_data["wofs_ann"]["wofs_ann_freq"],
    )
    return combined
