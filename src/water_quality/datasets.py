import logging
from typing import Any

import xarray as xr
from datacube import Datacube
from odc.geo.geobox import GeoBox

log = logging.getLogger(__name__)

INSTRUMENTS_PRODUCTS = {
    "msi": ["s2_l2a_c1"],
    "msi_agm": ["gm_s2_annual"],
    "oli": ["ls8_sr", "ls9_sr"],
    "oli_agm": ["gm_ls8_annual", "gm_ls8_ls9_annual"],
    "tirs": ["l5_st", "l7_st", "l8_st", "l9_st"],
    "tm": ["l5_sr", "l7_sr"],
    "tm_agm": ["gm_ls5_ls7_annual"],
    "wofs_ann": ["wofs_ls_summary_annual"],
    "wofs_all": ["wofs_ls_summary_alltime"],
}
INSTRUMENTS_MEASUREMENTS = {
    "wofs_ann": {
        "frequency": {"varname": ("wofs_ann_freq"), "parameters": (True, "other")},
        "count_clear": {"varname": ("wofs_ann_clearcount"), "parameters": (True,)},
        "count_wet": {"varname": ("wofs_ann_wetcount"), "parameters": (True,)},
    },
    "wofs_all": {
        "frequency": {"varname": ("wofs_all_freq"), "parameters": (True, "other")},
        "count_clear": {"varname": ("wofs_all_clearcount"), "parameters": (True,)},
        "count_wet": {"varname": ("wofs_all_wetcount"), "parameters": (True,)},
    },
    "oli_agm": {
        "SR_B2": {"varname": ("oli02_agm"), "parameters": (True, "450-510")},
        "SR_B3": {"varname": ("oli03_agm"), "parameters": (True, "530-590")},
        "SR_B4": {"varname": ("oli04_agm"), "parameters": (True, "640-670")},
        "SR_B5": {"varname": ("oli05_agm"), "parameters": (True, "850-880")},
        "SR_B6": {"varname": ("oli06_agm"), "parameters": (True, "1570-1650")},
        "SR_B7": {"varname": ("oli07_agm"), "parameters": (True, "2110-2290")},
        "smad": {"varname": ("oli_agm_smad"), "parameters": (True,)},
        "emad": {"varname": ("oli_agm_emad"), "parameters": (False,)},
        "bcmad": {"varname": ("oli_agm_bcmad"), "parameters": (False,)},
        "count": {"varname": ("oli_agm_count"), "parameters": (True,)},
    },
    "msi_agm": {
        "B02": {"varname": ("msi02_agm"), "parameters": (True, "460-525")},
        "B03": {"varname": ("msi03_agm"), "parameters": (True,)},
        "B04": {"varname": ("msi04_agm"), "parameters": (True,)},
        "B05": {"varname": ("msi05_agm"), "parameters": (True,)},
        "B06": {"varname": ("msi06_agm"), "parameters": (True,)},
        "B07": {"varname": ("msi07_agm"), "parameters": (True,)},
        "B08": {
            "varname": ("msi08_agm"),
            "parameters": (
                False,
                "uint16 	1 	0.0 	[band_08, nir, nir_1] 	NaN",
            ),
        },
        "B8A": {
            "varname": ("msi8a_agm"),
            "parameters": (
                False,
                "uint16 	1 	0.0 	[band_8a, nir_narrow, nir_2] 	NaN",
            ),
        },
        "B11": {
            "varname": ("msi11_agm"),
            "parameters": (
                False,
                "uint16 	1 	0.0 	[band_11, swir_1, swir_16] 	NaN",
            ),
        },
        "B12": {
            "varname": ("msi12_agm"),
            "parameters": (
                True,
                "uint16 	1 	0.0 	[band_12, swir_2, swir_22] 	NaN",
            ),
        },
        "smad": {"varname": ("msi05_agm_smad"), "parameters": (True,)},
        "emad": {"varname": ("msi_agm_emad"), "parameters": (False,)},
        "bcmad": {"varname": ("msi_agm_bcmad"), "parameters": (False,)},
        "count": {"varname": ("msi_agm_count"), "parameters": (True,)},
    },
    "tm_agm": {
        "SR_B1": {"varname": ("tm01_agm"), "parameters": (True, "blue 450-520")},
        "SR_B2": {"varname": ("tm02_agm"), "parameters": (True, "green 520-600")},
        "SR_B3": {"varname": ("tm03_agm"), "parameters": (True, "red   630-690")},
        "SR_B4": {"varname": ("tm04_agm"), "parameters": (True, "nir   760-900")},
        "SR_B5": {"varname": ("tm05_agm"), "parameters": (True, "swir1 1550-1750")},
        "SR_B7": {"varname": ("tm07_agm"), "parameters": (True, "swir2 2080-2350")},
        "smad": {"varname": ("tm_agm_smad"), "parameters": (True,)},
        "emad": {"varname": ("tm_agm_emad"), "parameters": (False,)},
        "bcmad": {"varname": ("tm_agm_bcmad"), "parameters": (False,)},
        "count": {"varname": ("tm_agm_count"), "parameters": (True,)},
    },
    "tirs": {
        "ST_B10": {
            "varname": ("tirs10"),
            "parameters": (
                True,
                "ST_B10, uint16 	Kelvin 	0.0 	[band_10, st, surface_temperature]",
            ),
        },
        "ST_TRAD": {
            "varname": ("tirs_trad"),
            "parameters": (
                False,
                "ST_TRAD, int16 	W/(m2.sr.μm) 	-9999.0 	[trad, thermal_radiance]",
            ),
        },
        "ST_URAD": {
            "varname": ("tirs_urad"),
            "parameters": (
                False,
                "ST_URAD 	int16 	W/(m2.sr.μm) 	-9999.0 	[urad, upwell_radiance]",
            ),
        },
        "ST_DRAD": {
            "varname": ("tirs_drad"),
            "parameters": (
                False,
                "ST_DRAD 	int16 	W/(m2.sr.μm) 	-9999.0 	[drad, downwell_radiance]",
            ),
        },
        "ST_ATRAN": {
            "varname": ("tirs_atran"),
            "parameters": (
                False,
                "ST_ATRAN 	int16 	1 	-9999.0 	[atran, atmospheric_transmittance]",
            ),
        },
        "ST_EMIS": {
            "varname": ("tirs_emis"),
            "parameters": (
                True,
                "ST_EMIS 	int16 	1 	-9999.0 	[emis, emissivity]",
            ),
        },
        "ST_EMSD": {
            "varname": ("tirs_emsd"),
            "parameters": (
                True,
                "ST_EMSD 	int16 	1 	-9999.0 	[emsd, emissivity_stddev]",
            ),
        },
        "ST_CDIST": {
            "varname": ("tirs_cdist"),
            "parameters": (
                False,
                "ST_CDIST 	int16 	Kilometers 	-9999.0 	[cdist, cloud_distance]",
            ),
        },
        "QA_PIXEL": {
            "varname": ("tirs_qa_pixel"),
            "parameters": (
                False,
                "QA_PIXEL 	uint16 	bit_index 	1.0 	[pq, pixel_quality]",
            ),
        },
        "QA_RADSAT": {
            "varname": ("tirs_radsat"),
            "parameters": (
                False,
                "QA_RADSAT 	uint16 	bit_index 	0.0 	[radsat, radiometric_saturation]",
            ),
        },
        "ST_QA": {
            "varname": ("tirs_qa_st"),
            "parameters": (
                True,
                "ST_QA 	int16 	Kelvin 	-9999.0 	[st_qa, surface_temperature_quality]",
            ),
        },
    },
}


def get_dc_products(instrument_name: str) -> list[str]:
    dc_products = INSTRUMENTS_PRODUCTS.get(instrument_name, None)
    if dc_products is None:
        raise NotImplementedError(
            f"Datacube products for the instrument {instrument_name} are not defined."
        )
    else:
        return dc_products


def get_dc_measurements(instrument_name: str) -> list[str]:
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


def build_wq_agm_dataset(dc_queries: dict[str, dict[str, Any]]) -> xr.Dataset:
    dc = Datacube()
    loaded_data = {}
    for instrument_name, dc_query in dc_queries.items():
        ds = dc.load(**dc_query)
        ds = ds.rename(get_measurements_name_dict(instrument_name))
        loaded_data[instrument_name] = ds
    combined_ds = xr.merge(
        list(loaded_data.values()),
        compat="no_conflicts",
        join="outer",
        combine_attrs="override",
    )
    return combined_ds
