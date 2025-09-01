"""
This module provides functions to apply various water quality algorithms
to EO data from a set of instruments.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# =============================================================================
# Water Quality Algorithms
# =============================================================================

# For each algorithm function ensure the first argument is the
# input dataset i.e data for all instruments used in the analysis.
# The function should return a single-band `xarray.DataArray`


def NDCI_NIR_R(
    dataset: xr.Dataset, NIR_band: str, red_band: str
) -> xr.DataArray:
    """
    Function to calculate ndci from input nir and red bands ----
    """
    return (dataset[NIR_band] - dataset[red_band]) / (
        dataset[NIR_band] + dataset[red_band]
    )


# Functions to estimate ChlA using the MERIS and
# MODIS 2-Band models, using closest bands from MSI (6, 5, 4) or other
def ChlA_MERIS2B(
    dataset: xr.Dataset,
    band_708: str,
    band_665: str,
) -> xr.DataArray:
    """
    MERIS two band:
    meris2b = 25.28 x MERIS_2BM^2 + 14:85 x / MERIS_2BM – 15.18
    where MERIS_2BM = band9(703.75–713.75 nm) / band7(660–670 nm)
    closest S2 bands are : B5 (703.9) and B4(664) respectively
    may need to divide by pi

    MERIS2B = 25.28 * ((ds.msi05 / ds.msi04r)*2) + 14.85 * (ds.msi05 / ds.msi04r) - 15.18
    """
    # matching MSI bands are 5 and 4
    log.info("ChlA_MERIS two-band model")
    X = dataset[band_708] / dataset[band_665]
    return (25.28 * (X) * 2) + 14.85 * (X) - 15.18


def ChlA_MODIS2B(
    dataset: xr.Dataset, band_748: str, band_667: str
) -> xr.DataArray:
    """
    MODIS
    Chla = 190.34 x MODIS_2BM – 32.45
    where: MODIS_2BM = Band15(743–753 nm) / Band13(662–672 nm)
    closest S2 bands are : B6(740) and B4(664)

    MODIS2B = 190.34 * (ds.msi06/ ds.msi04r) - 32.45
    """
    # matching MSI bands are 6 and 4
    log.info("ChlA_MODIS two-band model")
    X = dataset[band_748] / dataset[band_667]
    return (190.34 * X) - 32.45


# ---- Normalised Difference Suspeneded Sediment Index NDSSI ---
#    These are essentially
#        red-green / red_+green, or,
#        blue-NIR / blue + NIR
def NDSSI_RG(
    dataset: xr.Dataset, red_band: str, green_band: str
) -> xr.DataArray:
    log.info("NDSSI_RG")
    return (dataset[red_band] - dataset[green_band]) / (
        dataset[red_band] + dataset[green_band]
    )


def NDSSI_BNIR(
    dataset: xr.Dataset, blue_band: str, NIR_band: str
) -> xr.DataArray:
    log.info("NDSSI_BNIR")
    return (dataset[blue_band] - dataset[NIR_band]) / (
        dataset[blue_band] + dataset[NIR_band]
    )


# ---- Turbidity index of Yu, X. et al.
#    An empirical algorithm to seamlessly retrieve the concentration
#    of suspended particulate matter from water color across ocean to
#    turbid river mouths. Remote Sens. Environ. 235, 111491 (2019).
#    Used in screening turbid waters for mapping floating algal blooms
#    Initially developed with TM
#    -TI = ((Red − green) − (NIR − Rgreen)) ^ 0.5
def TI_yu(
    dataset: xr.Dataset,
    NIR: str,
    Red: str,
    Green: str,
    scalefactor: float = 0.01,
) -> xr.DataArray:
    log.info("TI_yu")
    # TODO check scalefactor * ((dataset[Red] - dataset[Green])
    # - ((dataset[NIR] - dataset[Green]) * 0.5)) correction!
    return scalefactor * (
        ((dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green]))
        ** 0.5
    )


# ---- Lymburner Total Suspended Matter (TSM)
# Paper: [Lymburner et al. 2016] (https://www.sciencedirect.com/science/article/abs/pii/S0034425716301560)
# Units of mg/L concentration. Variants for ETM and OLT, slight
# difference in parameters.
# These models, developed by leo lymburner and arnold dekker, are
# simple, stable, and produce credible results over a range of
# observations
def TSM_LYM_ETM(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    log.info("TSM_LYM_ETM")
    return (
        3983
        * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2)
        ** 1.6246
    )


def TSM_LYM_OLI(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    log.info("TSM_LYM_OLI")
    return (
        3957
        * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2)
        ** 1.6436
    )


# Qui Function to calculate Suspended Particulate Model value
# Paper: Zhongfeng Qiu et.al. 2013 - except it's not.
# This model seems to discriminate well although the scaling is
# questionable and it goes below zero due to the final subtraction.
# (The final subtraction seems immaterial in the context of our work
# (overly precise) and I skip it.)
def SPM_QIU(
    dataset: xr.Dataset, green_band: str, red_band: str
) -> xr.DataArray:
    log.info("SPM_QIU")
    return (
        10.0
        ** (
            2.26 * ((dataset[red_band] / dataset[green_band]) ** 3)
            - 5.42 * ((dataset[red_band] / dataset[green_band]) ** 2)
            + 5.58 * (dataset[red_band] / dataset[green_band])
            - 0.72
        )
        # - 1.43
    )


# ---- Quang Total Suspended Solids (TSS)
# Paper: Quang et al. 2017
# Units of mg/L concentration
def TSS_QUANG8(dataset: xr.Dataset, red_band: str) -> xr.DataArray:
    # ---- Function to calculate quang8 value ----
    log.info("TSS_QUANG8")
    return 380.32 * (dataset.red_band) * 0.0001 - 1.7826


# ---- Model of Zhang et al 2023: ----
#      This model seems to be funamentally unstable, using band ratios as an exponent
#      is asking for extreme values and numerical overflows. It runs to inf all over the place.
#      Green = B3 520-600; Blue = B2 450-515 Red = B4 630-680
#  The function is not scale-less.
#  The factor of 0.0001 is not part of the forumula but  scales back from 10000 range which is clearly
#  ridiculous (exp(10000) etc. is not a good number). This therefore avoids overflow.
# This model can only be used together with other models and indices; it may handle some situations well...
def TSS_Zhang(
    dataset: xr.Dataset,
    blue_band: str,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    log.info("TSS_Zhang")
    abovezero = 0.00001  # avoids div by zero if blue is zero
    GplusR = dataset[green_band] + dataset[red_band]
    RdivB = dataset[red_band] / (dataset[blue_band] + abovezero)
    X = (GplusR * RdivB) * scale_factor
    # return(10**(14.44*X))*1.20
    # return  np.exp(14.44* X)*1.20
    return (
        14.44 * X
    )  # the distribution of results is exponential; this measure will be more stable without raising to the power.


# =============================================================================
# Water Quality Algorithm Applications to Instrument Data
# =============================================================================

# For each algorithm, provide a mapping of the instruments it should be
# applied to, along with details on how the algorithm function should be
# applied to the instrument's data. This includes the specific band
# combinations to use for each instrument and the name of the water quality
# variable produced after applying the algorithm.

ndci_nir_r = {
    "msi_agm": {
        "54": {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_msi54_agm",
            "args": {"NIR_band": "msi05_agmr", "red_band": "msi04_agmr"},
        },
        "64": {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_msi64_agm",
            "args": {"NIR_band": "msi06_agmr", "red_band": "msi04_agmr"},
        },
        "74": {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_msi74_agm",
            "args": {"NIR_band": "msi07_agmr", "red_band": "msi04_agmr"},
        },
    },
    "tm_agm": {
        "func": NDCI_NIR_R,
        "wq_varname": "ndci_tm43_agm",
        "args": {"NIR_band": "tm04_agm", "red_band": "tm03_agmr"},
    },
    "oli_agm": {
        "func": NDCI_NIR_R,
        "wq_varname": "ndci_oli54_agm",
        "args": {"NIR_band": "oli05_agm", "red_band": "oli04_agmr"},
    },
}

chla_meris2b = {
    "msi_agm": {
        "func": ChlA_MERIS2B,
        "wq_varname": "chla_meris2b_msi_agm",
        "args": {"band_708": "msi05_agmr", "band_665": "msi04_agmr"},
    },
    "msi": {
        "func": ChlA_MERIS2B,
        "wq_varname": "chla_meris2b_msi",
        "args": {"band_708": "msi05", "band_665": "msi04"},
    },
}

chla_modis2b = {
    "msi_agm": {
        "func": ChlA_MODIS2B,
        "wq_varname": "chla_modis2b_msi_agm",
        "args": {"band_748": "msi06_agmr", "band_667": "msi04_agmr"},
    },
    "msi": {
        "func": ChlA_MODIS2B,
        "wq_varname": "chla_modis2b_msi",
        "args": {"band_748": "msi06", "band_667": "msi04"},
    },
    "tm_agm": {
        "func": ChlA_MODIS2B,
        "wq_varname": "chla_modis2b_tm_agm",
        "args": {"band_748": "tm04_agmr", "band_667": "tm03_agmr"},
    },
}

ndssi_rg = {
    "msi_agm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_msi_agm",
        "args": {"red_band": "msi04_agmr", "green_band": "msi03_agmr"},
    },
    "msi": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_msi",
        "args": {"red_band": "msi04r", "green_band": "msi03_agmr"},
    },
    "oli_agm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_oli_agm",
        "args": {"red_band": "oli04_agmr", "green_band": "oli03_agmr"},
    },
    "oli": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_oli",
        "args": {"red_band": "oli04r", "green_band": "oli03r"},
    },
    "tm_agm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_tm_agm",
        "args": {"red_band": "tm03_agmr", "green_band": "tm02_agmr"},
    },
    "tm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_tm",
        "args": {"red_band": "tm03r", "green_band": "tmi02r"},
    },
}

ndssi_bnir = {  # "msi_agm" : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_msi_agm'     ,'args' : { "NIR_band":'msi08_agmr', "blue_band":'msi02_agmr'}},
    "msi": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_msi",
        "args": {"NIR_band": "msi08", "blue_band": "msi02_agmr"},
    },
    "oli_agm": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_oli_agm",
        "args": {"NIR_band": "oli06_agm", "blue_band": "oli02_agmr"},
    },
    "oli": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_oli",
        "args": {"NIR_band": "oli06", "blue_band": "oli02r"},
    },
    # "tm_agm"  : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_tm_agm'      ,'args' : { "NIR_band":'tm04_agm'  , "blue_band":'tm01_agmr' }},
    "tm": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_tm",
        "args": {"NIR_band": "tm04", "blue_band": "tm01r"},
    },
}


ti_yu = {  # "msi_agm" : {'func': TI_yu,        "wq_varname" : 'ti_yu_msi_agm'        ,'args' : {"NIR" : 'msi08_agmr', "Red":'msi04_agmr', "Green":'msi03_agmr'}},
    "msi": {
        "func": TI_yu,
        "wq_varname": "ti_yu_msi",
        "args": {"NIR": "msi08", "Red": "msi04r", "Green": "msi03_agmr"},
    },
    "oli_agm": {
        "func": TI_yu,
        "wq_varname": "ti_yu_oli_agm",
        "args": {
            "NIR": "oli06_agm",
            "Red": "oli04_agmr",
            "Green": "oli03_agmr",
        },
    },
    "oli": {
        "func": TI_yu,
        "wq_varname": "ti_yu_oli",
        "args": {"NIR": "oli06", "Red": "oli04r", "Green": "oli03r"},
    },
    "tm_agm": {
        "func": TI_yu,
        "wq_varname": "ti_yu_tm_agm",
        "args": {"NIR": "tm04_agm", "Red": "tm03_agmr", "Green": "tm02_agmr"},
    },
    "tm": {
        "func": TI_yu,
        "wq_varname": "ti_yu_tm",
        "args": {"NIR": "tm04", "Red": "tm03r", "Green": "tmi02r"},
    },
}

tsm_lym = {
    "oli_agm": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_oli_agm",
        "args": {"red_band": "oli04_agmr", "green_band": "oli03_agmr"},
    },
    "oli": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_oli",
        "args": {"red_band": "oli04r", "green_band": "oli03r"},
    },
    "msi_agm": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_msi_agm",
        "args": {"red_band": "msi04_agmr", "green_band": "msi03_agmr"},
    },
    "msi": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_msi",
        "args": {"red_band": "msi04r", "green_band": "msi03r"},
    },
    "tm_agm": {
        "func": TSM_LYM_ETM,
        "wq_varname": "tsm_lym_tm_agm",
        "args": {"red_band": "tm03_agmr", "green_band": "tm02_agmr"},
    },
    "tm": {
        "func": TSM_LYM_ETM,
        "wq_varname": "tsm_lym_tm",
        "args": {"red_band": "tm03r", "green_band": "tm02r"},
    },
}

spm_qiu = {
    "oli_agm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_oli_agm",
        "args": {"red_band": "oli04_agmr", "green_band": "oli03_agmr"},
    },
    "oli": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_oli",
        "args": {"red_band": "oli04r", "green_band": "oli03r"},
    },
    "tm_agm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_tm_agm",
        "args": {"red_band": "tm03_agmr", "green_band": "tm02_agmr"},
    },
    "tm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_tm",
        "args": {"red_band": "tm03r", "green_band": "tm02r"},
    },
    "msi_agm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_msi_agm",
        "args": {"red_band": "msi04_agmr", "green_band": "msi03_agmr"},
    },
    "msi": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_msi",
        "args": {"red_band": "msi04r", "green_band": "msi03r"},
    },
}

tss_zhang = {
    "msi_agm": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_msi_agm",
        "args": {
            "blue_band": "msi02_agmr",
            "red_band": "msi04_agmr",
            "green_band": "msi03_agmr",
        },
    },
    "msi": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_msi",
        "args": {
            "blue_band": "msi02r",
            "red_band": "msi04r",
            "green_band": "msi03_agmr",
        },
    },
    "oli_agm": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_oli_agm",
        "args": {
            "blue_band": "oli02_agmr",
            "red_band": "oli04_agmr",
            "green_band": "oli03_agmr",
        },
    },
    "oli": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_oli",
        "args": {
            "blue_band": "oli02r",
            "red_band": "oli04r",
            "green_band": "oli03r",
        },
    },
}

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

# =============================================================================
# Water Quality Algorithms Groups
# =============================================================================

# Algorithms to produce extimates of
# Chlorophyll-A (ChlA) (generalised as ‘trophic state’)
ALGORITHMS_CHLA = {
    "ndci_nir_r": ndci_nir_r,
    "chla_meris2b": chla_meris2b,
    "chla_modis2b": chla_modis2b,
}

# Algorithms to produce extimates of
# Total Suspended Solids (TSS) (generalised as  ‘turbidity’)
ALGORITHMS_TSS = {
    "ndssi_rg": ndssi_rg,
    "ndssi_bnir": ndssi_bnir,
    "ti_yu": ti_yu,
    "tsm_lym": tsm_lym,
    "tss_zhang": tss_zhang,
    "spm_qiu": spm_qiu,
}


def run_wq_algorithms(
    ds: xr.Dataset,
    instruments_list: dict[str, dict[str, dict[str, str | tuple]]],
    algorithms_group: dict[str, dict[str, dict[str, Any]]],
) -> tuple[xr.Dataset, list[str]]:
    """
    Run a group of water quality algorithms on water areas in the input
    dataset `ds`. The dataset should already be masked so that non-water
    pixels are set to ``np.nan``.
    Each algorithm is applied to the specified instruments and band
    combinations in the provided dictionary, ensuring that all required
    instruments are present in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing data for the instruments listed in
        `instruments_list`. Non-water pixels should already be masked as
        ``np.nan``.
    instruments_list :  dict[str, dict[str, dict[str, str | tuple]]],
        Master list of instruments used to derive the input dataset.
    algorithms_group : dict[str, dict[str, dict[str, Any]]]
        A group of water quality algorithms. Each item in this mapping
        describes:
        1) the instruments an algorithm is to be applied to and
        2) for each instrument, the band combinations to use for an
            algorithm
        3) and the name of the water quality variable produced after
            applying the algorithm to an instrument.
    Returns
    -------
    tuple[xr.Dataset, list[str]]:
        A tuple containing:
        - The input dataset with added water quality variables.
        - A list of names for the generated water quality variables.
    """
    algorithm_names = list(algorithms_group.keys())
    log.info(
        f"Running water quality algorithms for: {', '.join(algorithm_names)}"
    )

    # Apply the water quality algorithms
    wq_varlist = []
    for algorithm_name in algorithm_names:
        log.info(f"Running water quality algorithm {algorithm_name}")

        algorithm_applications = algorithms_group[algorithm_name]
        for instrument_name in list(algorithm_applications.keys()):
            # check if data was loaded for the instrument.
            if instrument_name not in list(instruments_list.keys()):
                continue
            else:
                if (
                    algorithm_name == "ndci_nir_r"
                    and instrument_name == "msi_agm"
                ):
                    # There are multiple ndci_nir_r algorithm applications
                    # for the instrument `msi_agm`.
                    instrument_algorithm_apps = list(
                        algorithm_applications[instrument_name].values()
                    )
                else:
                    instrument_algorithm_apps = [
                        algorithm_applications[instrument_name]
                    ]
                for inst_alg_app in instrument_algorithm_apps:
                    alg_function = inst_alg_app["func"]
                    alg_function_args = inst_alg_app["args"]
                    wq_varname = inst_alg_app["wq_varname"]
                    ds[wq_varname] = alg_function(ds, **alg_function_args)

                    # Add the nodata, scale and offset metadata
                    scale_and_offset = NORMALISATION_PARAMETERS.get(
                        wq_varname, None
                    )
                    if scale_and_offset is None:
                        scale = 1
                        offset = 0
                    else:
                        scale = scale_and_offset["scale"]
                        offset = scale_and_offset["offset"]
                    ds[wq_varname].attrs = dict(
                        nodata=np.nan, scales=scale, offsets=offset
                    )

                    wq_varlist.append(wq_varname)

    return ds, wq_varlist


def WQ_vars(
    ds: xr.Dataset,
    instruments_list: dict[str, dict[str, dict[str, str | tuple]]],
    stack_wq_vars: bool,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Run Chlorophyll-A (ChlA) and Total Suspended Solids (TSS) algorithms
    on water areas in the input dataset `ds`. The dataset should already
    be masked so that non-water pixels are set to ``np.nan``.
    Computed water quality variables are added to the dataset, and their
    names are collected into a list.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing data for the instruments listed in
        `instruments_list`. Non-water pixels must already be masked as
        ``np.nan``.
    instruments_list :  dict[str, dict[str, dict[str, str | tuple]]],
        Master list of instruments used to derive the input dataset.
    stack_wq_vars : bool
        If False, outputs are retained as separate variables in a 3D dataset.
        If True, water quality variables are stacked into a multi-dimensional
        dataset.

    Returns
    -------
    tuple[xr.Dataset, pd.DataFrame]:
        A tuple containing:
        - The input dataset with added water quality variables.
        - A table containing the generated TSS and ChlA water quality
            variables.

    """

    ds, tss_wq_vars = run_wq_algorithms(
        ds, algorithms_group=ALGORITHMS_TSS, instruments_list=instruments_list
    )
    ds, chla_wq_vars = run_wq_algorithms(
        ds, algorithms_group=ALGORITHMS_CHLA, instruments_list=instruments_list
    )

    # Save a table containing the water quality variables
    chla_df = pd.DataFrame(data=dict(chla_measures=chla_wq_vars))
    tss_df = pd.DataFrame(data=dict(tss_measures=tss_wq_vars))
    all_wq_vars_df = pd.concat([tss_df, chla_df], axis=1)

    if stack_wq_vars:
        # Keep the  dimensions of the 3D dataset
        original_ds_dims = list(ds.dims)

        # Stack the TSS water quality variables.
        ds["tss"] = ds[tss_wq_vars].to_stacked_array(
            new_dim="tss_measures",
            sample_dims=original_ds_dims,
            variable_dim="tss_wq_vars",
            name="tss",
        )
        # Keep TSS water quality variables attributes
        tss_wq_vars_attrs = {
            var: ds[var].attrs for var in tss_wq_vars if ds[var].attrs
        }
        ds["tss"].attrs = {"tss_wq_vars_attrs": tss_wq_vars_attrs}

        # Stack the Chla water quality variables.
        ds["chla"] = ds[chla_wq_vars].to_stacked_array(
            new_dim="chla_measures",
            sample_dims=original_ds_dims,
            variable_dim="chla_wq_vars",
            name="chla",
        )
        # Keep Chla water quality variables attributes
        chla_wq_vars_attrs = {
            var: ds[var].attrs for var in chla_wq_vars if ds[var].attrs
        }
        ds["chla"].attrs = {"chla_wq_vars_attrs": chla_wq_vars_attrs}

        all_wq_vars = tss_wq_vars.extend(chla_wq_vars)
        ds = ds.drop_vars(all_wq_vars)

    return ds, all_wq_vars_df
