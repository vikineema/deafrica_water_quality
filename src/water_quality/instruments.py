import calendar
import logging
from datetime import date, datetime

log = logging.getLogger(__name__)

INSTRUMENTS_DATES = {
    "msi": [2017, 2024],
    "msi_agm": [2017, 2024],
    "oli": [2013, 2024],
    "oli_agm": [2013, 2024],
    "tirs": [2000, 2024],
    "tm": [1990, 2012],
    "tm_agm": [1990, 2012],
    "wofs_ann": [1990, 2024],
    "wofs_all": [1990, 2024],
}

# Here is where to turn a particular band on or off, using the 'parameters' entry
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
        "st": {
            "varname": ("tirs_st"),
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
        "emis": {
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
        "st_qa": {
            "varname": ("tirs_qa_st"),
            "parameters": (
                True,
                "ST_QA 	int16 	Kelvin 	-9999.0 	[st_qa, surface_temperature_quality]",
            ),
        },
    },
}


def validate_date_str(date_str: str) -> tuple[date, str]:
    """
    Parse a date string into a datetime.date object and find the
    format the date matches.

    Parameters
    ----------
    date_str : str
        Date to parse in string format.

    Returns
    -------
    tuple[date, str]
        Date object and date format.

    """
    expected_date_patterns = ["%Y-%m-%d", "%Y-%m", "%Y"]

    if not isinstance(date_str, str):
        raise TypeError(f"{date_str} is type ({type(date_str)}) not a string")
    else:
        for date_pattern in expected_date_patterns:
            try:
                valid_date = datetime.strptime(date_str, date_pattern).date()
            except Exception:
                continue
            else:
                return valid_date, date_pattern
        raise ValueError(
            f"{date_str} does not match any expected format {' or '.join(expected_date_patterns)}"
        )


def validate_start_date(date_str: str) -> date:
    """
    Parse a start date into the correct format.

    Parameters
    ----------
    date_str : str
        Start date as string

    Returns
    -------
    date
        Start date as datetime.date object.
    """
    valid_start_date, date_pattern = validate_date_str(date_str)
    if date_pattern == "%Y-%m":
        year = valid_start_date.year
        month = valid_start_date.month
        day = 1
        return datetime(year, month, day).date()
    elif date_pattern == "%Y":
        year = valid_start_date.year
        month = 1
        day = 1
        return datetime(year, month, day).date()
    else:
        return valid_start_date


def validate_end_date(date_str: str) -> date:
    """
    Parse a end date into the correct format.

    Parameters
    ----------
    date_str : str
        End date as string.

    Returns
    -------
    date
        End date as datetime.date object.
    """
    valid_end_date, date_pattern = validate_date_str(date_str)
    if date_pattern == "%Y-%m":
        year = valid_end_date.year
        month = valid_end_date.month
        day = calendar.monthrange(year, month)[1]
        return datetime(year, month, day).date()
    elif date_pattern == "%Y":
        year = valid_end_date.year
        month = 12
        day = calendar.monthrange(year, month)[1]
        return datetime(year, month, day).date()
    else:
        return valid_end_date


def check_instrument_dates(
    instruments_to_use: dict[str, dict[str, bool]],
    start_date: str,
    end_date: str,
):
    """
    Cross check the years data is available for each instrument against
    the years for which the analysis is going to be run.

    Parameters
    ----------
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the selected instruments to use for the analysis and
        a usage parameter showing whether it will be used in the analysis.
    start_date : str
        Start date of the analysis period.
    end_date : str
        End date of the analysis period.

    Returns
    -------
    dict[str, dict[str, bool]]
        Updated `instruments_to_use` where instruments where data is not available in
        the analysis period have their usage parameter set to False.
    """
    start_date = validate_start_date(start_date)
    end_date = validate_end_date(end_date)

    valid_instruments_to_use: dict[str, dict[str, bool]] = {}
    for instrument_name, usage in instruments_to_use.items():
        if usage["use"] is True:
            valid_date_range = INSTRUMENTS_DATES.get(instrument_name, None)
            if valid_date_range is None:
                valid_instruments_to_use[instrument_name] = {"use": False}
                log.error(
                    f"Valid date range for instrument {instrument_name} has not been set"
                )
            else:
                valid_date_range = [
                    validate_start_date(str(min(valid_date_range))),
                    validate_end_date(str(max(valid_date_range))),
                ]
                valid_date_range.sort()

                if (
                    start_date >= valid_date_range[0]
                    and end_date <= valid_date_range[-1]
                ):
                    valid_instruments_to_use[instrument_name] = {"use": True}
                else:
                    valid_instruments_to_use[instrument_name] = {"use": False}
                    log.error(
                        f"Instrument {instrument_name} has the date ranges "
                        f"{valid_date_range[0]} to {valid_date_range[-1]} which is outside"
                        f" the supplied date range of {start_date} to {end_date}."
                    )
        else:
            valid_instruments_to_use[instrument_name] = usage
    return valid_instruments_to_use


def get_instruments_list(
    instruments_to_use: dict[str, dict[str, bool]],
) -> dict[str, dict[str, dict[str, str | tuple]]]:
    """
    Primary list of instruments, measurements, and interoperable variable names

    Parameters
    ----------
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the selected instruments to use for the analysis and
        a usage parameter showing whether it will be used in the analysis.

    Returns
    -------
    dict
        A dictionary containing :
        - 'instruments', the 'master list' of instruments being used in the analysis.
            This is a subsetof the full list available.
        - 'measurements', a list of measurements to be accessed from the data cube collections - for each instrument
        -'rename_dict', a dictionary for re-naming measurements to have unique dataset variable names, when the time comes
    """
    instruments_list = {}
    for instrument_name, instrument_usage in instruments_to_use.items():
        if instrument_usage["use"] is True:
            instrument_info = INSTRUMENTS_MEASUREMENTS[instrument_name]
            meaurements_to_keep = {}
            for measurement_name, measurement_info in instrument_info.items():
                measurement_usage = measurement_info["parameters"][0]
                if measurement_usage is True:
                    meaurements_to_keep[measurement_name] = measurement_info
                else:
                    pass
            instruments_list[instrument_name] = meaurements_to_keep
        else:
            pass
    return instruments_list
