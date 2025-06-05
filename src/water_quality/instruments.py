import calendar
import logging
from datetime import date, datetime

log = logging.getLogger(__name__)

INSTRUMENTS_DATES = {
    "msi": [2017, 2024],
    "msi_agm": [2017, 2024],
    "oli": [2013, 2024],
    "oli_agm": [2013, 2024],
    "tirs": [1990, 2024],
    "tm": [1990, 2012],
    "tm_agm": [1990, 2012],
    "wofs_ann": [1990, 2024],
    "wofs_all": [1990, 2024],
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
