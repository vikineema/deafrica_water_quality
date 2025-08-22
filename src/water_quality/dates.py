"""
This module provides functions to validate input dates.
"""

import calendar
from datetime import date, datetime, timedelta

import pandas as pd
import toolz
from odc.stats.model import DateTimeRange
from odc.stats.utils import mk_season_rules, season_binner


def check_date_str_format(date_str: str, date_format: str) -> date:
    """
    Check if a date string matches a provided date format.

    Parameters
    ----------
    date_str : str
        Date in string to check.
    date_format : str
        Date format to match date sting to.

    Returns
    -------
    date
        Date object if the date string matches the specified date
        format.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"{date_str} is type ({type(date_str)}) not a string")

    try:
        valid_date = datetime.strptime(date_str, date_format).date()
    except ValueError:
        raise
    else:
        return valid_date


def date_str_to_date(date_str: str) -> tuple[date, str]:
    """
    Find the format a date string matches and parse a
    date string into a datetime.date object.

    Parameters
    ----------
    date_str : str
        Date string to parse.

    Returns
    -------
    tuple[date, str]
        Date object and date format.

    """
    expected_date_formats = ["%Y-%m-%d", "%Y-%m", "%Y"]

    if not isinstance(date_str, str):
        raise TypeError(f"{date_str} is type ({type(date_str)}) not a string")

    for date_format in expected_date_formats:
        try:
            valid_date = check_date_str_format(date_str, date_format)
        except ValueError:
            continue
        else:
            return valid_date, date_format
    raise ValueError(
        f"{date_str} does not match any expected format "
        f"{' or '.join(expected_date_formats)}"
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
    valid_start_date, date_format = date_str_to_date(date_str)
    if date_format == "%Y-%m":
        year = valid_start_date.year
        month = valid_start_date.month
        day = 1
        return datetime(year, month, day).date()
    elif date_format == "%Y":
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
    valid_end_date, date_format = date_str_to_date(date_str)
    if date_format == "%Y-%m":
        year = valid_end_date.year
        month = valid_end_date.month
        day = calendar.monthrange(year, month)[1]
        return datetime(year, month, day).date()
    elif date_format == "%Y":
        year = valid_end_date.year
        month = 12
        day = calendar.monthrange(year, month)[1]
        return datetime(year, month, day).date()
    else:
        return valid_end_date


def middle_of_year(year: int) -> date:
    """
    Returns the date of the day in the middle of the year.

    This function calculates the midpoint of a given year by finding
    the total number of days and dividing by two. It accounts for leap
    years automatically.

    Parameters
    ----------
    year : int
        The year for which to find the middle day.

    Returns
    -------
    datetime.date
        A date object representing the middle day of the year.
    """
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    delta = (end - start).days // 2
    return start + timedelta(days=delta)


def year_to_dc_datetime(year: int) -> datetime:
    """
    Convert a year into a datetime object matching ODC behaviour
    for annual datasets e.g. DE Africa's gm_s2_annual product, where
    the date of the middle of the year is assigned to a dataset.

    Parameters
    ----------
    year : int
        Year to convert

    Returns
    -------
    datetime
        Date of the middle of the year in datetime format.
    """
    mid_year_date = middle_of_year(year)
    # Convert to datetime (default time is midnight)
    mid_year_datetime = datetime.combine(mid_year_date, datetime.min.time())
    if mid_year_datetime.day == 2:
        mid_year_datetime = mid_year_datetime.replace(
            hour=11, minute=59, second=59, microsecond=999999
        )
    elif mid_year_datetime.day == 1:
        mid_year_datetime = mid_year_datetime.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    else:
        raise ValueError(
            f"Unexpected mid year date {mid_year_datetime} for the year {year}"
        )
    return mid_year_datetime


def get_temporal_ids(
    temporal_range: DateTimeRange, frequency: str
) -> list[str]:
    """
    Generate temporal bin identifiers for a given date range and frequency.

    The function creates a list of temporal IDs that represent the start date
    and duration of each bin within the provided `temporal_range`. The binning
    logic depends on the `frequency` specified.

    Supported frequencies and their behavior:
    - "annual"       → One bin per calendar year (format: YYYY--P1Y)
    - "semiannual"   → Two bins per year, starting January 1 and July 1
                        (format: YYYY-MM-DD--P6M)
    - "monthly"      → One bin per calendar month (format: YYYY-MM-DD--P1M)
    - "weekly"       → Fixed 7-day intervals starting from `temporal_range.start`
                        (format: YYYY-MM-DD--P1W)
    - "fortnightly"  → Fixed 14-day intervals starting from `temporal_range.start`
                        (format: YYYY-MM-DD--P2W)

    Parameters
    ----------
    temporal_range : DateTimeRange
        Object with `.start` and `.end` datetime attributes representing the
        inclusive temporal range to bin.
    frequency : str
        Temporal binning frequency. Must be one of:
        {"annual", "semiannual", "monthly", "weekly", "fortnightly"}.

    Returns
    -------
    list[str]
        List of temporal ID strings, ordered by start date, each representing
        one temporal bin.

    """

    temporal_range_start = temporal_range.start
    temporal_range_end = temporal_range.end

    dates = pd.date_range(start=temporal_range_start, end=temporal_range_end)

    if frequency in ["annual"]:
        grouped = toolz.groupby(lambda dt: dt.year, dates)
        temporal_ids = [
            f"{year}--P1Y" for year in list(grouped.keys()) if year != ""
        ]

    elif frequency in ["semiannual", "monthly"]:
        if frequency == "semiannual":
            months = 6
            anchor = 1
        elif frequency == "monthly":
            months = 1
            anchor = 1

        binner = season_binner(mk_season_rules(months, anchor))
        grouped = toolz.groupby(lambda dt: binner(dt), dates)
        temporal_ids = [i for i in list(grouped.keys()) if i != ""]

    elif frequency in ["weekly", "fortnightly"]:
        if frequency == "weekly":
            freq = "7D"
            label_suffix = "--P1W"
        elif frequency == "fortnightly":
            freq = "14D"
            label_suffix = "--P2W"

        bins_start_dates = pd.date_range(
            start=temporal_range_start, end=temporal_range_end, freq=freq
        )
        bins = {k: [] for k in bins_start_dates}
        for idx, start_dt in enumerate(bins_start_dates):
            if idx < len(bins_start_dates) - 1:
                end_dt = bins_start_dates[idx + 1]
                mask = (dates >= start_dt) & (dates < end_dt)
            else:
                # Last bin: include all remaining dates
                mask = dates >= start_dt

            bins[start_dt] = list(dates[mask])
        temporal_ids = [
            f"{i.strftime('%Y-%m-%d')}{label_suffix}"
            for i in list(bins.keys())
        ]
    else:
        raise NotImplementedError(
            f"Temporal binning for frequency {frequency} not implementd."
        )
    return temporal_ids
