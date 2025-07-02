"""
This module provides functions to validate input dates.
"""

import calendar
from datetime import date, datetime


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
        Date object if the date string matches the specified date format.
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
        f"{date_str} does not match any expected format {' or '.join(expected_date_formats)}"
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
