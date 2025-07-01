import calendar
from datetime import date, datetime


def check_date_format(date_str: str, date_format: str) -> tuple[date, str]:
    """
    Check if a date string matches a provided date format.

    Parameters
    ----------
    date_str : str
        Date string to check.
    date_format : str
        Date format to match date sting to.

    Returns
    -------
    tuple[date, str]
        Date in the specified format if it matches and the date format strings.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"{date_str} is type ({type(date_str)}) not a string")

    try:
        valid_date = datetime.strptime(date_str, date_format).date()
    except ValueError:
        raise
    else:
        return valid_date, date_format


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
    expected_date_formats = ["%Y-%m-%d", "%Y-%m", "%Y"]

    if not isinstance(date_str, str):
        raise TypeError(f"{date_str} is type ({type(date_str)}) not a string")

    for date_str_format in expected_date_formats:
        try:
            valid_date, date_format = check_date_format(date_str, date_str_format)
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
    valid_start_date, date_format = validate_date_str(date_str)
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
    valid_end_date, date_format = validate_date_str(date_str)
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
