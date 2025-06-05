from datetime import date, datetime

import pytest

from water_quality.instruments import (
    check_instrument_dates,
    validate_date_str,
    validate_end_date,
    validate_start_date,
)


def test_invalid_datestring_year_as_int():
    date_str = 2015
    with pytest.raises(TypeError):
        validate_date_str(date_str)


def test_valid_datestring_year_as_str():
    date_str = "2015"
    assert validate_date_str(date_str)[0] == date(year=2015, month=1, day=1)


def test_valid_datestring_year_and_month():
    date_str = "2015-09"
    assert validate_date_str(date_str)[0] == date(year=2015, month=9, day=1)


def test_valid_datestring_year_month_and_day():
    date_str = "2015-09-15"
    assert validate_date_str(date_str)[0] == date(year=2015, month=9, day=15)


def test_invalid_datestring_format():
    date_str = "2015/09/01"
    with pytest.raises(ValueError):
        validate_date_str(date_str)


def test_valid_start_date_year():
    start_date = "2019"
    assert validate_start_date(start_date) == date(year=2019, month=1, day=1)


def test_valid_start_date_year_and_month():
    start_date = "2024-02"
    assert validate_start_date(start_date) == date(year=2024, month=2, day=1)


def test_valid_start_date_year_month_and_day():
    start_date = "2004-12-01"
    assert validate_start_date(start_date) == date(year=2004, month=12, day=1)


def test_valid_end_date_year():
    end_date = "2019"
    assert validate_end_date(end_date) == date(year=2019, month=12, day=31)


def test_valid_end_date_year_and_month():
    end_date = "2024-02"
    assert validate_end_date(end_date) == date(year=2024, month=2, day=29)


def test_valid_end_date_year_month_and_day():
    end_date = "2004-12-01"
    assert validate_start_date(end_date) == date(year=2004, month=12, day=1)


def test_invalid_date_range_oli_agm():
    instruments_to_use = {"oli_agm": {"use": True}}
    start_date = "2012"
    end_date = "2024"

    expected_result = {"oli_agm": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_oli():
    instruments_to_use = {"oli": {"use": True}}
    start_date = "2012"
    end_date = "2024"

    expected_result = {"oli": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_msi_agm():
    instruments_to_use = {"msi_agm": {"use": True}}
    start_date = "2016"
    end_date = "2025"

    expected_result = {"msi_agm": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_msi():
    instruments_to_use = {"msi": {"use": True}}
    start_date = "2016"
    end_date = "2025"

    expected_result = {"msi": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_wofs_ann():
    instruments_to_use = {"wofs_ann": {"use": True}}
    start_date = "1989"
    end_date = "2024"

    expected_result = {"wofs_ann": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_wofs_all():
    instruments_to_use = {"wofs_all": {"use": True}}
    start_date = "1989"
    end_date = "2024"

    expected_result = {"wofs_all": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_tm_agm():
    instruments_to_use = {"tm_agm": {"use": True}}
    start_date = "1989"
    end_date = "2013"

    expected_result = {"tm_agm": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_tm():
    instruments_to_use = {"tm": {"use": True}}
    start_date = "1989"
    end_date = "2013"

    expected_result = {"tm": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def test_invalid_date_range_tirs():
    instruments_to_use = {"tirs": {"use": True}}
    start_date = "1989"
    end_date = "2013"

    expected_result = {"tirs": {"use": False}}

    assert expected_result == check_instrument_dates(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )
