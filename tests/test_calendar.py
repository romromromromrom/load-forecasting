import datetime as dt

import pytest

from rte_forecast.calendar import (
    NORMAL,
    SUMMER,
    YEAR_END,
    YEAR_START,
    build_daily_calendar,
    french_holidays,
    is_bridge_day,
    summer_trough_week,
)


def test_eleven_national_holidays_per_year():
    hol = french_holidays([2023])
    assert len(hol) == 11
    assert dt.date(2023, 12, 25) in hol and dt.date(2023, 8, 15) in hol
    assert dt.date(2023, 12, 26) not in hol       # pas de Saint-Étienne (Alsace-Moselle)


def test_bridge_thursday_holiday_gives_friday_bridge():
    hol = set(french_holidays([2024]))            # Ascension jeudi 9 mai 2024
    assert is_bridge_day(dt.date(2024, 5, 10), hol)
    assert not is_bridge_day(dt.date(2024, 5, 9), hol)    # le férié n'est pas un pont


def test_bridge_tuesday_holiday_gives_monday_bridge():
    hol = set(french_holidays([2022]))            # Toussaint mardi 1er novembre 2022
    assert is_bridge_day(dt.date(2022, 10, 31), hol)


def test_wednesday_holiday_creates_no_bridge():
    hol = set(french_holidays([2024]))            # Fête de la Victoire, mercredi 8 mai 2024
    assert dt.date(2024, 5, 8) in hol
    assert not is_bridge_day(dt.date(2024, 5, 7), hol)
    assert not is_bridge_day(dt.date(2024, 5, 9), hol)


@pytest.mark.parametrize("year,monday", [
    (2020, dt.date(2020, 8, 10)),   # 15/08 samedi : week-end 15-16
    (2021, dt.date(2021, 8, 9)),    # 15/08 dimanche : week-end 14-15
    (2022, dt.date(2022, 8, 8)),    # 15/08 lundi : week-end 13-14 (précédent)
    (2023, dt.date(2023, 8, 7)),    # 15/08 mardi : week-end 12-13
    (2024, dt.date(2024, 8, 12)),   # 15/08 jeudi : week-end 17-18
])
def test_summer_trough_week_is_the_week_whose_weekend_is_closest_to_15_august(year, monday):
    assert summer_trough_week(year) == monday
    assert monday.weekday() == 0


def test_summer_trough_tie_is_deterministic():
    # 15/08/2018 est un mercredi : égalité entre deux week-ends -> le postérieur, à chaque appel
    assert summer_trough_week(2018) == summer_trough_week(2018) == dt.date(2018, 8, 13)


def test_iso_weeks_51_52_are_year_end_and_1_2_year_start():
    cal = build_daily_calendar("2023-11-01", "2024-02-15")
    d = cal.loc["2023-12-18":"2023-12-31"]
    assert set(d["special_period_type"]) == {YEAR_END}       # S51 + S52
    d = cal.loc["2024-01-01":"2024-01-14"]
    assert set(d["special_period_type"]) == {YEAR_START}     # S1 + S2
    assert cal.loc["2024-01-15", "special_period_type"] == NORMAL
    assert cal.loc["2023-12-17", "special_period_type"] == NORMAL


def test_iso_week_53_belongs_to_year_end():
    cal = build_daily_calendar("2020-12-01", "2021-01-31")
    assert cal.loc["2021-01-01", "iso_week"] == 53
    assert cal.loc["2021-01-01", "special_period_type"] == YEAR_END


def test_special_period_precedence_and_flags():
    cal = build_daily_calendar("2024-01-01", "2024-12-31")
    assert cal.loc["2024-08-15", "special_period_type"] == SUMMER      # férié dans l'été
    assert cal.loc["2024-08-15", "is_public_holiday"]                  # le drapeau reste vrai
    assert cal.loc["2024-12-25", "special_period_type"] == YEAR_END


def test_required_calendar_columns_exist():
    cal = build_daily_calendar("2024-01-01", "2024-01-10")
    for c in ["year", "month", "iso_week", "weekday", "is_weekend", "is_public_holiday",
              "is_bridge_day", "is_ramp_up", "is_ramp_down", "is_august_special_period",
              "special_period_type"]:
        assert c in cal.columns
