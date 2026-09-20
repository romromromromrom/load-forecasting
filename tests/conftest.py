import pytest

from rte_forecast.calendar import build_daily_calendar, build_hourly_calendar
from rte_forecast.config import load_config
from rte_forecast.data.synthetic import make_synthetic_hourly


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth():
    return make_synthetic_hourly("2019-10-01", "2024-12-31", seed=1)


@pytest.fixture(scope="session")
def cal_h(synth):
    return build_hourly_calendar(synth.index)


@pytest.fixture(scope="session")
def cal_d():
    return build_daily_calendar("2019-08-01", "2025-02-15")
