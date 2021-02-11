from pyread7k import PingDataset, PingType, Ping
import pytest
import pandas as pd
import numpy as np
import os
from dotenv import find_dotenv, load_dotenv
from test import bf_filepath

load_dotenv(find_dotenv())

@pytest.fixture
def pingdataset() -> PingDataset:
    return PingDataset(bf_filepath, include=PingType.BEAMFORMED)

@pytest.fixture
def ping(pingdataset) -> Ping:
    return pingdataset[10]


def test_read_s7kfile(pingdataset: PingDataset):
    assert isinstance(pingdataset, PingDataset)

def test_get_data(ping: Ping):
    assert ping.data_is_loaded() == False
    ping.load_data()
    assert ping.data_is_loaded() == True
    assert isinstance(ping.amp, np.ndarray)
    assert isinstance(ping.phs, np.ndarray)

def test_range_exclusion(ping: Ping):
    # Min exclusion
    ping.load_data()
    original_shape = ping.shape
    ping.exclude_ranges(min_range_meter=50)
    assert ping.range_samples.min() >= 50
    assert ping.shape != original_shape

    # Max exclusion
    ping.reset()
    ping.load_data()
    ping.exclude_ranges(max_range_meter=50)
    assert ping.range_samples.max() <= 50
    assert ping.shape != original_shape

    # ValueError
    ping.reset()
    ping.load_data()
    with pytest.raises(ValueError):
        ping.exclude_ranges(min_range_meter=100, max_range_meter=50)
