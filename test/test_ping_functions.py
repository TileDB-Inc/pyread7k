from test import bf_filepath

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

from pyread7k import Ping, PingDataset, PingType
from pyread7k.processing import Beamformed

load_dotenv(find_dotenv())


@pytest.fixture
def pingdataset() -> PingDataset:
    return PingDataset(bf_filepath, include=PingType.BEAMFORMED)


@pytest.fixture
def ping(pingdataset) -> Ping:
    return pingdataset[10]


def test_read_s7kfile(pingdataset: PingDataset):
    assert isinstance(pingdataset, PingDataset)


def test_range_exclusion(ping: Ping):
    # Min exclusion
    p = Beamformed(ping)
    original_shape = p.shape
    p.exclude_ranges(min_range_meter=50)
    assert p.ranges.min() >= 50
    assert p.shape != original_shape

    # Max exclusion
    p.reset()
    p.exclude_ranges(max_range_meter=50)
    assert p.ranges.max() <= 50
    assert p.shape != original_shape

    # ValueError
    p.reset()
    with pytest.raises(ValueError):
        p.exclude_ranges(min_range_meter=100, max_range_meter=50)
