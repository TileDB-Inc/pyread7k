from test import bf_filepath

import pytest
from dotenv import find_dotenv, load_dotenv

from pyread7k import Ping, PingDataset, PingType

load_dotenv(find_dotenv())


@pytest.fixture
def pingdataset() -> PingDataset:
    return PingDataset(bf_filepath, include=PingType.BEAMFORMED)


@pytest.fixture
def ping(pingdataset) -> Ping:
    return pingdataset[10]


def test_read_s7kfile(pingdataset: PingDataset):
    assert isinstance(pingdataset, PingDataset)
