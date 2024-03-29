# %%
# import itertools
# import math
import datetime
import os
from test import bf_filepath

import psutil
import pytest

from pyread7k import PingDataset, PingType

# %%


def get_current_memory():
    """ Prints current memory of active process """
    pid = os.getpid()
    own_process = psutil.Process(pid)
    return own_process.memory_info()[0] / (1024 ** 2)


@pytest.fixture
def dataset() -> PingDataset:
    return PingDataset(bf_filepath, include=PingType.BEAMFORMED)


@pytest.fixture
def ping(dataset):
    return dataset[10]


def test_sonar_settings_time(dataset):
    for p in dataset:
        assert isinstance(p.sonar_settings.frame.time, datetime.datetime)

def test_can_loop_multiple_times(dataset: PingDataset):
    loop1count = 0
    for p in dataset:
        p.position_set
        loop1count += 1

    loop2count = 0
    for p in dataset:
        p.position_set
        loop2count += 1

    assert loop1count == loop2count

def test_dataset_memory_use(dataset: PingDataset):
    init_memory = get_current_memory()
    for i, p in enumerate(dataset):
        # Make some ping calls to load data into the cached properties
        p.position_set
        p.roll_pitch_heave_set
        p.heading_set
        p.beamformed
        p.beam_geometry
        p.raw_iq

    # Check whether accessing all of the data of the pings resulted
    # in higher memeory use
    cur_memory = get_current_memory()
    assert cur_memory > init_memory

    # Minimize the ping data
    for p in dataset:
        p.minimize_memory()

    # Final check is whether the current memory is reduced
    # by minimizing the ping datastructure
    assert cur_memory > get_current_memory()
