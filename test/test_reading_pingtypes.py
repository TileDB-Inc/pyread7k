# %%
from test import bf_filepath, iq_filepath
import pyread7k

def test_read_beamformed():
    assert len(pyread7k.PingDataset(bf_filepath, include=pyread7k.PingType.BEAMFORMED)) > 0

def test_read_iq():
    assert len(pyread7k.PingDataset(iq_filepath, include=pyread7k.PingType.IQ)) > 0
