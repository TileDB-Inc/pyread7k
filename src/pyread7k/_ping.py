"""
This module is an abstraction on top of the low-level 7k records, which allows
the user to work in terms of "pings" with associated data, instead of thinking
in the traditional 7k records.
"""
# %%
from enum import Enum
from functools import cached_property
import math

import numpy as np

from ._utils import read_file_catalog, read_file_header, read_records, get_record_offsets
from . import _datarecord
from ._datarecord import DataParts

class LazyMap(dict):
    """
    An advanced defaultdict, where the initializer may depend on the key.
    """
    def __init__(self, initializer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initializer = initializer

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.initializer(key)
        return super().__getitem__(key)
        


class Manager7k:
    """
    Internal class for Pings to share access to a file.
    """
    def __init__(self, fhandle, file_catalog):
        self.fhandle = fhandle
        self.file_catalog = file_catalog
        # self.cached_offsets = {}
        self._offsets_for_type = LazyMap(
            initializer=lambda key: get_record_offsets(
                key, self.file_catalog)
        )

    def get_next_record(self, record_type, offset_start, offset_end):
        """
        Get the offset and first record of type record_type which has a higher
        file offset than offset_start.
        """
        offset = self.get_next_offset(record_type, offset_start, offset_end)
        if offset is None:
            return None
        return self.read_record(record_type, offset)

    def get_next_offset(self, record_type, offset_start, offset_end):
        """
        Get the offset of type record_type which has a file offset between
        offset_start and offset_end.

        The data from a new ping always starts with a 7000 record, and so you
        can get the offset of a record for a specific ping by searching
        for an offset higher than the ping's 7000 record, but lower than the
        next ping's 7000 record.
        """
        record_offsets = self._offsets_for_type[record_type]
        next_index = np.searchsorted(record_offsets, offset_start, side="right")
        offset = record_offsets[next_index]
        if offset < offset_end:
            # No record exists in the interval
            return offset
        return None

    def read_record(self, record_type, offset):
        """
        Read a record from file using a known offset
        """
        self.fhandle.seek(offset)
        return _datarecord.record(record_type).read(self.fhandle)


class Ping:
    """
    A sound ping from a sonar, with associated data about settings and conditions.
    Properties of the ping are loaded efficiently on-demand.
    """

    minimizable_properties = ["beamformed", "tvg"]

    def __init__(self, settings_record, settings_offset : int,
                 next_offset : int, manager : Manager7k):
        self.records = {
            7300: settings_record,
        }
        self.manager = manager
        self.ping_number = settings_record.header["ping_number"]
        self._own_offset = settings_offset # This ping's start offset
        self._next_offset = next_offset # Next ping's start offset, meaning this ping has ended
        self.cached_offsets = {}

    def __str__(self):
        return "<Ping %i>" % self.records[7300].header["ping_number"]

    def minimize_memory(self):
        """
        Clears all memory-heavy properties.
        Retains offsets for easy reloading.
        """
        for key in self.minimizable_properties:
            if key in self.__dict__:
                del self.__dict__[key]

    @cached_property
    def beamformed(self) -> DataParts:
        """ Returns 7018 records """
        if not self.has_beamformed_data: # Forces loading offset into cache
            return None
        record = self.manager.read_record(7018, self.cached_offsets[7018])
        assert record.header["ping_number"] == self.ping_number
        return record

    @cached_property
    def has_beamformed_data(self):
        """ Checks if the ping has 7018 data without reading it. """
        offset = self.manager.get_next_offset(7018, self._own_offset, self._next_offset)
        self.cached_offsets[7018] = offset
        return offset is not None

    def _get_single_associated_record(self, record_type : int):
        """
        Read a record associated with the ping. The requested record must:
        - Be the only of its type for the ping
        """
        offset = self.manager.get_next_offset(record_type, self._own_offset,
            self._next_offset)
        if offset is None:
            return None
        record = self.manager.read_record(record_type, offset)
        assert record.header["ping_number"] == self.ping_number
        return record

    @cached_property
    def tvg(self):
        return self._get_single_associated_record(7010)

    @property
    def sonar_settings(self):
        return self.records[7300]


# %%
class PingType(Enum):
    BEAMFORMED = 1
    IQ = 2
    ANY = 3


class PingDataset:
    """
    Indexable PyTorch dataset returning Pings from a 7k file.

    Provides random access into pings in a file with minimal overhead.
    """
    def __init__(self, filename, include=PingType.ANY):
        """
        if include argument is not ANY, pings will be filtered.
        """
        self.filename = filename

        self.fhandle = open(filename, "rb", buffering=0)

        file_header = read_file_header(self.fhandle)
        file_catalog = read_file_catalog(self.fhandle, file_header)

        manager = Manager7k(self.fhandle, file_catalog)
        settings_records = read_records(7000, self.fhandle, file_catalog)
        settings_offsets = get_record_offsets(7000, file_catalog)
        pings = [Ping(rec, offset, next_off, manager) for rec, offset, next_off
                          in zip(settings_records, settings_offsets,
                                 settings_offsets[1:] + (math.inf,))]

        if include == PingType.BEAMFORMED:
            self.pings = [p for p in pings if p.has_beamformed_data]
        elif include == PingType.IQ:
            self.pings = pings # TODO: Make actual filter
        elif include == PingType.ANY:
            self.pings = pings


    def __len__(self) -> int:
        return len(self.pings)

    def __getitem__(self, index: int) -> Ping:
        return self.pings[index]

    # def __getstate__(self):
    #     """
    #     Remove unpicklable file handle from dict before pickling
    #     """
    #     state = self.__dict__.copy()
    #     del state["fhandle"]
    #     return state

    # def __setstate__(self, state):
    #     """
    #     Open new file handle after unpickling
    #     """
    #     self.__dict__.update(state)
    #     self.fhandle = open(self.filename, "rb", buffering=0)

    # def __del__(self):
    #     self.fhandle.close()
