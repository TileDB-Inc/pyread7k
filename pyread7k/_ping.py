"""
This module is an abstraction on top of the low-level 7k records, which allows
the user to work in terms of "pings" with associated data, instead of thinking
in the traditional 7k records.

Expected order of records for a ping:
7000, 7503, 1750, 7002, 7004, 7017, 7006, 7027, 7007, 7008, 7010, 7011, 7012,
7013, 7018, 7019, 7028, 7029, 7037, 7038, 7039, 7041, 7042, 7048, 7049, 7057,
7058, 7068, 7070

"""
import bisect
import sys
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from itertools import chain
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import geopy
import numpy as np

from . import _datarecord, records
from ._utils import cached_property, window


class PingType(Enum):
    """ Kinds of pings based on what data they have available """

    BEAMFORMED = 1
    IQ = 2
    ANY = 3


class S7KReader(metaclass=ABCMeta):
    """Base abstract class of S7K readers"""

    @cached_property
    def file_header(self) -> records.FileHeader:
        """Return the file header record for this reader"""
        return cast(records.FileHeader, self._read_record(7200, 0))

    @cached_property
    def file_catalog(self) -> records.FileCatalog:
        """Return the file catalog record for this reader"""
        return cast(
            records.FileCatalog,
            self._read_record(7300, self.file_header.catalog_offset),
        )

    @cached_property
    def configuration(self) -> records.Configuration:
        """Return the configuration record for this reader"""
        offsets = self._get_offsets(7001)
        assert len(offsets) == 1
        return cast(records.Configuration, self._read_record(7001, offsets[0]))

    def iter_pings(self, include: PingType = PingType.ANY) -> Iterator["Ping"]:
        """Iterate over Pings. if include argument is not ANY, filter pings by type"""
        offsets_records = chain(self._iter_offset_records(7000), [None])
        pings = (
            Ping(
                cast(Tuple[int, records.SonarSettings], offset_record),
                cast(Optional[Tuple[int, records.SonarSettings]], next_offset_record),
                reader=self,
            )
            for offset_record, next_offset_record in window(offsets_records, 2)
        )
        if include == PingType.ANY:
            return pings
        if include == PingType.BEAMFORMED:
            return (p for p in pings if p.has_beamformed)
        if include == PingType.IQ:
            return (p for p in pings if p.has_raw_iq)
        raise NotImplementedError(f"Encountered unknown PingType: {include!r}")

    def get_first_offset(
        self, record_type: int, offset_start: int, offset_end: int
    ) -> Optional[int]:
        """
        Get the offset of the first record of type record_type which has a
        file offset between offset_start and offset_end.
        """
        offsets = self._get_offsets(record_type)
        i = bisect.bisect_right(offsets, offset_start)
        return offsets[i] if i < len(offsets) and offsets[i] < offset_end else None

    def read_first_record(
        self, record_type: int, offset_start: int, offset_end: int
    ) -> Optional[records.BaseRecord]:
        """
        Read the first record of type record_type which has a file offset between
        offset_start and offset_end.
        """
        offset = self.get_first_offset(record_type, offset_start, offset_end)
        return self._read_record(record_type, offset) if offset is not None else None

    def read_records_during_ping(
        self,
        record_type: int,
        ping_start: datetime,
        ping_end: datetime,
        offset_hint: int,
    ) -> List[records.BaseRecord]:
        """
        Read all records of record_type which are timestamped in the interval between
        ping_start and ping_end. An offset_hint is given as an initial offset of a record
        close to the interval, to be used if it can make the search more efficient.
        """
        # Performs a brute-force search starting around the offset_hint. If the
        # hint is good (which it should usually be), this is pretty efficient.
        #
        # Records of different types are not guaranteed to be chronological, so
        # we cannot know a specific record interval to search.
        read_record = self._read_record
        offsets = self._get_offsets(record_type)
        initial_index = bisect.bisect_left(offsets, offset_hint)

        # Search forward in file
        forward_records = []
        searching_backward = True
        for index in range(initial_index, len(offsets)):
            next_record = read_record(record_type, offsets[index])
            next_record_time = next_record.frame.time
            if next_record_time > ping_end:
                # Reached upper end of interval
                break
            elif next_record_time <= ping_start:
                # Did not yet reach interval, backward search is unnecessary
                searching_backward = False
            else:
                forward_records.append(next_record)

        if not searching_backward:
            return forward_records

        # Search backward in file
        backward_records = []
        for index in range(initial_index - 1, -1, -1):
            next_record = read_record(record_type, offsets[index])
            next_record_time = next_record.frame.time
            if next_record_time < ping_start:
                # Reached lower end of interval
                break
            elif next_record_time >= ping_end:
                # Did not yet reach interval
                pass
            else:
                backward_records.append(next_record)

        # Discovered in reverse order, so un-reverse
        backward_records.reverse()
        backward_records.extend(forward_records)
        return backward_records

    def _read_record(self, record_type: int, offset: int) -> records.BaseRecord:
        """Read a record of record_type at the given offset"""
        return _datarecord.record(record_type).read(self._get_stream_for_read(offset))

    def _iter_offset_records(
        self, record_type: int
    ) -> Iterator[Tuple[int, records.BaseRecord]]:
        """Generate all the (offset, record) tuples for the given record type"""
        read_record = _datarecord.record(record_type).read
        get_stream = self._get_stream_for_read
        for offset in self._get_offsets(record_type):
            yield offset, read_record(get_stream(offset))

    def _get_offsets(self, record_type: int) -> Sequence[int]:
        """Return all the offsets for the given record type"""
        try:
            return self.__cached_offsets[record_type]
        except (AttributeError, KeyError) as ex:
            offsets: List[int] = []
            if record_type != 7300:
                catalog = self.file_catalog
                offsets.extend(
                    offset
                    for offset, rt in zip(catalog.offsets, catalog.record_types)
                    if rt == record_type
                )
            else:
                # the file catalog does not contain an entry for the 7300 record
                offsets.append(self.file_header.catalog_offset)

            if isinstance(ex, AttributeError):
                self.__cached_offsets: Dict[int, Sequence[int]] = {}
            return self.__cached_offsets.setdefault(record_type, offsets)

    @abstractmethod
    def _get_stream_for_read(self, offset: int) -> BinaryIO:
        """Return a byte stream for reading a record at the given offset"""


class S7KFileReader(S7KReader):
    """Reader class for s7k files"""

    def __init__(self, filename: str):
        self._filename = filename
        self._fhandle = open(self._filename, "rb", buffering=0)

    def _get_stream_for_read(self, offset: int) -> BinaryIO:
        self._fhandle.seek(offset)
        return self._fhandle

    def __getstate__(self) -> Dict[str, Any]:
        """ Remove unpicklable file handle from dict before pickling. """
        state = self.__dict__.copy()
        del state["_fhandle"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """ Open new file handle after unpickling. """
        self.__dict__.update(state)
        self._fhandle = open(self._filename, "rb", buffering=0)

    def __del__(self) -> None:
        self._fhandle.close()


class Ping:
    """
    A sound ping from a sonar, with associated data about settings and conditions.
    Properties of the ping are loaded efficiently on-demand.
    """

    def __init__(
        self,
        offset_record: Tuple[int, records.SonarSettings],
        next_offset_record: Optional[Tuple[int, records.SonarSettings]],
        reader: S7KReader,
    ):
        self._reader = reader
        self._offset, sonar_settings = offset_record
        self._ping_number = sonar_settings.ping_number
        self._sample_rate = sonar_settings.sample_rate
        self._ping_start = sonar_settings.frame.time
        if next_offset_record is not None:
            self._next_offset, next_sonar_settings = next_offset_record
            self._next_ping_start = next_sonar_settings.frame.time
        else:
            self._next_offset = sys.maxsize
            self._next_ping_start = datetime.max

    def __str__(self) -> str:
        return f"<Ping {self.ping_number}>"

    @property
    def ping_number(self) -> int:
        return self._ping_number

    @property
    def configuration(self) -> records.Configuration:
        """Return the 7001 record, which is shared for all pings in a file"""
        return self._reader.configuration

    @cached_property
    def position_set(self) -> List[records.Position]:
        """ Returns all 1003 records timestamped within this ping. """
        return cast(List[records.Position], self._read_records(1003))

    @cached_property
    def roll_pitch_heave_set(self) -> List[records.RollPitchHeave]:
        """ Returns all 1012 records timestamped within this ping. """
        return cast(List[records.RollPitchHeave], self._read_records(1012))

    @cached_property
    def heading_set(self) -> List[records.Heading]:
        """ Returns all 1013 records timestamped within this ping. """
        return cast(List[records.Heading], self._read_records(1013))

    @cached_property
    def beam_geometry(self) -> Optional[records.BeamGeometry]:
        """ Returns 7004 record """
        return cast(Optional[records.BeamGeometry], self._read_record(7004))

    @cached_property
    def tvg(self) -> Optional[records.TVG]:
        """ Returns 7010 record """
        return cast(Optional[records.TVG], self._read_record(7010))

    @cached_property
    def has_beamformed(self) -> bool:
        """ Checks if the ping has 7018 data without reading it. """
        return (
            self._reader.get_first_offset(7018, self._offset, self._next_offset)
            is not None
        )

    @cached_property
    def beamformed(self) -> Optional[records.Beamformed]:
        """ Returns 7018 record """
        return cast(Optional[records.Beamformed], self._read_record(7018))

    @cached_property
    def has_raw_iq(self) -> bool:
        """ Checks if the ping has 7038 data without reading it. """
        return (
            self._reader.get_first_offset(7038, self._offset, self._next_offset)
            is not None
        )

    @cached_property
    def raw_iq(self) -> Optional[records.RawIQ]:
        """ Returns 7038 record """
        return cast(Optional[records.RawIQ], self._read_record(7038))

    @cached_property
    def gps_position(self) -> geopy.Point:
        lat = self.position_set[0].latitude * 180 / np.pi
        long = self.position_set[0].longitude * 180 / np.pi
        return geopy.Point(lat, long)

    def receiver_motion_for_sample(
        self, sample: int
    ) -> Tuple[records.RollPitchHeave, records.Heading]:
        """ Find the most appropriate motion data for a sample based on time """
        time = self._ping_start + timedelta(seconds=sample / self._sample_rate)
        rph_index = min(
            bisect.bisect_left([m.frame.time for m in self.roll_pitch_heave_set], time),
            len(self.roll_pitch_heave_set) - 1,
        )
        heading_index = min(
            bisect.bisect_left([m.frame.time for m in self.heading_set], time),
            len(self.heading_set) - 1,
        )
        return self.roll_pitch_heave_set[rph_index], self.heading_set[heading_index]

    def minimize_memory(self) -> None:
        """
        Clears all memory-heavy properties.
        Retains offsets for easy reloading.
        """
        for key in "beamformed", "tvg", "beam_geometry", "raw_iq":
            if key in self.__dict__:
                del self.__dict__[key]

    def _read_record(self, record_type: int) -> Optional[records.BaseRecord]:
        record = self._reader.read_first_record(
            record_type, self._offset, self._next_offset
        )
        if record is not None:
            ping_number = self.ping_number
            assert getattr(record, "ping_number", ping_number) == ping_number
        return record

    def _read_records(self, record_type: int) -> List[records.BaseRecord]:
        return self._reader.read_records_during_ping(
            record_type, self._ping_start, self._next_ping_start, self._offset
        )


class PingDataset:
    """
    Indexable dataset returning Pings from a 7k file.

    Provides random access into pings in a file with minimal overhead.
    """

    def __init__(self, filename: str, include: PingType = PingType.ANY):
        """
        if include argument is not ANY, pings will be filtered.
        """
        self.pings = list(S7KFileReader(filename).iter_pings(include))
        self.__ping_numbers = [p.ping_number for p in self.pings]

    @property
    def ping_numbers(self) -> List[int]:
        return self.__ping_numbers

    def minimize_memory(self) -> None:
        for p in self.pings:
            p.minimize_memory()

    def __len__(self) -> int:
        return len(self.pings)

    def index_of(self, ping_number: int) -> int:
        return self.__ping_numbers.index(ping_number)

    def get_by_number(
        self, ping_number: int, default: Optional[Ping] = None
    ) -> Optional[Ping]:
        if not isinstance(ping_number, int):
            raise TypeError("Ping number must be an integer")
        try:
            ping_index = self.ping_numbers.index(ping_number)
        except ValueError:
            return default
        return self.pings[ping_index]

    def __getitem__(self, index: Union[int, slice]) -> Union[Ping, List[Ping]]:
        return self.pings[index]


class ConcatDataset:
    """
    Dataset concatenation object
    """

    def __init__(self, datasets):
        self.cum_lengths = np.cumsum([len(d) for d in datasets])
        self.datasets = datasets
        self.__ping_numbers = [pn for ds in datasets for pn in ds.ping_numbers]

    def __len__(self) -> int:
        return self.cum_lengths[-1]

    @property
    def ping_numbers(self) -> List[int]:
        return self.__ping_numbers

    def index_of(self, ping_number: int) -> int:
        return self.ping_numbers.index(ping_number)

    def get_by_number(
        self, ping_number: int, default: Optional[int] = None
    ) -> Union[Ping, None]:
        if not isinstance(ping_number, int):
            raise TypeError("Ping number must be an integer")
        for ds in self.datasets:
            if (ping_index := ds.get_by_number(ping_number, default)) is not None:
                return ds[ping_index]
        return default

    def __getitem__(self, index: Union[slice, int]) -> Union[Ping, List[Ping]]:
        if not isinstance(index, slice):
            if index < 0:
                if -index > len(self):
                    raise ValueError("Index out of range")
                index = len(self) + index
            dataset_index = np.searchsorted(self.cum_lengths, index, side="right")
            if dataset_index == 0:
                sample_index = index
            else:
                sample_index = index - self.cum_lengths[dataset_index - 1]
            return self.datasets[dataset_index][sample_index]
        else:
            return [self[i] for i in range(*index.indices(len(self)))]
