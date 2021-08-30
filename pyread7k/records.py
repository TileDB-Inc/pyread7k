"""
Class definitions for Data Format Definition records.

Naming conventions:
Classes are named after their DFD entry, excluding any redundat "data" or "record" endings.
Fields are named as closely after DFD as possible, preferring verbose over ambiguous.
"""
from __future__ import annotations

import io
import struct
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any, BinaryIO, ClassVar, Dict, Optional, Sequence, Tuple, Type, Union
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from ._datablock import DataBlock, Element, ElementType


c8 = partial(Element, ElementType.c8)
i8 = partial(Element, ElementType.i8)
u8 = partial(Element, ElementType.u8)
i16 = partial(Element, ElementType.i16)
u16 = partial(Element, ElementType.u16)
u32 = partial(Element, ElementType.u32)
u64 = partial(Element, ElementType.u64)
f32 = partial(Element, ElementType.f32)
f64 = partial(Element, ElementType.f64)


def record(record_type_id: int) -> Type[BaseRecord]:
    """Get a s7k record reader by record id"""
    try:
        return BaseRecord._registry[record_type_id]
    except KeyError:
        raise ValueError(f"Records with type-ID={record_type_id} are not supported")


def _parse_7k_timestamp(bs: Union[bytes, Sequence[bytes]]) -> datetime:
    """Parse a timestamp from a bytes object"""
    if not isinstance(bs, bytes):
        bs = b"".join(bs)
    # We have raw days, datetime takes days and months. Easier to just add them
    # as timedelta, and let datetime worry about leap-whatever
    y, d, s, h, m = struct.unpack("<HHfBB", bs)
    t = datetime(year=y, month=1, day=1)
    t += timedelta(
        # subtract 1 since datetime already starts at 1
        days=d - 1,
        hours=h,
        minutes=m,
        seconds=s,
    )
    return t


def _bytes_to_str(b: Union[bytes, Sequence[bytes]], encoding: str = "UTF-8") -> str:
    """Convert a null-terminated byte string (or sequence of bytes) to unicode string"""
    non_null_bytes = b[: b.index(b"\x00")]
    if not isinstance(non_null_bytes, bytes):
        non_null_bytes = b"".join(non_null_bytes)
    return non_null_bytes.decode(encoding)


@dataclass
class DataRecordFrame:
    """
    The Data Record Frame is the wrapper in which all records (sensor data or
    otherwise) shall be embedded.
    """

    protocol_version: int
    offset: int
    sync_pattern: int
    size: int
    optional_data_offset: int
    optional_data_id: int
    time: datetime
    record_version: int
    record_type_id: int
    device_id: int
    system_enumerator: int
    flags: int
    checksum: Optional[int] = None

    _block_drf = DataBlock(
        u16("protocol_version"),
        u16("offset"),
        u32("sync_pattern"),
        u32("size"),
        u32("optional_data_offset"),
        u32("optional_data_id"),
        c8("time", 10),
        u16("record_version"),
        u32("record_type_id"),
        u32("device_id"),
        u16(),
        u16("system_enumerator"),
        u32(),
        u16("flags"),
        u16(),
        u32(),
        u32(),
        u32(),
    )
    _block_checksum = DataBlock(u32("checksum"))

    def __str__(self) -> str:
        return f"DataRecordFrame(record_type_id={self.record_type_id}, time={str(self.time)}"

    @property
    def embedded_data_size(self) -> int:
        """Size of the embedded data section in this record frame"""
        return self.size - self._block_drf.size - self._block_checksum.size

    @classmethod
    def read(cls, source: BinaryIO) -> DataRecordFrame:
        start_offset = source.tell()
        drf = cls._block_drf.read(source)
        # convert time from bytes to datetime
        drf["time"] = _parse_7k_timestamp(drf["time"])
        # read checksum at the end
        source.seek(start_offset + drf["size"] - cls._block_checksum.size)
        checksum = cls._block_checksum.read(source)["checksum"]
        if drf["flags"] & 0b1 > 0:  # Check if checksum is valid
            drf["checksum"] = checksum
        return cls(**drf)


@dataclass
class BaseRecord(metaclass=ABCMeta):
    """
    The base from which all records inherit.
    """

    frame: DataRecordFrame

    _registry: ClassVar[Dict[int, Type[BaseRecord]]] = {}

    def __init_subclass__(cls, record_type_id: int):
        cls._registry[record_type_id] = cls

    @classmethod
    def read(cls, source: BinaryIO) -> BaseRecord:
        start_offset = source.tell()
        try:
            drf = DataRecordFrame.read(source)
            source.seek(start_offset)
            source.seek(4, io.SEEK_CUR)  # to sync pattern
            source.seek(drf.offset, io.SEEK_CUR)
            return cls._read(source, drf, start_offset)
        finally:
            source.seek(start_offset)  # reset source to start

    @classmethod
    @abstractmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> BaseRecord:
        """Read a record of this class from source and data record frame drf"""


@dataclass
class Position(BaseRecord, record_type_id=1003):
    """Record 1003 - Global or local positioning data. Also see record 1011.

    Depending on position_type, this is either latitude/longitude data
    or northing/easting data. The corresponding properties are None if
    they are not available.
    """

    datum_id: int
    latency: float
    latitude_northing: float
    longitude_easting: float
    height: float
    position_type: int
    utm_zone: int
    quality_flag: int
    positioning_method: int
    number_of_satellites: int

    _block_rth = DataBlock(
        u32("datum_id"),
        f32("latency"),
        f64("latitude_northing"),
        f64("longitude_easting"),
        f64("height"),
        u8("position_type"),
        u8("utm_zone"),
        u8("quality_flag"),
        u8("positioning_method"),
        u8("number_of_satellites"),
    )

    @property
    def latitude(self) -> Optional[float]:
        """Latitude is only available for position type 0"""
        return self.latitude_northing if self.position_type == 0 else None

    @property
    def longitude(self) -> Optional[float]:
        """Longitude is only available for position type 0"""
        return self.longitude_easting if self.position_type == 0 else None

    @property
    def northing(self) -> Optional[float]:
        """Northing is only available for position type 1"""
        return self.latitude_northing if self.position_type == 1 else None

    @property
    def easting(self) -> Optional[float]:
        """Easting is only available for position type 1"""
        return self.longitude_easting if self.position_type == 1 else None

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> Position:
        return cls(drf, **cls._block_rth.read(source))


@dataclass
class RollPitchHeave(BaseRecord, record_type_id=1012):
    """Record 1012 - Vessel motion data"""

    roll: float
    pitch: float
    heave: float

    _block_rth = DataBlock(f32("roll"), f32("pitch"), f32("heave"))

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> RollPitchHeave:
        return cls(drf, **cls._block_rth.read(source))


@dataclass
class Heading(BaseRecord, record_type_id=1013):
    """Record 1013 - Heading"""

    heading: float

    _block_rth = DataBlock(f32("heading"))

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> Heading:
        return cls(drf, **cls._block_rth.read(source))


@dataclass
class SonarSettings(BaseRecord, record_type_id=7000):
    """Record 7000 - current sonar settings"""

    sonar_id: int
    ping_number: int
    multi_ping_sequence: int
    frequency: float
    sample_rate: float
    receiver_bandwidth: float
    tx_pulse_width: float
    tx_pulse_type_id: int
    tx_pulse_envelope_id: int
    tx_pulse_envelope_parameter: float
    tx_pulse_mode: int
    max_ping_rate: float
    ping_period: float
    range_selection: float
    power_selection: float
    gain_selection: float
    control_flags: int
    projector_id: int
    projector_beam_angle_vertical: float
    projector_beam_angle_horizontal: float
    projector_beam_width_vertical: float
    projector_beam_width_horizontal: float
    projector_beam_focal_point: float
    projector_beam_weighting_window_type: int
    projector_beam_weighting_window_parameter: float
    transmit_flags: int
    hydrophone_id: int
    receive_beam_weighting_window: int
    receive_beam_weighting_parameter: float
    receive_flags: int
    receive_beam_width: float
    bottom_detection_filter_min_range: float
    bottom_detection_filter_max_range: float
    bottom_detection_filter_min_depth: float
    bottom_detection_filter_max_depth: float
    absorption: float
    sound_velocity: float
    spreading: float

    _block_rth = DataBlock(
        u64("sonar_id"),
        u32("ping_number"),
        u16("multi_ping_sequence"),
        f32("frequency"),
        f32("sample_rate"),
        f32("receiver_bandwidth"),
        f32("tx_pulse_width"),
        u32("tx_pulse_type_id"),
        u32("tx_pulse_envelope_id"),
        f32("tx_pulse_envelope_parameter"),
        u16("tx_pulse_mode"),
        u16(),
        f32("max_ping_rate"),
        f32("ping_period"),
        f32("range_selection"),
        f32("power_selection"),
        f32("gain_selection"),
        u32("control_flags"),
        u32("projector_id"),
        f32("projector_beam_angle_vertical"),
        f32("projector_beam_angle_horizontal"),
        f32("projector_beam_width_vertical"),
        f32("projector_beam_width_horizontal"),
        f32("projector_beam_focal_point"),
        u32("projector_beam_weighting_window_type"),
        f32("projector_beam_weighting_window_parameter"),
        u32("transmit_flags"),
        u32("hydrophone_id"),
        u32("receive_beam_weighting_window"),
        f32("receive_beam_weighting_parameter"),
        u32("receive_flags"),
        f32("receive_beam_width"),
        f32("bottom_detection_filter_min_range"),
        f32("bottom_detection_filter_max_range"),
        f32("bottom_detection_filter_min_depth"),
        f32("bottom_detection_filter_max_depth"),
        f32("absorption"),
        f32("sound_velocity"),
        f32("spreading"),
        u16(),
    )

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> SonarSettings:
        return cls(drf, **cls._block_rth.read(source))


@dataclass
class DeviceConfiguration:
    """ Configuration of a single device in a 7001 record """

    identifier: int
    description: str
    alphadata_card: int
    serial_number: int
    info_length: int
    info: ET.ElementTree


@dataclass
class Configuration(BaseRecord, record_type_id=7001):
    """Record 7001 - Configuration"""

    sonar_serial_number: int
    number_of_devices: int
    devices: Sequence[DeviceConfiguration]

    _block_rth = DataBlock(u64("sonar_serial_number"), u32("number_of_devices"))
    _block_rd_info = DataBlock(
        u32("identifier"),
        c8("description", 60),  # We should parse this better
        u32("alphadata_card"),
        u64("serial_number"),
        u32("info_length"),
    )

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> Configuration:
        rth = cls._block_rth.read(source)
        devices = []
        for _ in range(rth["number_of_devices"]):
            device_data = cls._block_rd_info.read(source)
            device_data["description"] = _bytes_to_str(device_data["description"])
            xml_string = source.read(device_data["info_length"])
            # Indexing removes a weird null-termination
            device_data["info"] = ET.fromstring(xml_string[:-1])
            devices.append(DeviceConfiguration(**device_data))
        return cls(drf, **rth, devices=devices)


@dataclass
class BeamGeometry(BaseRecord, record_type_id=7004):
    """Record 7004 - Beam Geometry"""

    sonar_id: int
    number_of_beams: int
    vertical_angles: NDArray[np.float32]
    horizontal_angles: NDArray[np.float32]
    beam_width_ys: NDArray[np.float32]
    beam_width_xs: NDArray[np.float32]
    tx_delays: Optional[NDArray[np.float32]] = None

    _block_rth = DataBlock(u64("sonar_id"), u32("number_of_beams"))

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> BeamGeometry:
        rth = cls._block_rth.read(source)
        n_beams = rth["number_of_beams"]
        block_rd_size = drf.embedded_data_size - cls._block_rth.size
        block_rd_elements = (
            f32("vertical_angles", n_beams),
            f32("horizontal_angles", n_beams),
            f32("beam_width_ys", n_beams),
            f32("beam_width_xs", n_beams),
            f32("tx_delays", n_beams),
        )
        block_rd = DataBlock(*block_rd_elements)
        if block_rd.size != block_rd_size:
            # tx_delays missing
            block_rd = DataBlock(*block_rd_elements[:-1])
            assert block_rd.size == block_rd_size, (block_rd.size, block_rd_size)

        array_rd = block_rd.read_dense(source)
        # Convert to dictionary
        rd = {name: array_rd[name].squeeze() for name in block_rd.names}
        return cls(drf, **rth, **rd)


@dataclass
class TVG(BaseRecord, record_type_id=7010):
    """Record 7010 - TVG Values"""

    sonar_id: int
    ping_number: int
    multi_ping_sequence: int
    number_of_samples: int
    gains: NDArray[np.float32]

    _block_rth = DataBlock(
        u64("sonar_id"),
        u32("ping_number"),
        u16("multi_ping_sequence"),
        u32("number_of_samples"),
        u32(count=8),
    )
    _block_gain_sample = DataBlock(f32("gains"))

    @classmethod
    def _read(cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int) -> TVG:
        rth = cls._block_rth.read(source)
        sample_count = rth["number_of_samples"]
        rd = cls._block_gain_sample.read_dense(source, sample_count)
        return cls(drf, **rth, gains=rd["gains"])


@dataclass
class Beamformed(BaseRecord, record_type_id=7018):
    """Record 7018 - sonar beam intensity (magnitude) and phase data"""

    sonar_id: int
    ping_number: int
    is_multi_ping: bool
    number_of_beams: int
    number_of_samples: int
    amplitudes: NDArray[np.uint16]
    phases: NDArray[np.int16]

    _block_rth = DataBlock(
        u64("sonar_id"),
        u32("ping_number"),
        u16("is_multi_ping"),
        u16("number_of_beams"),
        u32("number_of_samples"),
        u32(count=8),
    )
    _block_rd_amp_phs = DataBlock(u16("amp"), i16("phs"))

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> Beamformed:
        rth = cls._block_rth.read(source)
        n_samples = rth["number_of_samples"]
        n_beams = rth["number_of_beams"]
        count = n_samples * n_beams
        rd = cls._block_rd_amp_phs.read_dense(source, count)
        rd = rd.reshape((n_samples, n_beams))
        return cls(drf, **rth, amplitudes=rd["amp"], phases=rd["phs"])


@dataclass
class RawIQ(BaseRecord, record_type_id=7038):
    """Record 7038 - Raw IQ data. Draft definition!"""

    serial_number: int
    ping_number: int
    channel_count: int
    n_samples: int
    n_actual_channels: int
    start_sample: int
    stop_sample: int
    sample_type: int
    channel_array: NDArray[np.uint16]
    iq: NDArray[Any]  # [('i', np.integer), ('q', np.integer)]]

    _block_rth = DataBlock(
        u64("serial_number"),  # Sonar serial number
        u32("ping_number"),  # Sequential number
        u16(),  # Reserved (zeroed) but see note 1 below
        u16("channel_count"),  # Num system Rx elements
        u32("n_samples"),  # Num samples within ping
        u16("n_actual_channels"),  # Num elems in record
        u32("start_sample"),  # First sample in record
        u32("stop_sample"),  # Last sample in record
        u16("sample_type"),  # Sample type ID
        u32(count=7),
    )  # Reserved (zeroed)
    # Note 1: Original DFD20724.docx document defines this element as
    # 'Reserved u16'. The MATLAB reader parses this as "multipingSequence".
    # This implementation follows the document and sets as reserved.
    _block_rd_data_u16 = DataBlock(u16("amp"), i16("phs"))

    @classmethod
    def _read(cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int) -> RawIQ:
        rth = cls._block_rth.read(source)
        n_actual_channels = rth["n_actual_channels"]
        block_channel_array = DataBlock(u16("channel_array", n_actual_channels))
        channel_array = block_channel_array.read_dense(source)
        channel_array = np.squeeze(channel_array["channel_array"])
        rth["channel_array"] = channel_array
        n_actual_samples = rth["stop_sample"] - rth["start_sample"] + 1
        sample_type = rth["sample_type"]

        # From document DFD20724.docx:
        # System data is always 16 bits I & Q. Sample type is used only for
        # the purpose of the compatibility with older tools. The following
        # values can be contained by the field:
        #    12 – Data is reported as i16 I and i16 Q aligned with four least
        #         significant bits truncated and aligned by LSB.
        #    16 – Data is reported as i16 I and i16 Q as received from Rx HW.
        #     8 – Data is reported as i8 I and i8 Q. Only most significant
        #         eight bits of 16-bit data are used.
        #     0 – Indicates that the data is not valid.

        count = n_actual_channels * n_actual_samples * 2
        if sample_type == 8:
            # from MATLAB reader:
            datablock = DataBlock(i8("actual_data", count))
            actual_data = datablock.read_dense(source)
            actual_data = np.squeeze(actual_data["actual_data"])
            actual_data[actual_data < 0] += 65536
            actual_data *= 16
            actual_data[actual_data > 2047] -= 4096
        elif sample_type == 16:
            datablock = DataBlock(i16("actual_data", count))
            actual_data = datablock.read_dense(source)
            actual_data = np.squeeze(actual_data["actual_data"])
        else:
            # Data is either invalid (0) or 12 bit (not supported):
            raise NotImplementedError(f"Unsupported sample type ID {sample_type}")

        rd_value = np.zeros(
            (rth["n_samples"], n_actual_channels),
            dtype=[(elem, actual_data.dtype.name) for elem in ("i", "q")],
        )
        rd_view = rd_value[rth["start_sample"] : rth["stop_sample"] + 1, :]
        rd_view["i"][:, channel_array] = actual_data[0::2].reshape(
            (-1, n_actual_channels)
        )
        rd_view["q"][:, channel_array] = actual_data[1::2].reshape(
            (-1, n_actual_channels)
        )
        return cls(drf, **rth, iq=rd_value)


@dataclass
class FileHeader(BaseRecord, record_type_id=7200):
    """Record 7200 - First record of 7k data file"""

    file_id: Tuple[int, int]
    version_number: int
    session_id: Tuple[int, int]
    record_data_size: int
    number_of_devices: int
    recording_name: str
    recording_program_version_number: str
    user_defined_name: str
    notes: str
    device_ids: Sequence[int]
    system_enumerators: Sequence[int]
    catalog_size: int
    catalog_offset: int

    _block_rth = DataBlock(
        u64("file_id", 2),
        u16("version_number"),
        u16(),
        u64("session_id", 2),
        u32("record_data_size"),
        u32("number_of_devices"),
        c8("recording_name", 64),
        c8("recording_program_version_number", 16),
        c8("user_defined_name", 64),
        c8("notes", 128),
    )
    _block_rd_device_type = DataBlock(u32("device_ids"), u16("system_enumerators"))
    _block_od = DataBlock(u32("catalog_size"), u64("catalog_offset"))

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> FileHeader:
        rth = cls._block_rth.read(source)
        for key in (
            "recording_name",
            "recording_program_version_number",
            "user_defined_name",
            "notes",
        ):
            rth[key] = _bytes_to_str(rth[key])
        rd = cls._block_rd_device_type.read_multiple(source, rth["number_of_devices"])
        source.seek(start_offset)
        source.seek(drf.optional_data_offset, io.SEEK_CUR)
        od = cls._block_od.read(source)
        return cls(
            drf,
            **rth,
            device_ids=rd["device_ids"],
            system_enumerators=rd["system_enumerators"],
            **od,
        )


@dataclass
class FileCatalog(BaseRecord, record_type_id=7300):
    """Record 7300 -7k file catalog record, placed at the end of log files.

    The file catalog contains one entry for each record in the log file,
    including the 7200 file header record, but excluding the 7300 file catalog
    record. The information corresponds to the record frame, plus the offset in
    the file.
    """

    size: int
    version: int
    number_of_records: int
    sizes: Sequence[int]
    offsets: Sequence[int]
    record_types: Sequence[int]
    device_ids: Sequence[int]
    system_enumerators: Sequence[int]
    times: Sequence[datetime]
    record_counts: Sequence[int]

    _block_rth = DataBlock(u32("size"), u16("version"), u32("number_of_records"), u32())
    _block_rd_entry = DataBlock(
        u32("sizes"),
        u64("offsets"),
        u16("record_types"),
        u16("device_ids"),
        u16("system_enumerators"),
        c8("times", 10),
        u32("record_counts"),
        u16(count=8),
    )

    @classmethod
    def _read(
        cls, source: BinaryIO, drf: DataRecordFrame, start_offset: int
    ) -> FileCatalog:
        rth = cls._block_rth.read(source)
        rd = cls._block_rd_entry.read_multiple(source, rth["number_of_records"])
        # convert time from bytes to datetime
        rd["times"] = list(map(_parse_7k_timestamp, rd["times"]))
        return cls(drf, rth["size"], rth["version"], rth["number_of_records"], **rd)
