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
from typing import Optional, Sequence, Union
from xml.etree import ElementTree as ET

import numpy as np

from ._datablock import DataBlock, elemD_, elemT


def record(record_type_id: int):
    """Get a s7k record reader by record id"""
    try:
        return BaseRecord._registry[record_type_id]
    except KeyError:
        raise ValueError(f"Records with type-ID={record_type_id} are not supported")


def _parse_7k_timestamp(bs: bytes) -> datetime:
    """Parse a timestamp from a bytes object"""
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

    def __str__(self):
        return f"DataRecordFrame(record_type_id={self.record_type_id}, time={str(self.time)}"


@dataclass
class BaseRecord(metaclass=ABCMeta):
    """
    The base from which all records inherit.
    """

    frame: DataRecordFrame

    _block_drf = DataBlock(
        (
            elemD_("protocol_version", elemT.u16),
            elemD_("offset", elemT.u16),
            elemD_("sync_pattern", elemT.u32),
            elemD_("size", elemT.u32),
            elemD_("optional_data_offset", elemT.u32),
            elemD_("optional_data_id", elemT.u32),
            elemD_("time", elemT.c8, 10),
            elemD_("record_version", elemT.u16),
            elemD_("record_type_id", elemT.u32),
            elemD_("device_id", elemT.u32),
            elemD_(None, elemT.u16),
            elemD_("system_enumerator", elemT.u16),
            elemD_(None, elemT.u32),
            elemD_("flags", elemT.u16),
            elemD_(None, elemT.u16),
            elemD_(None, elemT.u32),
            elemD_(None, elemT.u32),
            elemD_(None, elemT.u32),
        )
    )
    _block_checksum = DataBlock((("checksum", ("u32",)),))
    _registry = {}

    def __init_subclass__(cls, record_type_id: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[record_type_id] = cls

    @classmethod
    def read(cls, source: io.RawIOBase) -> BaseRecord:
        """Read a record of record_type_id from source"""
        start_offset = source.tell()
        drf_dict = cls._block_drf.read(source)
        # convert time from bytes to datetime
        drf_dict["time"] = _parse_7k_timestamp(b"".join(drf_dict["time"]))
        drf = DataRecordFrame(**drf_dict)
        source.seek(start_offset)
        source.seek(4, io.SEEK_CUR)  # to sync pattern
        source.seek(drf.offset, io.SEEK_CUR)
        parsed_data = cls._read(source, drf, start_offset)
        checksum = cls._block_checksum.read(source)["checksum"]
        if drf.flags & 0b1 > 0:  # Check if checksum is valid
            drf.checksum = checksum
        source.seek(start_offset)  # reset source to start
        return parsed_data

    @classmethod
    @abstractmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
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
        (
            elemD_("datum_id", elemT.u32),
            elemD_("latency", elemT.f32),
            elemD_("latitude_northing", elemT.f64),
            elemD_("longitude_easting", elemT.f64),
            elemD_("height", elemT.f64),
            elemD_("position_type", elemT.u8),
            elemD_("utm_zone", elemT.u8),
            elemD_("quality_flag", elemT.u8),
            elemD_("positioning_method", elemT.u8),
            elemD_("number_of_satellites", elemT.u8),
        )
    )

    @property
    def latitude(self):
        """ Latitude is only available for position type 0 """
        if self.position_type == 0:
            return self.latitude_northing

    @property
    def longitude(self):
        """ Longitude is only available for position type 0 """
        if self.position_type == 0:
            return self.longitude_easting

    @property
    def northing(self):
        """ Northing is only available for position type 1 """
        if self.position_type == 1:
            return self.latitude_northing

    @property
    def easting(self):
        """ Easting is only available for position type 1 """
        if self.position_type == 1:
            return self.longitude_easting

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        return cls(**rth, frame=drf)


@dataclass
class RollPitchHeave(BaseRecord, record_type_id=1012):
    """Record 1012 - Vessel motion data"""

    roll: float
    pitch: float
    heave: float

    _block_rth = DataBlock(
        (
            elemD_("roll", elemT.f32),
            elemD_("pitch", elemT.f32),
            elemD_("heave", elemT.f32),
        )
    )

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        return cls(**rth, frame=drf)


@dataclass
class Heading(BaseRecord, record_type_id=1013):
    """Record 1013 - Heading"""

    heading: float

    _block_rth = DataBlock((elemD_("heading", elemT.f32),))

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        rd = None  # no rd
        od = None  # no optional data
        return Heading(**rth, frame=drf)


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
    tx_pulse_type_id: float
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
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("multi_ping_sequence", elemT.u16),
            elemD_("frequency", elemT.f32),
            elemD_("sample_rate", elemT.f32),
            elemD_("receiver_bandwidth", elemT.f32),
            elemD_("tx_pulse_width", elemT.f32),
            elemD_("tx_pulse_type_id", elemT.u32),
            elemD_("tx_pulse_envelope_id", elemT.u32),
            elemD_("tx_pulse_envelope_parameter", elemT.f32),
            elemD_("tx_pulse_mode", elemT.u16),
            elemD_(None, elemT.u16),
            elemD_("max_ping_rate", elemT.f32),
            elemD_("ping_period", elemT.f32),
            elemD_("range_selection", elemT.f32),
            elemD_("power_selection", elemT.f32),
            elemD_("gain_selection", elemT.f32),
            elemD_("control_flags", elemT.u32),
            elemD_("projector_id", elemT.u32),
            elemD_("projector_beam_angle_vertical", elemT.f32),
            elemD_("projector_beam_angle_horizontal", elemT.f32),
            elemD_("projector_beam_width_vertical", elemT.f32),
            elemD_("projector_beam_width_horizontal", elemT.f32),
            elemD_("projector_beam_focal_point", elemT.f32),
            elemD_("projector_beam_weighting_window_type", elemT.u32),
            elemD_("projector_beam_weighting_window_parameter", elemT.f32),
            elemD_("transmit_flags", elemT.u32),
            elemD_("hydrophone_id", elemT.u32),
            elemD_("receive_beam_weighting_window", elemT.u32),
            elemD_("receive_beam_weighting_parameter", elemT.f32),
            elemD_("receive_flags", elemT.u32),
            elemD_("receive_beam_width", elemT.f32),
            elemD_("bottom_detection_filter_min_range", elemT.f32),
            elemD_("bottom_detection_filter_max_range", elemT.f32),
            elemD_("bottom_detection_filter_min_depth", elemT.f32),
            elemD_("bottom_detection_filter_max_depth", elemT.f32),
            elemD_("absorption", elemT.f32),
            elemD_("sound_velocity", elemT.f32),
            elemD_("spreading", elemT.f32),
            elemD_(None, elemT.u16),
        )
    )

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        return cls(**rth, frame=drf)


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
    devices: list[DeviceConfiguration]

    _block_rth = DataBlock(
        (
            elemD_("sonar_serial_number", elemT.u64),
            elemD_("number_of_devices", elemT.u32),
        )
    )
    _block_rd_info = DataBlock(
        (
            elemD_("identifier", elemT.u32),
            elemD_("description", elemT.c8, 60),  # We should parse this better
            elemD_("alphadata_card", elemT.u32),
            elemD_("serial_number", elemT.u64),
            elemD_("info_length", elemT.u32),
        )
    )

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        rd = []
        for _ in range(rth["number_of_devices"]):
            device_data = cls._block_rd_info.read(source)
            device_data["description"] = _bytes_to_str(device_data["description"])
            xml_string = source.read(device_data["info_length"])
            # Indexing removes a weird null-termination
            device_data["info"] = ET.fromstring(xml_string[:-1])
            rd.append(DeviceConfiguration(**device_data))
        return cls(**rth, devices=rd, frame=drf)


@dataclass
class BeamGeometry(BaseRecord, record_type_id=7004):
    """Record 7004 - Beam Geometry"""

    sonar_id: int
    number_of_beams: int
    vertical_angles: np.ndarray
    horizontal_angles: np.ndarray
    beam_width_ys: np.ndarray
    beam_width_xs: np.ndarray
    tx_delays: Optional[np.ndarray] = None

    _block_rth = DataBlock(
        (elemD_("sonar_id", elemT.u64), elemD_("number_of_beams", elemT.u32))
    )

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        n_beams = rth["number_of_beams"]
        block_rd_size = (
            drf.size
            - cls._block_drf.size
            - cls._block_checksum.size
            - cls._block_rth.size
        )
        block_rd_elements = (
            elemD_("vertical_angles", elemT.f32, n_beams),
            elemD_("horizontal_angles", elemT.f32, n_beams),
            elemD_("beam_width_ys", elemT.f32, n_beams),
            elemD_("beam_width_xs", elemT.f32, n_beams),
            elemD_("tx_delays", elemT.f32, n_beams),
        )
        block_rd = DataBlock(block_rd_elements)
        if block_rd.size != block_rd_size:
            # tx_delays missing
            block_rd = DataBlock(block_rd_elements[:-1])
            assert block_rd.size == block_rd_size, (block_rd.size, block_rd_size)

        array_rd = block_rd.read_dense(source)
        # Convert to dictionary
        rd = {k[0]: array_rd[k[0]].squeeze() for k in block_rd.numpy_types}
        return cls(**rth, **rd, frame=drf)


@dataclass
class TVG(BaseRecord, record_type_id=7010):
    """Record 7010 - TVG Values"""

    sonar_id: int
    ping_number: int
    multi_ping_sequence: int
    number_of_samples: int
    gains: np.ndarray

    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("multi_ping_sequence", elemT.u16),
            elemD_("number_of_samples", elemT.u32),
            elemD_(None, elemT.u32, 8),
        )
    )
    _block_gain_sample = DataBlock((elemD_("gains", elemT.f32),))

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        sample_count = rth["number_of_samples"]
        rd = cls._block_gain_sample.read_dense(source, sample_count)
        return cls(**rth, gains=rd["gains"], frame=drf)


@dataclass
class Beamformed(BaseRecord, record_type_id=7018):
    """Record 7018 - sonar beam intensity (magnitude) and phase data"""

    sonar_id: int
    ping_number: int
    is_multi_ping: bool
    number_of_beams: int
    number_of_samples: int
    amplitudes: np.ndarray
    phases: np.ndarray

    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("is_multi_ping", elemT.u16),
            elemD_("number_of_beams", elemT.u16),
            elemD_("number_of_samples", elemT.u32),
            elemD_(None, elemT.u32, 8),
        )
    )
    _block_rd_amp_phs = DataBlock((elemD_("amp", elemT.u16), elemD_("phs", elemT.i16)))

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        n_samples = rth["number_of_samples"]
        n_beams = rth["number_of_beams"]
        count = n_samples * n_beams
        rd = cls._block_rd_amp_phs.read_dense(source, count)
        rd = rd.reshape((n_samples, n_beams))
        return cls(**rth, amplitudes=rd["amp"], phases=rd["phs"], frame=drf)


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
    channel_array: np.ndarray
    iq: np.ndarray

    _block_rth = DataBlock(
        (
            elemD_("serial_number", elemT.u64),  # Sonar serial number
            elemD_("ping_number", elemT.u32),  # Sequential number
            elemD_(None, elemT.u16),  # Reserved (zeroed) but see note 1 below
            elemD_("channel_count", elemT.u16),  # Num system Rx elements
            elemD_("n_samples", elemT.u32),  # Num samples within ping
            elemD_("n_actual_channels", elemT.u16),  # Num elems in record
            elemD_("start_sample", elemT.u32),  # First sample in record
            elemD_("stop_sample", elemT.u32),  # Last sample in record
            elemD_("sample_type", elemT.u16),  # Sample type ID
            elemD_(None, elemT.u32, 7),
        )
    )  # Reserved (zeroed)
    # Note 1: Original DFD20724.docx document defines this element as
    # 'Reserved u16'. The MATLAB reader parses this as "multipingSequence".
    # This implementation follows the document and sets as reserved.
    _block_rd_data_u16 = DataBlock((elemD_("amp", elemT.u16), elemD_("phs", elemT.i16)))

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        n_actual_channels = rth["n_actual_channels"]
        block_channel_array = DataBlock(
            (elemD_("channel_array", elemT.u16, n_actual_channels),)
        )
        channel_array = block_channel_array.read_dense(source)
        channel_array = np.squeeze(channel_array["channel_array"])
        rth["channel_array"] = channel_array
        n_actual_samples = rth["stop_sample"] - rth["start_sample"] + 1
        sample_type = rth["sample_type"]

        def f_block_actual_data(elem_type):
            return DataBlock(
                (
                    elemD_(
                        "actual_data",
                        elem_type,
                        n_actual_channels * n_actual_samples * 2,
                    ),
                )
            )

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

        if sample_type == 8:
            # from MATLAB reader:
            actual_data = f_block_actual_data(elemT.i8).read_dense(source)
            actual_data = np.squeeze(actual_data["actual_data"])
            actual_data[actual_data < 0] += 65536
            actual_data *= 16
            actual_data[actual_data > 2047] -= 4096
        elif sample_type == 16:
            actual_data = f_block_actual_data(elemT.i16).read_dense(source)
            actual_data = np.squeeze(actual_data["actual_data"])
        else:
            # Data is either invalid (0) or 12 bit (not supported):
            rd = dict(value=f"Unsupported sample type ID {sample_type}")
            return rth, rd, None  # <-- early RETURN

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
        return cls(**rth, iq=rd_value, frame=drf)


@dataclass
class FileHeader(BaseRecord, record_type_id=7200):
    """Record 7200 - First record of 7k data file"""

    file_id: int
    version_number: int
    session_id: int
    record_data_size: int
    number_of_devices: int
    recording_name: str
    recording_program_version_number: str
    user_defined_name: str
    notes: str
    device_ids: tuple[int, ...]
    system_enumerators: tuple[int, ...]
    catalog_size: int
    catalog_offset: int

    _block_rth = DataBlock(
        (
            elemD_("file_id", elemT.u64, 2),
            elemD_("version_number", elemT.u16),
            elemD_(None, elemT.u16),
            elemD_("session_id", elemT.u64, 2),
            elemD_("record_data_size", elemT.u32),
            elemD_("number_of_devices", elemT.u32),
            elemD_("recording_name", elemT.c8, 64),
            elemD_("recording_program_version_number", elemT.c8, 16),
            elemD_("user_defined_name", elemT.c8, 64),
            elemD_("notes", elemT.c8, 128),
        )
    )
    _block_rd_device_type = DataBlock(
        (elemD_("device_ids", elemT.u32), elemD_("system_enumerators", elemT.u16))
    )
    _block_od = DataBlock(
        (elemD_("catalog_size", elemT.u32), elemD_("catalog_offset", elemT.u64))
    )

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        for key in (
            "recording_name",
            "recording_program_version_number",
            "user_defined_name",
            "notes",
        ):
            rth[key] = _bytes_to_str(rth[key])
        rd = cls._block_rd_device_type.read(source, rth["number_of_devices"])
        source.seek(start_offset)
        source.seek(drf.optional_data_offset, io.SEEK_CUR)
        od = cls._block_od.read(source)
        return cls(**rth, **rd, **od, frame=drf)


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
    sizes: list[int]
    offsets: list[int]
    record_types: list[int]
    device_ids: list[int]
    system_enumerators: list[int]
    times: list[datetime]
    record_counts: list[int]

    _block_rth = DataBlock(
        (
            elemD_("size", elemT.u32),
            elemD_("version", elemT.u16),
            elemD_("number_of_records", elemT.u32),
            elemD_(None, elemT.u32),
        )
    )
    _block_rd_entry = DataBlock(
        (
            elemD_("sizes", elemT.u32),
            elemD_("offsets", elemT.u64),
            elemD_("record_types", elemT.u16),
            elemD_("device_ids", elemT.u16),
            elemD_("system_enumerators", elemT.u16),
            elemD_("times", elemT.c8, 10),
            elemD_("record_counts", elemT.u32),
            elemD_(None, elemT.u16, 8),
        )
    )

    @classmethod
    def _read(cls, source: io.RawIOBase, drf: DataRecordFrame, start_offset: int):
        rth = cls._block_rth.read(source)
        rd = cls._block_rd_entry.read(source, rth["number_of_records"])
        times_bytes = rd["times"]
        rd["times"] = tuple(
            _parse_7k_timestamp(b"".join(times_bytes[i : i + 10]))
            for i in range(0, len(times_bytes), 10)
        )
        return cls(**rth, **rd, frame=drf)
