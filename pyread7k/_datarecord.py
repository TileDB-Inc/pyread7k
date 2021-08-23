"""
Low-level classes for reading various 7k record types.
"""
# pylint: disable=invalid-name unnecessary-comprehension
from __future__ import annotations

import io
from abc import ABCMeta, abstractmethod
from typing import Dict, Type
from xml.etree import ElementTree as ET

import numpy as np

from . import records
from ._datablock import DataBlock, DRFBlock, elemD_, elemT, parse_7k_timestamp


def _bytes_to_str(dict, keys):
    """
    For each key, the corresponding dict value is transformed from
    a list of bytes to a string
    """
    for key in keys:
        byte_list = dict[key]
        termination = byte_list.index(b"\x00")
        dict[key] = b"".join(byte_list[:termination]).decode("UTF-8")


class DataRecord(metaclass=ABCMeta):
    """
    Base class for all record readers.

    Subclasses provide functionality for reading specific records.
    These are NOT the classes returned to the library user, they are only readers.
    """

    _block_drf = DRFBlock()
    _block_checksum = DataBlock((("checksum", ("u32",)),))
    _registry: Dict[int, Type[DataRecord]] = {}

    def __init_subclass__(cls, record_type_id: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[record_type_id] = cls

    @classmethod
    def get_class(cls, record_type_id: int) -> Type[DataRecord]:
        """Get a DataRecord subclass by record id"""
        try:
            return cls._registry[record_type_id]
        except KeyError:
            raise ValueError(f"DataRecord with type-ID={record_type_id} not supported")

    @classmethod
    def read(cls, source: io.RawIOBase):
        """Base record reader"""
        start_offset = source.tell()
        drf = cls._block_drf.read(source)
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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        """
        Return iterable of dicts:

            0: tuple of rth values (required)
            1: rd values (if not available, return None)
            2: od values (if not available, return None)
        """


class _DataRecord7000(DataRecord, record_type_id=7000):
    """Sonar Settings"""

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        return records.SonarSettings(**rth, frame=drf)


class _DataRecord7001(DataRecord, record_type_id=7001):
    """Configuration"""

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        rd = []
        for _ in range(rth["number_of_devices"]):
            device_data = cls._block_rd_info.read(source)
            _bytes_to_str(device_data, ["description"])
            xml_string = source.read(device_data["info_length"])
            # Indexing removes a weird null-termination
            device_data["info"] = ET.fromstring(xml_string[:-1])
            rd.append(records.DeviceConfiguration(**device_data))
        return records.Configuration(**rth, devices=rd, frame=drf)


class _DataRecord7200(DataRecord, record_type_id=7200):

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        _bytes_to_str(
            rth,
            [
                "recording_name",
                "recording_program_version_number",
                "user_defined_name",
                "notes",
            ],
        )
        rd = cls._block_rd_device_type.read(source, rth["number_of_devices"])
        source.seek(start_offset)
        source.seek(drf.optional_data_offset, io.SEEK_CUR)
        od = cls._block_od.read(source)
        return records.FileHeader(**rth, **rd, **od, frame=drf)


class _DataRecord7300(DataRecord, record_type_id=7300):

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        rd = cls._block_rd_entry.read(source, rth["number_of_records"])
        times_bytes = rd["times"]
        rd["times"] = tuple(
            parse_7k_timestamp(b"".join(times_bytes[i : i + 10]))
            for i in range(0, len(times_bytes), 10)
        )
        return records.FileCatalog(**rth, **rd, frame=drf)


class _DataRecord7004(DataRecord, record_type_id=7004):
    """Beam Geometry"""

    _block_rth = DataBlock(
        (elemD_("sonar_id", elemT.u64), elemD_("number_of_beams", elemT.u32))
    )

    @classmethod
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
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
        return records.BeamGeometry(**rth, **rd, frame=drf)


class _DataRecord7010(DataRecord, record_type_id=7010):
    """TVG Values"""

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        sample_count = rth["number_of_samples"]
        rd = cls._block_gain_sample.read_dense(source, sample_count)
        return records.TVG(**rth, gains=rd["gains"], frame=drf)


class _DataRecord7018(DataRecord, record_type_id=7018):
    """Beamformed data"""

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        n_samples = rth["number_of_samples"]
        n_beams = rth["number_of_beams"]
        count = n_samples * n_beams
        rd = cls._block_rd_amp_phs.read_dense(source, count)
        rd = rd.reshape((n_samples, n_beams))
        return records.Beamformed(
            **rth, amplitudes=rd["amp"], phases=rd["phs"], frame=drf
        )


class _DataRecord7038(DataRecord, record_type_id=7038):
    """IQ data"""

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
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
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
        return records.RawIQ(**rth, iq=rd_value, frame=drf)


class _DataRecord1003(DataRecord, record_type_id=1003):
    """Position - GPS Coordinates"""

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

    @classmethod
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        return records.Position(**rth, frame=drf)


class _DataRecord1012(DataRecord, record_type_id=1012):
    """Roll Pitch Heave"""

    _block_rth = DataBlock(
        (
            elemD_("roll", elemT.f32),
            elemD_("pitch", elemT.f32),
            elemD_("heave", elemT.f32),
        )
    )

    @classmethod
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        return records.RollPitchHeave(**rth, frame=drf)


class _DataRecord1013(DataRecord, record_type_id=1013):
    """Heading"""

    _block_rth = DataBlock((elemD_("heading", elemT.f32),))

    @classmethod
    def _read(
        cls, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = cls._block_rth.read(source)
        rd = None  # no rd
        od = None  # no optional data
        return records.Heading(**rth, frame=drf)


# for backwards compatibility
record = DataRecord.get_class
