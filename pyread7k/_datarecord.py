"""
Low-level classes for reading various 7k record types.
"""
# pylint: disable=invalid-name unnecessary-comprehension
import abc
import io
from collections import namedtuple
from typing import Any, Dict, Optional

import numpy as np

from ._datablock import DataBlock, DRFBlock, elemD_, elemT
from . import records


class DataRecord(metaclass=abc.ABCMeta):
    """
    Base class for all record readers.

    Subclasses provide functionality for reading specific records.
    These are NOT the classes returned to the library user, they are only readers.
    """

    _block_drf = DRFBlock()
    _block_checksum = DataBlock((("checksum", ("u32",)),))
    implemented: Optional[Dict[int, Any]] = None
    _record_type_id = None

    def read(self, source: io.RawIOBase):
        """Base record reader"""
        start_offset = source.tell()
        drf = self._block_drf.read(source)
        source.seek(start_offset)
        source.seek(4, io.SEEK_CUR)  # to sync pattern
        source.seek(drf.offset, io.SEEK_CUR)

        parsed_data = self._read(source, drf, start_offset)

        checksum = self._block_checksum.read(source)["checksum"]
        if drf.flags & 0b1 > 0: # Check if checksum is valid
            drf.checksum = checksum
        source.seek(start_offset)  # reset source to start

        return parsed_data

    @classmethod
    def instance(cls, record_type_id: int):
        """Gets a specific datarecord by type id"""
        if not cls.implemented is None:
            return cls.implemented.get(record_type_id, None)
        subclasses = cls.__subclasses__()
        cls.implemented = dict((c.record_type_id(), c()) for c in subclasses)
        return cls.implemented.get(record_type_id, None)

    @abc.abstractmethod
    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        # returns iterable of dicts:
        #    0: tuple of rth values (required)
        #    1: rd values (if not available, return None)
        #    2: od values (if not available, return None)
        raise NotImplementedError

    @classmethod
    def record_type_id(cls):
        """return data record type id"""
        return cls._record_type_id


class _DataRecord7000(DataRecord):

    """Sonar Settings"""

    _record_type_id = 7000

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

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        return records.SonarSettings(**rth, frame=drf)


def _bytes_to_str(dict, keys):
    """
    For each key, the corresponding dict value is transformed from 
    a list of bytes to a string
    """
    for key in keys:
        byte_list = dict[key]
        termination = byte_list.index(b"\x00")
        dict[key] = b"".join(byte_list[:termination]).decode("UTF-8")


class _DataRecord7200(DataRecord):

    _record_type_id = 7200

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

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        _bytes_to_str(rth,
            ["recording_name", "recording_program_version_number",
            "user_defined_name", "notes"])
        rd = self._block_rd_device_type.read(source, rth["number_of_devices"])
        source.seek(start_offset)
        source.seek(drf.optional_data_offset, io.SEEK_CUR)
        od = self._block_od.read(source)

        # return rth, rd, od
        return records.FileHeader(**rth, **rd, **od, frame=drf)


class _DataRecord7300(DataRecord):

    _record_type_id = 7300

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
            elemD_("times", elemT.u8, 10),
            elemD_("record_counts", elemT.u32),
            elemD_(None, elemT.u16, 8),
        )
    )

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        rd = self._block_rd_entry.read(source, rth["number_of_records"])
        return records.FileCatalog(**rth, **rd, frame=drf)


class _DataRecord7004(DataRecord):
    """Beam Geometry"""

    _record_type_id = 7004
    _block_rth = DataBlock(
        (elemD_("sonar_id", elemT.u64), elemD_("number_of_beams", elemT.u32))
    )

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        n_beams = rth["number_of_beams"]
        block_rd = DataBlock(
            (
                elemD_("vertical_angles", elemT.f32, n_beams),
                elemD_("horizontal_angles", elemT.f32, n_beams),
                elemD_("beam_width_ys", elemT.f32, n_beams),
                elemD_("beam_width_xs", elemT.f32, n_beams),
                elemD_("tx_delays", elemT.f32, n_beams), # TODO: handle when missing
            )
        )
        array_rd = block_rd.read_dense(source)
        # Convert to dictionary
        rd = {k[0]: array_rd[k[0]].squeeze() for k in block_rd.numpy_types}
        return records.BeamGeometry(**rth, **rd, frame=drf)


class _DataRecord7010(DataRecord):
    """ TVG Values """

    _record_type_id = 7010
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

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        sample_count = rth["number_of_samples"]
        rd = self._block_gain_sample.read_dense(source, sample_count)
        return records.TVG(**rth, gains=rd["gains"], frame=drf)


class _DataRecord7018(DataRecord):
    """ Beamformed data """

    _record_type_id = 7018
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

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        n_samples = rth["number_of_samples"]
        n_beams = rth["number_of_beams"]
        count = n_samples * n_beams
        rd = self._block_rd_amp_phs.read_dense(source, count)
        rd = rd.reshape((n_samples, n_beams))

        return records.Beamformed(**rth, amplitudes=rd["amp"], phases=rd["phs"], frame=drf)


class _DataRecord7038(DataRecord):
    """ IQ data """

    _record_type_id = 7038
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

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)

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


class _DataRecord1003(DataRecord):
    """Position - GPS Coordinates"""

    _record_type_id = 1003
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

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        return records.Position(**rth, frame=drf)


class _DataRecord1012(DataRecord):
    """Roll Pitch Heave"""

    _record_type_id = 1012
    _block_rth = DataBlock(
        (
            elemD_("roll", elemT.f32),
            elemD_("pitch", elemT.f32),
            elemD_("heave", elemT.f32),
        )
    )

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        return records.RollPitchHeave(**rth, frame=drf)


class _DataRecord1013(DataRecord):
    """Heading"""

    _record_type_id = 1013
    _block_rth = DataBlock((elemD_("heading", elemT.f32),))

    def _read(self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int):
        rth = self._block_rth.read(source)
        rd = None  # no rd
        od = None  # no optional data
        return records.Heading(**rth, frame=drf)


def record(type_id: int) -> DataRecord:
    """Get a s7k record reader by record id """

    rec = DataRecord.instance(type_id)
    if rec is None:
        raise ValueError(f"DataRecord with type-ID " f"{type_id} is not supported.")
    return rec
