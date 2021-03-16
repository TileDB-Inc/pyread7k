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

DataPartsBase = namedtuple("DataParts", ("drf", "rth", "rd", "od", "checksum"))


class DataParts(DataPartsBase):
    """
    Extends DataPartsBase to provide some more descriptive names
    """

    @property
    def frame(self):
        return self.drf

    @property
    def header(self):
        return self.rth

    @property
    def data(self):
        return self.rd

    @property
    def optional_data(self):
        return self.od

    def __str__(self):
        msg = (
            "DataParts("
            + f"type={self.frame.record_type_id},"
            + f"time={self.frame.time})"
        )
        return msg


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
        rth, rd, od = self._read(source, drf, start_offset)
        checksum = self._block_checksum.read(source)["checksum"]
        source.seek(start_offset)  # reset source to start
        return DataParts(**dict(zip(DataParts._fields, (drf, rth, rd, od, checksum))))

    @classmethod
    def instance(cls, record_type_id: int):
        """Gets a specific datarecord by type id"""
        if not cls.implemented is None:
            return cls.implemented.get(record_type_id, None)
        subclasses = cls.__subclasses__()
        cls.implemented = dict((c.record_type_id(), c()) for c in subclasses)
        return cls.implemented.get(record_type_id, None)

    @abc.abstractmethod
    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
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
            elemD_("is_multi_ping", elemT.u16),
            elemD_("freq", elemT.f32),
            elemD_("sample_rate", elemT.f32),
            elemD_("recv_bandwidth", elemT.f32),
            elemD_("tx_pulse_width", elemT.f32),
            elemD_("tx_pulse_type_id", elemT.u32),
            elemD_("tx_pulse_env_id", elemT.u32),
            elemD_("tx_pulse_env_param", elemT.f32),
            elemD_("tx_pulse_mode", elemT.u16),
            elemD_(None, elemT.u16),
            elemD_("max_ping_rate", elemT.f32),
            elemD_("ping_period", elemT.f32),
            elemD_("range", elemT.f32),
            elemD_("power", elemT.f32),
            elemD_("gain", elemT.f32),
            elemD_("ctrl_flags", elemT.u32),
            elemD_("proj_id", elemT.u32),
            elemD_("proj_beam_ang_vert", elemT.f32),
            elemD_("proj_beam_ang_horz", elemT.f32),
            elemD_("proj_beam_width_vert", elemT.f32),
            elemD_("proj_beam_width_horz", elemT.f32),
            elemD_("proj_beam_focal_point", elemT.f32),
            elemD_("proj_beam_weight_win_type", elemT.u32),
            elemD_("proj_beam_weight_win_param", elemT.f32),
            elemD_("tx_flags", elemT.u32),
            elemD_("hydrophone_id", elemT.u32),
            elemD_("recv_beam_weight_win", elemT.u32),
            elemD_("recv_beam_weight_param", elemT.f32),
            elemD_("rx_flags", elemT.u32),
            elemD_("recv_beam_width", elemT.f32),
            elemD_("bottom_detect_info_0", elemT.f32),
            elemD_("bottom_detect_info_1", elemT.f32),
            elemD_("bottom_detect_info_2", elemT.f32),
            elemD_("bottom_detect_info_3", elemT.f32),
            elemD_("absorption", elemT.f32),
            elemD_("sound_velocity", elemT.f32),
            elemD_("spreading", elemT.f32),
            elemD_(None, elemT.u16),
        )
    )

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        return rth, None, None


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
            elemD_("recording_prog_ver_num", elemT.c8, 16),
            elemD_("user_def_name", elemT.c8, 64),
            elemD_("notes", elemT.c8, 128),
        )
    )

    _block_rd_device_type = DataBlock(
        (elemD_("device_type_id", elemT.u32), elemD_("system_enum_id", elemT.u16))
    )

    _block_od = DataBlock(
        (elemD_("catalog_size", elemT.u32), elemD_("catalog_offset", elemT.u64))
    )

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        rd = self._block_rd_device_type.read(source, rth["number_of_devices"])
        source.seek(start_offset)
        source.seek(drf.od_offset, io.SEEK_CUR)
        od = self._block_od.read(source)
        return rth, rd, od


class _DataRecord7300(DataRecord):

    _record_type_id = 7300

    _block_rth = DataBlock(
        (
            elemD_("size", elemT.u32),
            elemD_("version", elemT.u16),
            elemD_("catalog_size", elemT.u32),
            elemD_(None, elemT.u32),
        )
    )

    _block_rd_entry = DataBlock(
        (
            elemD_("size", elemT.u32),
            elemD_("offset", elemT.u64),
            elemD_("record_type_id", elemT.u16),
            elemD_("device_id", elemT.u16),
            elemD_("system_enum", elemT.u16),
            elemD_("time", elemT.u8, 10),
            elemD_("count", elemT.u32),
            elemD_(None, elemT.u16, 8),
        )
    )

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        rd = self._block_rd_entry.read(source, rth["catalog_size"])
        return rth, rd, None


class _DataRecord7004(DataRecord):
    """Beam Geometry"""

    _record_type_id = 7004
    _block_rth = DataBlock(
        (elemD_("sonar_id", elemT.u64), elemD_("n_beams", elemT.u32))
    )

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        n_beams = rth["n_beams"]
        block_rd = DataBlock(
            (
                elemD_("vert_angle", elemT.f32, n_beams),
                elemD_("horz_angle", elemT.f32, n_beams),
                elemD_("beam_width_y", elemT.f32, n_beams),
                elemD_("beam_width_x", elemT.f32, n_beams),
                elemD_("tx_delay", elemT.f32, n_beams),
            )
        )
        # the dtype becomes rather complex, with the n_beams
        # dimension embedded into each type. This is inverted
        # by creating a new structure dtype of 5 scalars. A
        # new array of size (n_beams,) is created and
        # scalar data # is copied into it.
        orig_rd = block_rd.read_dense(source)
        new_dtype = np.dtype([(n, s) for n, s, *_ in block_rd.numpy_types])
        rd = np.zeros((n_beams,), dtype=new_dtype)
        rd["vert_angle"][:] = orig_rd["vert_angle"]
        rd["horz_angle"][:] = orig_rd["horz_angle"]
        rd["beam_width_y"][:] = orig_rd["beam_width_y"]
        rd["beam_width_x"][:] = orig_rd["beam_width_x"]
        rd["tx_delay"][:] = orig_rd["tx_delay"]
        return rth, rd, None


class _DataRecord7010(DataRecord):
    """ TVG Values """

    _record_type_id = 7010
    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("is_multi_ping", elemT.u16),
            elemD_("sample_count", elemT.u32),
            elemD_(None, elemT.u32, 8),
        )
    )

    _block_gain_sample = DataBlock((elemD_("gain", elemT.f32),))

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        sample_count = rth["sample_count"]
        rd = self._block_gain_sample.read_dense(source, sample_count)
        return rth, rd, None


class _DataRecord7018(DataRecord):
    """ Beamformed data """

    _record_type_id = 7018
    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("is_multi_ping", elemT.u16),
            elemD_("n_beams", elemT.u16),
            elemD_("n_samples", elemT.u32),
            elemD_(None, elemT.u32, 8),
        )
    )

    _block_rd_amp_phs = DataBlock((elemD_("amp", elemT.u16), elemD_("phs", elemT.i16)))

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        n_samples = rth["n_samples"]
        n_beams = rth["n_beams"]
        count = n_samples * n_beams
        rd = self._block_rd_amp_phs.read_dense(source, count)
        rd = rd.reshape((n_samples, n_beams))
        return rth, rd, None


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

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
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

        return rth, rd_value, None


class _DataRecord1003(DataRecord):
    """Position - GPS Coordinates"""

    _record_type_id = 1003
    _block_rth = DataBlock(
        (
            elemD_("datum_id", elemT.u32),
            elemD_("latency", elemT.f32),
            elemD_("lat", elemT.f64),
            elemD_("long", elemT.f64),
            elemD_("height", elemT.f64),
            elemD_("pos_type", elemT.u8),
            elemD_("utm_zone", elemT.u8),
            elemD_("quality", elemT.u8),
            elemD_("pos_method", elemT.u8),
            elemD_("num_sat", elemT.u8),
        )
    )

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        return rth, None, None


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

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        return rth, None, None


class _DataRecord1013(DataRecord):
    """Heading"""

    _record_type_id = 1013
    _block_rth = DataBlock((elemD_("heading", elemT.f32),))

    def _read(self, source: io.RawIOBase, drf: DRFBlock.DRF, start_offset: int):
        rth = self._block_rth.read(source)
        rd = None  # no rd
        od = None  # no optional data
        return rth, rd, od


def record(type_id: int) -> DataRecord:
    """Get a s7k record reader by record id """

    rec = DataRecord.instance(type_id)
    if rec is None:
        raise ValueError(f"DataRecord with type-ID " f"{type_id} is not supported.")
    return rec
