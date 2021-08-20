"""
Class definitions for Data Format Definition records.
Records returned by the pyread7k API are of these types, NOT of _datarecord types.

Naming conventions:
Classes are named after their DFD entry, excluding any redundat "data" or "record" endings.
Fields are named as closely after DFD as possible, preferring verbose over ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np


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
class BaseRecord:
    """
    The base from which all records inherit.
    """

    frame: DataRecordFrame
    record_type: int  # Should be overridden by subclasses


@dataclass
class Position(BaseRecord):
    """
    Record 1003
    Global or local positioning data. Also see record 1011.
    Depending on position_type, this is either latitude/longitude data
    or northing/easting data. The corresponding properties are None if
    they are not available.
    """

    record_type: int = field(default=1003, init=False)

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


@dataclass
class RollPitchHeave(BaseRecord):
    """
    Record 1012
    Vessel motion data
    """

    record_type: int = field(default=1012, init=False)

    roll: float
    pitch: float
    heave: float


@dataclass
class Heading(BaseRecord):
    """ Record 1013 """

    record_type: int = field(default=1013, init=False)

    heading: float


@dataclass
class SonarSettings(BaseRecord):
    """
    Record 7000
    Contains the current sonar settings.
    """

    record_type: int = field(default=7000, init=False)

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
class Configuration(BaseRecord):
    """ Record 7001 """

    record_type: int = field(default=7001, init=False)

    sonar_serial_number: int
    number_of_devices: int
    devices: list[DeviceConfiguration]


@dataclass
class BeamGeometry(BaseRecord):
    """ Record 7004 """

    record_type: int = field(default=7004, init=False)

    sonar_id: int
    number_of_beams: int
    vertical_angles: np.ndarray
    horizontal_angles: np.ndarray
    beam_width_ys: np.ndarray
    beam_width_xs: np.ndarray
    tx_delays: Optional[np.ndarray] = None


@dataclass
class TVG(BaseRecord):
    """ Record 7010 """

    record_type: int = field(default=7010, init=False)

    sonar_id: int
    ping_number: int
    multi_ping_sequence: int
    number_of_samples: int

    gains: np.ndarray


@dataclass
class Beamformed(BaseRecord):
    """
    Record 7018
    Contains the sonar beam intensity (magnitude) and phase data.
    """

    record_type: int = field(default=7018, init=False)

    sonar_id: int
    ping_number: int
    is_multi_ping: bool
    number_of_beams: int
    number_of_samples: int

    amplitudes: np.ndarray
    phases: np.ndarray


@dataclass
class RawIQ(BaseRecord):
    """
    Record 7038
    Raw IQ data. Draft definition!
    """

    record_type: int = field(default=7038, init=False)
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


@dataclass
class FileHeader(BaseRecord):
    """ Record 7200. First record of 7k data file. """

    record_type: int = field(default=7200, init=False)

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


@dataclass
class FileCatalog(BaseRecord):
    """Record 7300.
    7k file catalog record, placed at the end of log files.
    record_type : int = field(default=7300, init=False)

    The file catalog contains one entry for each record in the log file,
    including the 7200 file header record, but excluding the 7300 file catalog
    record. The information corresponds to the record frame, plus the offset in
    the file.
    """

    record_type: int = field(default=7300, init=False)

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
