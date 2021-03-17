"""
Class definitions for Data Format Definition records.
"""
from typing import NamedTuple
import numpy as np
import datetime

from dataclasses import dataclass


@dataclass
class DataRecordFrame:
    protocol_version : int
    offset : int
    sync_pattern : int
    size : int
    optional_data_offset : int
    optional_data_id : int
    time : datetime.datetime
    record_version : int
    record_type_id : int
    device_id : int
    system_enumerator : int
    flags : int


@dataclass
class BaseRecord:
    frame : DataRecordFrame

@dataclass
class SonarSettings(BaseRecord):
    sonar_id : int
    ping_number : int
    multi_ping_sequence : int
    frequency : float
    sample_rate : float
    receiver_bandwidth : float
    tx_pulse_width : float
    tx_pulse_type_id : float
    tx_pulse_type_id : int
    tx_pulse_envelope_id : int
    tx_pulse_envelope_parameter : float
    tx_pulse_mode : int
    max_ping_rate : float
    ping_period : float
    range_selection : float
    power_selection : float
    gain_selection : float
    control_flags : int
    projector_id : int
    projector_beam_angle_vertical : float
    projector_beam_angle_horizontal : float
    projector_beam_width_vertical : float
    projector_beam_width_horizontal : float
    projector_beam_focal_point : float
    projector_beam_weighting_window_type : int
    projector_beam_weighting_window_parameter : float
    transmit_flags : int
    hydrophone_id : int
    receive_beam_weighting_window : int
    receive_beam_weighting_parameter : float
    receive_flags : int
    receive_beam_width : float
    bottom_detection_filter_min_range : float
    bottom_detection_filter_max_range : float
    bottom_detection_filter_min_depth : float
    bottom_detection_filter_max_depth : float
    absorption : float
    sound_velocity : float
    spreading : float

@dataclass
class Beamformed(BaseRecord):
    sonar_id : int
    ping_number : int
    is_multi_ping : bool
    number_of_beams : int
    number_of_samples : int
    amplitudes : np.ndarray
    phases : np.ndarray