"""
This is a simple example of using the Ping type to combine several record types
to perform basic outlier detection, and pinpoint position of the detection.
"""
from pyread7k import ConcatDataset, PingDataset, PingType
import matplotlib.pyplot as plt
import numpy as np

# Setup dataset
path = "/home/localadmin/sonar_data/2020-06-03_F50_Wreck_seabed_target_Raw_and_beamformed/Port_200kHz_CW_125m_E/20200603_130905.s7k"
dataset = PingDataset(path, include=PingType.BEAMFORMED)
ping = dataset[40]  # Arbitrarily chosen

# Simple range and tvg compensation
gain_transform = np.arange(ping.tvg.data["gain"].shape[0]) ** 3 / 10 ** (
    ping.tvg.data["gain"] / 20
)
compensated_amp = ping.beamformed.data["amp"] * gain_transform[:, None]

# Find max intensity
point_range, point_beam = np.unravel_index(
    np.argmax(compensated_amp, axis=None), compensated_amp.shape
)

# Translate indices to real-world metrics
point_direction = ping.beam_geometry.data["horz_angle"][point_beam]
point_distance = (
    point_range
    * ping.sonar_settings.header["sound_velocity"]
    / ping.sonar_settings.header["freq"]
)

print(
    "Detection at %.2f meters in direction %.2f rad" % (point_distance, point_direction)
)
