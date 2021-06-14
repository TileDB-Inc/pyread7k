"""
This is a simple example of using the Ping type to combine several record types
to perform basic outlier detection, and pinpoint position of the detection.
"""
import matplotlib.pyplot as plt
import numpy as np

from pyread7k import ConcatDataset, PingDataset, PingType

# Setup dataset
path = "<path to s7k file>.s7k"
dataset = PingDataset(path, include=PingType.BEAMFORMED)
ping = dataset[40]  # Arbitrarily chosen

# Simple range and tvg compensation
gain_transform = np.arange(ping.tvg.gains.shape[0]) ** 3 / 10 ** (
    ping.tvg.gains / 20
)
compensated_amp = ping.beamformed.amplitudes * gain_transform[:, None]

# Find max intensity
point_range, point_beam = np.unravel_index(
    np.argmax(compensated_amp, axis=None), compensated_amp.shape
)

# Translate indices to real-world metrics
point_direction = ping.beam_geometry.horizontal_angles[point_beam]
point_distance = (
    point_range
    * ping.sonar_settings.sound_velocity
    / ping.sonar_settings.frequency
)

print(
    "Detection at %.2f meters in direction %.2f rad" % (point_distance, point_direction)
)
