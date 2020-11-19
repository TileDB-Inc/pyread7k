# %% 
# import itertools
# import math
import os
import datetime

import psutil

from pyread7k import PingDataset, PingType
# import pyread7k

path = "/home/localadmin/sonar_data/2020-06-03_F50_Wreck_seabed_target_Raw_and_beamformed/Port_200kHz_CW_125m_E/20200603_130712.s7k"
# %%

def print_current_memory():
    """ Prints current memory of active process """
    pid = os.getpid()
    own_process = psutil.Process(pid)
    memory_use = own_process.memory_info()[0] / (1024**2)
    print('Memory use:', memory_use, "MB")

# %%
dataset = PingDataset(path, include=PingType.IQ)
# %%
# for ping in dataset[:4]:
#     print("Ping time", ping.sonar_settings.frame.time)
#     for position in ping.position_set():
#         print(position.frame.time)
# %%

print("Before:")
print_current_memory()

before_time = datetime.datetime.now()
for ping in dataset:
    print(ping)
    print("- Position records:", len(ping.position_set))
    print("- Roll pitch heave records:", len(ping.roll_pitch_heave_set))
    print("- Heading records:", len(ping.heading_set))
    if ping.beamformed is not None:
        print("- Beamformed", ping.beamformed.data["amp"].shape)
        print("- Beam Geometry", ping.beam_geometry.data.shape)
    if ping.raw_iq is not None:
        print("- Raw IQ", ping.raw_iq.data["i"].shape)

after_time = datetime.datetime.now()
print("Time taken: %.4f s" % (after_time - before_time).total_seconds())

print("After:")
print_current_memory()

for ping in dataset:
    ping.minimize_memory()

print("Minimized:")
print_current_memory()
# %%
