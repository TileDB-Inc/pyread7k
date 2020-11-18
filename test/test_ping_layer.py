# %% 
# import itertools
# import math

from pyread7k import PingDataset, PingType
# import pyread7k

path = "/home/localadmin/sonar_data/2020-06-03_F50_Wreck_seabed_target_Raw_and_beamformed/Port_200kHz_CW_125m_E/20200603_130712.s7k"
# %%

def print_current_memory():
    """ Prints current memory of active process """
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / (1024**2)
    print('Memory use:', memory_use, "MB")

# %%
dataset = PingDataset(path, include=PingType.BEAMFORMED)
# %%

print("Before:")
print_current_memory()

for ping in dataset:
    print(ping, ping.beamformed.data["amp"].shape)
    print(ping, ping.tvg.data["gain"].shape)

print("After:")
print_current_memory()

for ping in dataset:
    ping.minimize_memory()

print("Minimized:")
print_current_memory()
# %%
