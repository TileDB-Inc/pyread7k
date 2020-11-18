# %%
import numpy as np
import matplotlib.pyplot as plt

import pyread7k as r7k

import os

# %%
data_root = os.environ['PATH_7K_DATA_ROOT']
# data_path = r'reson_20200417\Raw and beamformed data from SeaBat F50\F50_200kHz_256_Nadir_to_45_20dBTarget_160m_1.s7k'
# filename = data_root + '\\' + data_path
filename = "/home/localadmin/sonar_data/2020-06-03_F50_Wreck_seabed_target_Raw_and_beamformed/Port_200kHz_CW_125m_E/20200603_130712.s7k"

with open(filename, 'rb', buffering=0) as fid:
    file_header = r7k.read_file_header(fid)
    file_catalog = r7k.read_file_catalog(fid, file_header)
    n_records = r7k.get_record_count(7018, file_catalog)
    print(f"Total num records = {n_records}")
    count = 1
    records = dict()
    records[7018] = r7k.read_records(7018, fid, file_catalog, count=count)
    records[7038] = r7k.read_records(7038, fid, file_catalog, count=count)

# %% Show first I/Q record:
plt.figure('Beamformed')
plt.imshow(records[7018][0].rd['amp'], origin='lower', aspect='auto')
plt.figure('Raw I')
plt.imshow(records[7038][0].rd['i'], origin='lower', aspect='auto')
plt.figure('Raw Q')
plt.imshow(records[7038][0].rd['q'], origin='lower', aspect='auto')

# %%
