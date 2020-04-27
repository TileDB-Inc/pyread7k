# %%
import numpy as np
import matplotlib.pyplot as plt

import pyread7k as r7k

import os

# %%
data_root = os.environ['PATH_7K_DATA_ROOT']
data_path = r'reson_20200417\Raw and beamformed data from SeaBat F50\F50_200kHz_256_Nadir_to_45_20dBTarget_160m_1.s7k'
filename = data_root + '\\' + data_path

with open(filename, 'rb', buffering=0) as fid:
    file_header = r7k.read_file_header(fid)
    file_catalog = r7k.read_file_catalog(fid, file_header)
    n_records = r7k.get_record_count(7018, file_catalog)
    print(f"Total num records = {n_records}")
    count = 10
    records = dict()
    records[7038] = r7k.read_records(7038, fid, file_catalog, count=count)