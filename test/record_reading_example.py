# %%
import numpy as np
import matplotlib.pyplot as plt

import pyread7k as r7k

import os

data_root = os.environ['PATH_7K_DATA_ROOT']
data_path = r'2019-07-12 F50 Mines Elefantgrunden/SoloMine/45 degree tilt/400 kHz/20190712_085609.s7k'
filename = data_root + '/' + data_path

# iq_filepath = "/home/localadmin/sonar_data/_Scenarios/7/20200930_095427.s7k"

with open(filename, 'rb', buffering=0) as fid:
    file_header = r7k.read_file_header(fid)
    file_catalog = r7k.read_file_catalog(fid, file_header)
    n_records = r7k.get_record_count(7018, file_catalog)
    print(f"Total num records = {n_records}")
    count = 10
    records = dict()
    records[7000] = r7k.read_records(7000, fid, file_catalog, count=count)
    records[7001] = r7k.read_records(7001, fid, file_catalog, count=count)
    records[7038] = r7k.read_records(7038, fid, file_catalog, count=count)
    records[7200] = r7k.read_records(7200, fid, file_catalog, count=count)
    records[7004] = r7k.read_records(7004, fid, file_catalog, count=count)
    records[7018] = r7k.read_records(7018, fid, file_catalog, count=count)
    records[1003] = r7k.read_records(1003, fid, file_catalog, count=count)
    records[1012] = r7k.read_records(1012, fid, file_catalog, count=count)
    records[1013] = r7k.read_records(1013, fid, file_catalog, count=count)

# %%