#%%
import glob
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

#%%

wrfdirs = ["/scratch/waseem/CHOPIN_rain_oct6-9_MYJ25/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_MYJ25/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_MYJ25/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct12-15/wrfout_CHPN_d02_2024-10-12_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct12-15/wrfout_CHPN_d03_2024-10-12_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct12-15/wrfout_CHPN_d01_2024-10-12_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_KEPS/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_KEPS/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_KEPS/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct10-13/wrfout_CHPN_d03_2024-10-10_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct10-13/wrfout_CHPN_d02_2024-10-10_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct10-13/wrfout_CHPN_d01_2024-10-10_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct4-7/wrfout_CHPN_d01_2024-10-04_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct4-7/wrfout_CHPN_d03_2024-10-04_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct4-7/wrfout_CHPN_d02_2024-10-04_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_MYJ/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_MYJ/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_MYJ/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct5-8/wrfout_CHPN_d02_2024-10-05_12:00:00.nc",
"/scratch/waseem/CHOPIN_oct5-8/wrfout_CHPN_d01_2024-10-05_12:00:00.nc",
"/scratch/waseem/CHOPIN_oct5-8/wrfout_CHPN_d03_2024-10-05_12:00:00.nc",
"/scratch/waseem/CHOPIN_envelop_oct17-20/wrfout_CHPN_d03_2024-10-17_18:00:00.nc",
"/scratch/waseem/CHOPIN_envelop_oct17-20/wrfout_CHPN_d02_2024-10-17_18:00:00.nc",
"/scratch/waseem/CHOPIN_envelop_oct17-20/wrfout_CHPN_d01_2024-10-17_18:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct9-11/wrfout_CHPN_d01_2024-10-09_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct9-11/wrfout_CHPN_d03_2024-10-09_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct9-11/wrfout_CHPN_d02_2024-10-09_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_SH/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_SH/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_rain_oct6-9_SH/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
"/scratch/waseem/CHOPIN_oct1-4/wrfout_CHPN_d01_2024-10-01_06:00:00.nc",
"/scratch/waseem/CHOPIN_oct1-4/wrfout_CHPN_d03_2024-10-01_06:00:00.nc",
"/scratch/waseem/CHOPIN_oct1-4/wrfout_CHPN_d02_2024-10-01_06:00:00.nc"
]

dir_times = []
for dir in wrfdirs:
    mod_time = os.path.getmtime(dir)
    pd_time = pd.Timestamp(mod_time, unit="s")
    dir_times.append(pd_time)

data = pd.DataFrame({"path":wrfdirs, "time":dir_times}).sort_values("time", ascending=True).reset_index(drop=True)


for i, row in data.iterrows():
    print(f"cutting {row.path}")
    cmd = ["bash", "/home/waseem/scripts/cut_wrfout.sh", f"{row.path}"]
    subprocess.run(args=cmd)

paths = glob.glob("/scratch/waseem/CHOPIN*")
for path in paths:
    files = glob.glob(f"{path}/*-split*")
    if len(files) == 0:
        continue
    cmd = ["gzip"]
    cmd.extend(files)
    subprocess.run(cmd)