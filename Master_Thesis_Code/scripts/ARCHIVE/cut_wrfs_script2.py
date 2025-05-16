#%%
import glob
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

#%%

# wrfdirs = [
# "/scratch/waseem/CHOPIN_clear_oct27-30_KEPS/wrfout_CHPN_d01_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30_KEPS/wrfout_CHPN_d02_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30_KEPS/wrfout_CHPN_d03_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30_MYNN/wrfout_CHPN_d01_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30_MYNN/wrfout_CHPN_d02_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30_MYNN/wrfout_CHPN_d03_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d01_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d02_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d03_2024-10-27_18:00:00.nc",
# "/scratch/waseem/CHOPIN_envelop_oct17-20_KEPS/wrfout_CHPN_d01_2024-10-17_18:00:00.nc",
# "/scratch/waseem/CHOPIN_envelop_oct17-20_KEPS/wrfout_CHPN_d02_2024-10-17_18:00:00.nc",
# "/scratch/waseem/CHOPIN_envelop_oct17-20_KEPS/wrfout_CHPN_d03_2024-10-17_18:00:00.nc",
# "/scratch/waseem/CHOPIN_envelop_oct17-20_MYNN/wrfout_CHPN_d01_2024-10-17_18:00:00.nc",
# "/scratch/waseem/CHOPIN_envelop_oct17-20_MYNN/wrfout_CHPN_d02_2024-10-17_18:00:00.nc",
# "/scratch/waseem/CHOPIN_envelop_oct17-20_MYNN/wrfout_CHPN_d03_2024-10-17_18:00:00.nc",
# "/scratch/waseem/CHOPIN_nov3-6/wrfout_CHPN_d01_2024-11-03_00:00:00.nc",
# "/scratch/waseem/CHOPIN_nov3-6/wrfout_CHPN_d02_2024-11-03_00:00:00.nc",
# "/scratch/waseem/CHOPIN_nov3-6/wrfout_CHPN_d03_2024-11-03_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct16-19/wrfout_CHPN_d01_2024-10-16_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct16-19/wrfout_CHPN_d02_2024-10-16_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct16-19/wrfout_CHPN_d03_2024-10-16_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct16-19/wrfout_CHPN_extended_d01_2024-10-19_00:30:00.nc",
# "/scratch/waseem/CHOPIN_oct16-19/wrfout_CHPN_extended_d02_2024-10-19_00:30:00.nc",
# "/scratch/waseem/CHOPIN_oct16-19/wrfout_CHPN_extended_d03_2024-10-19_full_dom.nc",
# "/scratch/waseem/CHOPIN_oct20-23/wrfout_CHPN_d01_2024-10-20_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct20-23/wrfout_CHPN_d02_2024-10-20_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct20-23/wrfout_CHPN_d03_2024-10-20_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct24-27/wrfout_CHPN_d01_2024-10-24_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct24-27/wrfout_CHPN_d02_2024-10-24_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct24-27/wrfout_CHPN_d03_2024-10-24_00:00:00.nc",
# "/scratch/waseem/CHOPIN_oct29-nov1/wrfout_CHPN_d01_2024-10-29_18:00:00.nc",
# "/scratch/waseem/CHOPIN_oct29-nov1/wrfout_CHPN_d02_2024-10-29_18:00:00.nc",
# "/scratch/waseem/CHOPIN_oct29-nov1/wrfout_CHPN_d03_2024-10-29_18:00:00.nc",
# "/scratch/waseem/CHOPIN_rain_oct6-9_THMP/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
# "/scratch/waseem/CHOPIN_rain_oct6-9_THMP/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
# "/scratch/waseem/CHOPIN_rain_oct6-9_THMP/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
# "/scratch/waseem/CHOPIN_rain_oct6-9_WDM6/wrfout_CHPN_d01_2024-10-06_00:00:00.nc",
# "/scratch/waseem/CHOPIN_rain_oct6-9_WDM6/wrfout_CHPN_d02_2024-10-06_00:00:00.nc",
# "/scratch/waseem/CHOPIN_rain_oct6-9_WDM6/wrfout_CHPN_d03_2024-10-06_00:00:00.nc",
# ]

# dir_times = []
# for dir in wrfdirs:
#     mod_time = os.path.getmtime(dir)
#     pd_time = pd.Timestamp(mod_time, unit="s")
#     dir_times.append(pd_time)

# data = pd.DataFrame({"path":wrfdirs, "time":dir_times}).sort_values("time", ascending=True).reset_index(drop=True)


# for i, row in data.iterrows():
#     print(f"cutting {row.path}")
#     cmd = ["bash", "/home/waseem/scripts/cut_wrfout.sh", f"{row.path}"]
#     subprocess.run(args=cmd)

paths = glob.glob("/scratch/waseem/CHOPIN*")
for path in paths:
    files = glob.glob(path+"/*-split*")
    if len(files) == 0:
        continue
    cmd = ["gzip", "-v"]
    for file in files:
        if file.endswith(".gz"):
            continue
        cmd.append(file)
    if len(cmd) == 2:
        continue
    
    subprocess.run(cmd)


paths = glob.glob("/scratch/waseem/CHOPIN*/*_NPRK*")
paths2 = glob.glob("/scratch/waseem/CHOPIN*/*_SUBGRID*")
paths3 = glob.glob("/scratch/waseem/CHOPIN*/*_SPRK*")
paths4 = glob.glob("/scratch/waseem/CHOPIN*/*_HAC*")

paths.extend(paths2)
paths.extend(paths3)
paths.extend(paths4)

cmd = ["gzip", "-v", "-k"]
cmd.extend(paths)
subprocess.run(cmd)