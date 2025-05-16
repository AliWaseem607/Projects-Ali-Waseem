# %%

import datetime
import sys
import time

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import ALL_TIMES, CoordPair, destagger, getvar, interplevel, interpline, to_np

sys.path.append("./")
from utils import get_plot_end_time, get_plot_start_time

# %%
# Load data for comparison of all times available


data_times = pd.read_csv("data/data_times.csv")
file_ids = list(data_times.file_id)
HAC_met = pd.read_csv("data/HAC_meteo.csv")
HAC_met.time = pd.to_datetime(HAC_met.time)
sorted_data_times = data_times.copy()
sorted_data_times["start_time"] = pd.to_datetime(sorted_data_times.start_time)
sorted_data_times["end_time"] = pd.to_datetime(sorted_data_times.end_time)
sorted_data_times = sorted_data_times.sort_values("start_time").reset_index(drop=True)
sorted_data_times["plot_indx"] = 0
latest_value = {0: sorted_data_times.iloc[0]["start_time"]}
for i, row in sorted_data_times.iterrows():
    add_new_row = True
    for key in range(len(latest_value)):
        if row.start_time + pd.Timedelta(row.spinup, "h") > latest_value[key]:
            latest_value[key] = row.end_time
            sorted_data_times.loc[i, "plot_indx"] = key
            add_new_row = False
            break
    if add_new_row == True:
        new_key = len(latest_value)
        latest_value[new_key] = row.end_time
        sorted_data_times.loc[i, "plot_indx"] = new_key

# %%
# get overlapped sections
overlaps = []
sorted_data_times["true_start_time"] = sorted_data_times.start_time + sorted_data_times.spinup.apply(lambda x: pd.Timedelta(x, "h"))
for i in range(1, len(sorted_data_times)):
    if sorted_data_times.iloc[i]["true_start_time"] < np.max(sorted_data_times.iloc[:i]["end_time"]):
        overlaps.append((sorted_data_times.iloc[i]["true_start_time"],np.max(sorted_data_times.iloc[:i]["end_time"])))

HAC = pd.read_csv("data/HAC_meteo.csv")
HAC.time = pd.to_datetime(HAC_met.time)
# %%

spinup = int(24*60/5)
for i, row in sorted_data_times.iterrows():
    mask_1 = sorted_data_times.index > i
    mask_2 = sorted_data_times.true_start_time < row.end_time
    mask_3 = sorted_data_times.end_time > row.true_start_time
    mask = mask_1 & mask_2 & mask_3
    overlapped_df = sorted_data_times.loc[mask]

    if overlapped_df.empty:
        continue

    ncfile = Dataset(f"data/wrfout/wrfout_CHPN_d03_{row.file_id}_HAC.nc")
    times = pd.Series(getvar(ncfile, "Times", meta=False, timeidx=None))[spinup:] # type: ignore
    time_set_1 = set(times)
    HAC_times = set(HAC.time)

    for j, overlap_row in overlapped_df.iterrows():
        overlap_ncfile = Dataset(f"data/wrfout/wrfout_CHPN_d03_{overlap_row.file_id}_HAC.nc")
        overlap_times = pd.Series(getvar(overlap_ncfile, "Times", meta=False, timeidx=None))[spinup:] # type: ignore
        min_time = np.max([np.min(overlap_times), np.min(times)])
        max_time = np.min([np.max(overlap_times), np.max(times)])
        time_set_2 = set(overlap_times)
        times_both = time_set_1.intersection(time_set_2).intersection(HAC_times)


        time_mask = [x in times_both for x in times]
        overlap_time_mask = [x in times_both for x in overlap_times]
        HAC_mask = [x in times_both for x in HAC.time]

        temp_2m_1 = ncfile.variables["T2"][spinup:][time_mask]
        rh_2m_1 = getvar(ncfile, "rh2", meta=False, timeidx=None)[spinup:][time_mask]

        temp_2m_2 = overlap_ncfile.variables["T2"][spinup:][overlap_time_mask]
        rh_2m_2 = getvar(overlap_ncfile, "rh2", meta=False, timeidx=None)[spinup:][overlap_time_mask]
        
        
        RMSE_temp_1 = np.sqrt(np.mean((temp_2m_1 - HAC.loc[HAC_mask, "temp"].to_numpy()-273.15)**2))
        RMSE_rh2_1 = np.sqrt(np.mean((rh_2m_1 - HAC.loc[HAC_mask, "rh"].to_numpy())**2))

        RMSE_temp_2 = np.sqrt(np.mean((temp_2m_2 - HAC.loc[HAC_mask, "temp"].to_numpy()-273.15)**2))
        RMSE_rh2_2 = np.sqrt(np.mean((rh_2m_2 - HAC.loc[HAC_mask, "rh"].to_numpy())**2))
        
        print(min_time, max_time)
        print(f"early: {RMSE_temp_1}, late: {RMSE_temp_2}")
        print(f"early: {RMSE_rh2_1}, late: {RMSE_rh2_2}")
        print()





# %%
