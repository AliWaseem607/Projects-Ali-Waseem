# %%

import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar

sys.path.append("./")
from utils import get_plot_end_time, get_plot_start_time

# %%
# Gather data for comparison amongst rain periods
spin_up_idx = int(24 * 60 / 5)
HAC_met = pd.read_csv("data/HAC_meteo.csv")
HAC_met.time = pd.to_datetime(HAC_met.time)

runs = {
    "YSU": "Yonsei Univeristy",
    "MYJ": "Mello-Yamada-Janjic",
    "KEPS": "K-epsilon-theta^2",
    "SH": "Shin-Hong",
    "MYJ25": "Mellor-Yamada Nakanishi and Niino level 2.5",
    "THMP": "Thompson Microphysics",
    "WDM6": "WRF Double-moment 6-class",
}

colors = {
    "YSU": "cornflowerblue",
    "MYJ": "olivedrab",
    "KEPS": "darkorchid",
    "SH": "firebrick",
    "MYJ25": "forestgreen",
    "THMP": "dodgerblue",
    "WDM6": "deepskyblue",
}

temp2m = {}
rh2m = {}
pblh = {}
times = None
for id in runs.keys():
    ncfile = Dataset(f"data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_{id}.nc")
    temp2m[id] = np.squeeze(ncfile.variables["T2"][spin_up_idx:])
    rh2m[id] = getvar(ncfile, "rh2", timeidx=None, meta=False)[spin_up_idx:]
    if times is None:
        times = pd.Series(getvar(ncfile, "Times", timeidx=None, meta=False))[spin_up_idx:]  # type: ignore


HGT_NPRK = None
HGT_HAC = None

for id in runs.keys():
    ncfile = Dataset(f"data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_{id}.nc")
    pblh[id] = np.squeeze(ncfile.variables["PBLH"][spin_up_idx:])
    if HGT_HAC is None and HGT_NPRK is None:
        ncfile_HAC = Dataset(f"data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_{id}.nc")
        HGT_NPRK = np.squeeze(ncfile.variables["HGT"][0])
        HGT_HAC = np.squeeze(ncfile_HAC.variables["HGT"][0])


HGT_diff = HGT_HAC - HGT_NPRK  # type: ignore

tick_locs = mdates.drange(times.iloc[0], times.iloc[-1], pd.Timedelta(6, "h"))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

# %%
# Plot meteorological variables
fig, axs = plt.subplots(2, 1, figsize=(9, 7))

for id in runs.keys():
    axs[0].plot(times, temp2m[id], color=colors[id], label=runs[id], alpha=0.75)
    axs[1].plot(times, rh2m[id], color=colors[id], label=runs[id], alpha=0.75)

axs[0].set_title("Temperature 2m")
axs[1].set_title("Relative Humdity 2m")
mask = (HAC_met.time >= times.iloc[0]) & (HAC_met.time <= times.iloc[-1])
axs[0].plot(HAC_met.loc[mask, "time"], HAC_met.loc[mask, "temp"] + 273.15, label="HAC", color="k")
axs[1].plot(HAC_met.loc[mask, "time"], HAC_met.loc[mask, "rh"], label="HAC", color="k")
axs[1].legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncols=3)
axs[0].set_ylabel("Temperature [K]")
axs[1].set_ylabel("[%]")
axs[0].set_xticks(tick_locs)
axs[0].set_xticklabels([])
axs[1].set_xticks(tick_locs)
axs[1].set_xticklabels(tick_labels)
axs[1].set_xlabel("Date [UTC]")


plt.tight_layout()
# %%
# Get statistics

metric_data = []
HAC_met_slice = HAC_met.loc[mask]


for id in runs.keys():
    rmse_temp = np.sqrt(np.mean((temp2m[id] - (HAC_met_slice.temp.to_numpy() + 273.15)) ** 2))
    rmse_rh = np.sqrt(np.mean((rh2m[id] - HAC_met_slice.rh.to_numpy()) ** 2))

    metric_data.append({"scheme": runs[id], "RMSE_temp": rmse_temp, "RMSE_rh": rmse_rh})

metrics = pd.DataFrame(metric_data)

# %%
# Plot boundary layer height above NPRK

step = 1
plt.figure()
for id in runs.keys():
    if id in ["THMP", "WDM6"]:
        continue
    plt.plot(times[::step], pblh[id][::step], color=colors[id], label=runs[id], alpha=0.75)
plt.hlines(HGT_diff, times.iloc[0], times.iloc[-1], linestyle="--", color="lightgrey", label="HAC grid cell")
plt.hlines(2314 - HGT_NPRK, times.iloc[0], times.iloc[-1], linestyle="--", color="k", label="HAC Altitude")

lgnd = plt.legend(bbox_to_anchor=(0.5, -0.175), loc="upper center", ncols=2)
plt.title(f"BLH from {times.iloc[0].strftime('%d/%m %H:%M')} to {times.iloc[-1].strftime('%d/%m %H:%M')}")
plt.xticks(tick_locs, labels=tick_labels)
plt.xlabel("Date [Local]")
plt.ylabel("Height Above Ground [m]")
