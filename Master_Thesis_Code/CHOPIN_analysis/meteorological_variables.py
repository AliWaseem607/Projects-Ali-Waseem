# %%

import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar

sys.path.append("./")
from utils import (
    get_middled_geopotential_height,
    get_plot_end_time,
    get_plot_start_time,
)

# %%
# Load data for comparison of all times available


data_times = pd.read_csv("data/data_times.csv")
file_ids = list(data_times.file_id)
HAC_met = pd.read_csv("data/insitu_measurements/HAC_meteo.csv")
HAC_met.time = pd.to_datetime(HAC_met.time)
sorted_data_times = data_times.copy()
sorted_data_times["start_time"] = pd.to_datetime(sorted_data_times.start_time)
sorted_data_times["end_time"] = pd.to_datetime(sorted_data_times.end_time)
sorted_data_times = sorted_data_times.sort_values("start_time")
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
# Plot data for comparison to all run simulations

fig, ax = plt.subplots(3, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 3, 1]})
min_time = None
max_time = None
for id in file_ids:
    wrf_data = Dataset(f"data/wrfout/wrfout_CHPN_d03_{id}_HAC.nc")

    spin_up_idx = int(24 * 60 / 5)
    times = getvar(wrf_data, "Times", timeidx=None, meta=False)[spin_up_idx:]

    pd_times = pd.Series(times)  # type: ignore

    start_time = get_plot_start_time(pd_times.iloc[0])  # type: ignore
    end_time = get_plot_end_time(pd_times.iloc[-1])  # type: ignore

    rh_wrfpy = getvar(wrf_data, "rh2", meta=False, timeidx=None)[spin_up_idx:]

    t2 = np.squeeze(wrf_data.variables["T2"][spin_up_idx:])
    if min_time is None:
        ax[0].plot(pd_times, rh_wrfpy, color="r", label="WRF")
    else:
        ax[0].plot(pd_times, rh_wrfpy, color="r")
    ax[1].plot(pd_times, t2 - 273.15, color="r")

    if min_time is None:
        min_time = pd_times.iloc[0]
    if max_time is None:
        max_time = pd_times.iloc[-1]

    if min_time > pd_times.iloc[0]:
        min_time = pd_times.iloc[0]

    if max_time < pd_times.iloc[-1]:
        max_time = pd_times.iloc[-1]

mask = (HAC_met.time >= min_time) & (HAC_met.time <= max_time)

# # Add fill for periods if wanted
# rain_period = [
#     pd.Timestamp(year=2024, month=10, day=7, hour=3),
#     pd.Timestamp(year=2024, month=10, day=7, hour=6),
# ]
# envelopement_period = [
#     pd.Timestamp(year=2024, month=10, day=19, hour=0),
#     pd.Timestamp(year=2024, month=10, day=20, hour=18),
# ]
# ax[0].axvspan(rain_period[0], rain_period[1], color="blue", alpha=0.15)
# ax[1].axvspan(rain_period[0], rain_period[1], color="blue", alpha=0.15)
# ax[0].axvspan(envelopement_period[0], envelopement_period[1], color="green", alpha=0.15)
# ax[1].axvspan(envelopement_period[0], envelopement_period[1], color="green", alpha=0.15)

ax[0].plot(HAC_met.loc[mask, "time"], HAC_met.loc[mask, "rh"], color="b", label="HAC")
ax[1].plot(HAC_met.loc[mask, "time"], HAC_met.loc[mask, "temp"], color="b")
tick_locs = mdates.drange(min_time, max_time, pd.Timedelta(2, "d"))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

fig.supxlabel("Date [UTC]")
ax[0].legend()
ax[0].set_ylabel("2m Relative Humidity [%]")
ax[1].set_ylabel("2m temperature [C]")
ax[0].set_xticks(tick_locs)
ax[0].set_xticklabels([])
ax[1].set_xticks(tick_locs)
ax[1].set_xticklabels([])

for i, row in sorted_data_times.iterrows():
    ax[2].plot(
        [
            row.start_time + pd.Timedelta(row.spinup, "h") + pd.Timedelta(2, "h"),
            row.end_time + pd.Timedelta(2, "h"),
        ],
        [row.plot_indx, row.plot_indx],
        linewidth=3,
    )

ax[2].set_ylim((-0.5, sorted_data_times.plot_indx.max() + 0.5))
ax[2].set_title("WRF data availability")
ax[2].set_yticks([])
ax[2].set_xticks(tick_locs)
ax[2].set_xticklabels(tick_labels)
ax[0].set_title("Meterological Data Comparison at HAC")
# ax[0].set_xlim(min_time - pd.Timedelta(1, "D"), pd.Timestamp(year=2024, month=10, day=21, hour=12))
# ax[1].set_xlim(min_time - pd.Timedelta(1, "D"), pd.Timestamp(year=2024, month=10, day=21, hour=12))
# ax[2].set_xlim(min_time - pd.Timedelta(1, "D"), pd.Timestamp(year=2024, month=10, day=21, hour=12))


# %%
# Gather data for comparison amongst rain periods
spin_up_idx = int(24 * 60 / 5)
HAC_met = pd.read_csv("data/insitu_measurements/HAC_meteo.csv")
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
times = None
for id in runs.keys():
    ncfile = Dataset(f"data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_{id}.nc")
    temp2m[id] = np.squeeze(ncfile.variables["T2"][spin_up_idx:])
    rh2m[id] = getvar(ncfile, "rh2", timeidx=None, meta=False)[spin_up_idx:]
    if times is None:
        times = pd.Series(getvar(ncfile, "Times", timeidx=None, meta=False))[spin_up_idx:]


tick_locs = mdates.drange(times.iloc[0], times.iloc[-1], pd.Timedelta(6, "h"))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

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
    rmse_temp = np.sqrt(np.mean((temp2m[id] - HAC_met_slice.temp.to_numpy()) ** 2))
    rmse_rh = np.sqrt(np.mean((rh2m[id] - HAC_met_slice.rh.to_numpy()) ** 2))

    metric_data.append({"scheme": runs[id], "RMSE_temp": rmse_temp, "RMSE_rh": rmse_rh})


# %%
# Plot the envelopement radar comparison
envelop_bounds = (
    pd.Timestamp(year=2024, month=10, day=18, hour=12),
    pd.Timestamp(year=2024, month=10, day=20, hour=18),
)
tick_locs = mdates.drange(envelop_bounds[0], envelop_bounds[1] + pd.Timedelta(6, "h"), pd.Timedelta(6, "h"))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

fig, ax = plt.subplots(4, 1, figsize=(10, 8))
BASTA = Dataset("data/envelopment_period_1/insitu_radar/BASTA_combined.nc")
basta_times_all = pd.to_datetime(
    BASTA.variables["time"][:], unit="s", origin=pd.Timestamp(year=2024, month=10, day=17)
)
basta_range = BASTA.variables["range"][:]
basta_mask = (basta_times_all >= envelop_bounds[0]) & (basta_times_all <= envelop_bounds[1])
basta_times = basta_times_all[basta_mask]
basta_refl = BASTA.variables["raw_reflectivity"][basta_mask, :]
calib_db = -186
basta_cal_refl = (
    basta_refl + 20.0 * np.log10(np.tile(basta_range, (len(basta_times), 1)), dtype="f") + calib_db
)

im0 = ax[0].pcolormesh(
    basta_times[::25], basta_range / 1000, basta_cal_refl[::25, :].T, cmap="jet", vmin=-35, vmax=20
)
cbar = plt.colorbar(im0)
cbar.set_label("Reflectivity [dBz]")
ax[0].set_title("BASTA")

i = 1
for key, val in envelop_runs.items():
    ncfile = Dataset(f"./data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK_{key}.nc")
    wrf_times = pd.Series(getvar(ncfile, "Times", meta=False, timeidx=None))
    wrf_mask = (wrf_times >= envelop_bounds[0]) & (wrf_times <= envelop_bounds[1])
    im = ax[i].pcolormesh(
        wrf_times[wrf_mask],
        get_middled_geopotential_height(ncfile) / 1000,
        np.squeeze(ncfile.variables["Zhh_BASTA"])[wrf_mask, :].T,
        vmin=-35,
        vmax=20,
        cmap="jet",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Reflectivity [dBz]")
    ax[i].set_title(key)

    i += 1

for a in ax:
    a.set_xticks(tick_locs)
    a.set_xticklabels([])
    a.set_ylim(0, 12)
    a.set_ylabel("Height [km]")
ax[-1].set_xticklabels(tick_labels)
ax[-1].set_xlabel("Date [UTC]")

plt.tight_layout()
