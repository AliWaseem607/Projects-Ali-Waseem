# %%
import datetime
import glob
import sys
from functools import cached_property
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from netCDF4 import Dataset  # type: ignore
from wrf import getvar

sys.path.append("./")
from utils import get_middled_geopotential_height

# %%
# Get Data
times = np.load("data/rain_period_1/radar/REFL_times_CHPN_2024-10-06_MYJ.npy")
ysu_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_YSU.npy")
keps_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_KEPS.npy")
myj_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_MYJ.npy")
thmp_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_THMP.npy")
wdm6_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_WDM6.npy")
sh_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_SH.npy")
myj25_REFL = np.load("data/rain_period_1/radar/REFL_CHPN_2024-10-06_MYJ25.npy")


ysu_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_YSU.nc")
keps_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_KEPS.nc")
myj_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_MYJ.nc")
thmp_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_THMP.nc")
wdm6_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_WDM6.nc")
sh_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_SH.nc")
myj25_wrf = Dataset("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_MYJ25.nc")


# %%
# Calculate Geopotental height


ysu_ZZ = get_middled_geopotential_height(ysu_wrf)
keps_ZZ = get_middled_geopotential_height(keps_wrf)
myj_ZZ = get_middled_geopotential_height(myj_wrf)
thmp_ZZ = get_middled_geopotential_height(thmp_wrf)
wdm6_ZZ = get_middled_geopotential_height(wdm6_wrf)
sh_ZZ = get_middled_geopotential_height(sh_wrf)
myj25_ZZ = get_middled_geopotential_height(myj25_wrf)


# %%
# Plot reflectivities of rain period
period_of_interest_start = np.datetime64("2024-10-07T03:00")
period_of_interest_end = np.datetime64("2024-10-07T06:00")
spinup = np.datetime64("2024-10-07T00:00")
cmap = get_cmap("plasma", 24)

start_time = datetime.datetime(2024, 10, 7)  # Start time of the timeseries
end_time = datetime.datetime(2024, 10, 9)  # End time of the timeseries

tick_locs = mdates.drange(start_time, end_time, datetime.timedelta(hours=6))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

# mask = (times >= period_of_interest_start) & (times <= period_of_interest_end)
mask = times >= spinup

MIRA_start = pd.Timestamp(year=2024, month=10, day=7, hour=3)
MIRA_cutout = pd.Timestamp(year=2024, month=10, day=7, hour=6)
fig, axs = plt.subplots(7, 1, figsize=(9, 12))

axs[0].pcolormesh(times, ysu_ZZ / 1000, ysu_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[0].set_xticks(tick_locs)
axs[0].set_xticklabels([])
axs[0].set_title("YSU")

axs[1].pcolormesh(times, myj_ZZ / 1000, myj_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[1].set_xticks(tick_locs)
axs[1].set_xticklabels([])
axs[1].set_title("MYJ")

axs[2].pcolormesh(times, keps_ZZ / 1000, keps_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[2].set_xticks(tick_locs)
axs[2].set_xticklabels([])
axs[2].set_title("KEPS")

axs[3].pcolormesh(times, sh_ZZ / 1000, sh_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[3].set_xticks(tick_locs)
axs[3].set_xticklabels([])
axs[3].set_title("SH")

axs[4].pcolormesh(times, myj25_ZZ / 1000, myj25_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[4].set_xticks(tick_locs)
axs[4].set_xticklabels([])
axs[4].set_title("MYJ25")

axs[5].pcolormesh(times, wdm6_ZZ / 1000, wdm6_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[5].set_xticks(tick_locs)
axs[5].set_xticklabels([])
axs[5].set_title("WDM6")

axs[6].pcolormesh(times, thmp_ZZ / 1000, thmp_REFL[:, :-1].T, vmin=-35, vmax=20, cmap=cmap)
axs[6].set_xticks(tick_locs)
axs[6].set_xticklabels([])
axs[6].set_title("THMP")

axs[-1].set_xticks(tick_locs)
axs[-1].set_xticklabels(tick_labels)
axs[-1].set_xlabel("Dates [UTC]")
fig.supylabel("Altitude [km]")
for ax in axs:
    ax.axvline(MIRA_start, 0, 17, color="k", linestyle="--")
    ax.axvline(MIRA_cutout, 0, 17, color="k", linestyle="--")
plt.tight_layout()

# %%
# Plot just YSU
mask = (times >= period_of_interest_start) & (times <= period_of_interest_end)
plt.figure(figsize=(9, 4))
im0 = plt.pcolormesh(times[mask], ysu_ZZ / 1000, ysu_REFL[mask, :-1].T, vmin=-60, vmax=30, cmap="viridis")
plt.ylim(0, 13)
plt.xlabel("Date [UTC]")
plt.ylabel("Altitude [km]")
plt.colorbar(im0)
# %%
# get RH
spin_up_idx = int(24 * 60 / 5)
pd_times = pd.Series(getvar(ysu_wrf, "Times", meta=False, timeidx=None)[spin_up_idx:])  # type: ignore
ysu_rh_wrfpy = np.zeros((len(pd_times), 96))
myj_rh_wrfpy = np.zeros((len(pd_times), 96))
keps_rh_wrfpy = np.zeros((len(pd_times), 96))


j = 0
for i in range(spin_up_idx, len(pd_times) + spin_up_idx):
    ysu_rh_wrfpy[j, :] = getvar(ysu_wrf, "rh", meta=False, timeidx=i)
    myj_rh_wrfpy[j, :] = getvar(myj_wrf, "rh", meta=False, timeidx=i)
    keps_rh_wrfpy[j, :] = getvar(keps_wrf, "rh", meta=False, timeidx=i)
    j += 1

# %%
# Check for clouds with prognosticss
spinup = np.datetime64("2024-10-07T00:00")
cmap = get_cmap("plasma")

start_time = datetime.datetime(2024, 10, 7)  # Start time of the timeseries
end_time = datetime.datetime(2024, 10, 9)  # End time of the timeseries

tick_locs = mdates.drange(start_time, end_time, datetime.timedelta(hours=6))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

mask = times >= spinup

fig, axs = plt.subplots(3, 1, figsize=(9, 9))

var = "QVAPOR"

im0 = axs[0].pcolormesh(times[mask], ysu_ZZ / 1000, ysu_rh_wrfpy.T, vmin=0, vmax=100, cmap=cmap)
axs[0].set_xticks(tick_locs)
axs[0].set_xticklabels([])

axs[1].pcolormesh(times[mask], myj_ZZ / 1000, myj_rh_wrfpy.T, vmin=0, vmax=100, cmap=cmap)
axs[1].set_xticks(tick_locs)
axs[1].set_xticklabels([])

axs[2].pcolormesh(times[mask], keps_ZZ / 1000, keps_rh_wrfpy.T, vmin=0, vmax=100, cmap=cmap)
axs[2].set_xticks(tick_locs)
axs[2].set_xticklabels([])

axs[-1].set_xticks(tick_locs)
axs[-1].set_xticklabels(tick_labels)

cax = fig.add_axes([0.92, 0.10, 0.02, 0.85])
cbar = plt.colorbar(im0, cax=cax, orientation="vertical")

# %%

data_times = pd.read_csv("data/data_times.csv")
file_ids = list(data_times.file_id)

# wrf_data={}
# radar_data={}
# radar_times={}

for id in file_ids:
    wrf_data = Dataset(f"data/wrfout/wrfout_CHPN_d03_{id}_NPRK.nc")
    radar_data = np.load(f"data/radar/REFL_CHPN_{id}.npy")
    radar_times = np.load(f"data/radar/REFL_times_CHPN_{id}.npy")

    spin_up_idx = int(24 * 60 / 5)
    times = getvar(wrf_data, "Times", timeidx=None, meta=False)[spin_up_idx:]
    assert isinstance(times, np.ndarray)

    spin_up_time = times[0]
    mask = radar_times >= spin_up_time

    ZZ = get_middled_geopotential_height(wrf_data)
    fig, axs = plt.subplots(1, 1, figsize=(9, 3))

    im0 = axs.pcolormesh(
        radar_times[mask], ZZ / 1000, radar_data[mask, :-1].T, vmin=-35, vmax=20, cmap="plasma"
    )
    wrf_times = np.datetime_as_string(radar_times, "m")
    plt.title(f"{wrf_times[mask][0]} to {wrf_times[mask][-1]}")
    cax = fig.add_axes([0.92, 0.10, 0.02, 0.85])
    cbar = plt.colorbar(im0, cax=cax, orientation="vertical")


# %%
envel_wrf = Dataset("data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK.nc")
envel_REFL = np.load("data/envelopment_period_1/radar/REFL_CHPN_2024-10-17.npy")
envel_REFL_times = np.load("data/envelopment_period_1/radar/REFL_times_CHPN_2024-10-17.npy")

clear_wrf = Dataset("data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27_NPRK.nc")
clear_REFL = np.load("data/clear_period_1/radar/REFL_CHPN_2024-10-27.npy")
clear_REFL_times = np.load("data/clear_period_1/radar/REFL_times_CHPN_2024-10-27.npy")

envel_ZZ = get_middled_geopotential_height(envel_wrf)
clear_ZZ = get_middled_geopotential_height(clear_wrf)


# start_time = datetime.datetime(2024, 10, )  # Start time of the timeseries
# end_time = datetime.datetime(2024, 10, 9)  # End time of the timeseries

# tick_locs = mdates.drange(start_time, end_time, datetime.timedelta(hours=6))
# tick_labels = [mdates.num2date(t).strftime('%d/%m'+'\n'+ '%H:%M') for t in tick_locs]

# mask = (times >= period_of_interest_start) & (times <= period_of_interest_end)
# mask = times >= spinup

fig, axs = plt.subplots(2, 1, figsize=(9, 6))
# cmap = get_cmap("plasma",24)

axs[0].pcolormesh(envel_REFL_times, envel_ZZ / 1000, envel_REFL[:, :-1].T, vmin=-35, vmax=20, cmap="plasma")
axs[0].clabel(cs, inline=True, fontsize=12, fmt="%d$^\circ$C", colors="dimgrey")
axs[0].set_title("Envelopment")

axs[1].pcolormesh(clear_REFL_times, clear_ZZ / 1000, clear_REFL[:, :-1].T, vmin=-35, vmax=20, cmap="plasma")
axs[1].set_title("Clear")

# class WRFResults:
#     def __init__(self, ncfile_path:Path, reflectivity_path:Path, reflectivity_times_path:Path):
#         self.ncfile_path = data_path
#         self.REFL_path = reflectivity_path,
#         self.REFL_times_path = reflectivity_times_path

#     @cached_property
#     def
# %%
# Plot envelopment comparison with real radar data
REFL_all = np.load("data/envelopment_period_1/radar/REFL_CHPN_2024-10-17.npy")
REFL_times = np.load("data/envelopment_period_1/radar/REFL_times_CHPN_2024-10-17.npy")

REFL_ext_all = np.load("data/radar/REFL_CHPN_2024-10-16_BASTA.npy")
REFL_ext_times = np.load("data/radar/REFL_times_CHPN_2024-10-16_BASTA.npy")

MIRA = Dataset("data/envelopment_period_1/insitu_radar/MIRA_combined.znc")
BASTA = Dataset("data/envelopment_period_1/insitu_radar/BASTA_combined.nc")

wrf = Dataset("data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK_YSU.nc")

spinup = int(24 * 60 / 5)
times = pd.Series(getvar(wrf, "Times", timeidx=None, meta=False)[spinup:])  # type: ignore
mira_times_all = pd.to_datetime(MIRA.variables["time"][:], unit="s")
basta_times_all = pd.to_datetime(
    BASTA.variables["time"][:], unit="s", origin=pd.Timestamp(year=2024, month=10, day=17)
)

ZZ = get_middled_geopotential_height(wrf)
mira_range = MIRA.variables["range"][:]
basta_range = BASTA.variables["range"][:]

mira_elv_mask = MIRA.variables["elv"][:] > 80

refl_mask = (REFL_times >= times.iloc[0]) & (REFL_times <= times.iloc[-1])
mira_mask = (mira_times_all >= times.iloc[0]) & (mira_times_all <= times.iloc[-1]) & mira_elv_mask
basta_mask = (basta_times_all >= times.iloc[0]) & (basta_times_all <= times.iloc[-1])
ext_mask = (REFL_ext_times >= times.iloc[0]) & (REFL_ext_times <= times.iloc[-1])

mira_times = mira_times_all[mira_mask]
basta_times = basta_times_all[basta_mask]

mira_refl = MIRA.variables["Zcx"][mira_mask, :]
basta_refl = BASTA.variables["raw_reflectivity"][basta_mask, :]
calib_db = -186
basta_cal_refl = (
    basta_refl + 20.0 * np.log10(np.tile(basta_range, (len(basta_times), 1)), dtype="f") + calib_db
)
REFL = REFL_all[refl_mask, :]
REFL_ext = REFL_ext_all[ext_mask, :]

# %%
print(np.nanmin(mira_refl))
print(np.nanmin(basta_refl))
print(np.nanmin(REFL))

print(np.nanmax(mira_refl))
print(np.nanmax(basta_refl))
print(np.nanmax(REFL))

# %%
tick_locs = mdates.drange(times.iloc[0], times.iloc[-1], datetime.timedelta(hours=6))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]
fig, axs = plt.subplots(3, 1, figsize=(9, 4))

# im0 = axs[0].pcolormesh(
#     basta_times[::25], basta_range / 1000, basta_cal_refl[::25, :].T, cmap="jet", vmin=-50, vmax=30
# )

im0 = axs[0].pcolormesh(
    mira_times[::25], mira_range / 1000, mira_refl[::25, :].T, cmap="jet", vmin=-50, vmax=30
)
im1 = axs[1].pcolormesh(times, ZZ / 1000, REFL[:, :-1].T, cmap="jet", vmin=-35, vmax=20)
im2 = axs[2].pcolormesh(times[:-1], ZZ / 1000, REFL_ext[:, :-1].T, cmap="jet", vmin=-35, vmax=20)

cbar = plt.colorbar(im0)
cbar.set_label("Reflectivity [dBz]")
cbar = plt.colorbar(im1)
cbar.set_label("Reflectivity [dBz]")
cbar = plt.colorbar(im2)
cbar.set_label("Reflectivity [dBz]")

for ax in axs:
    ax.set_ylim(0, 12)
    ax.set_xticklabels([])
    ax.set_xticks(tick_locs)
    ax.set_ylabel("Altitude [km]")

axs[-1].set_xticklabels(tick_labels)
axs[0].set_title("BASTA radar")
axs[1].set_title("WRF simulation")
axs[-1].set_xlabel("Date [UTC]")


# %%
