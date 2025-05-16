# %%
import datetime
import glob
from functools import cached_property
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar


def get_middled_geopotential_height(dataset: Dataset) -> np.ndarray:
    PHB = np.squeeze(dataset.variables["PHB"][0, :])
    PH = np.squeeze(dataset.variables["PH"][0, :])
    HGT = np.squeeze(dataset.variables["HGT"][0])
    ZZ = (PH + PHB) / 9.81 - HGT
    return np.diff(ZZ) / 2 + ZZ[:-1]
# %%
reg = np.load("data/envelopment_period_1/radar/REFL_CHPN_2024-10-17.npy")
reg_times = np.load("data/envelopment_period_1/radar/REFL_times_CHPN_2024-10-17.npy")


at_00 = np.load("/scratch/waseem/radar_testing/crsim_at_00/REFL_at_00.npy")
mira = np.load("/scratch/waseem/radar_testing/crsim_MIRA/REFL_MIRA.npy")
basta = np.load("/scratch/waseem/radar_testing/crsim_BASTA/REFL_BASTA.npy")
varcut = np.load("/scratch/waseem/radar_testing/crsim_VARCUT/REFL_VARCUT.npy")
times = np.load("/scratch/waseem/radar_testing/crsim_at_00/REFL_times_at_00.npy")

wrf = Dataset("/scratch/waseem/radar_testing/wrfout_CHPN_d03_2024-10-17_NPRK.nc")

ZZ = get_middled_geopotential_height(wrf)

reg_mask = reg_times>= times[0]
reg = reg[reg_mask]

# %%

fig, axs = plt.subplots(5, 1, figsize=(9, 12))

axs[0].pcolormesh(times, ZZ / 1000, reg[:, :-1].T, vmin=-35, vmax=20, cmap="jet")
axs[0].set_title("with random ixc, iyc")

axs[1].pcolormesh(times, ZZ / 1000, at_00[:, :-1].T, vmin=-35, vmax=20, cmap="jet")
axs[1].set_title("ixc, iyc, 0,0")

axs[2].pcolormesh(times, ZZ / 1000, mira[:, :-1].T, vmin=-35, vmax=20, cmap="jet")
axs[2].set_title("MIRA set up")

axs[3].pcolormesh(times, ZZ / 1000, basta[:, :-1].T, vmin=-35, vmax=20, cmap="jet")
axs[3].set_title("BASTA set up")

axs[4].pcolormesh(times, ZZ / 1000, varcut[:, :-1].T, vmin=-35, vmax=20, cmap="jet")
axs[4].set_title("BASTA set up")
plt.tight_layout()

