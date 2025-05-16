# %%
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import (
    BASTAMultiDataset,
    BASTAMultiDatasetFactory,
    MIRAMultiDataset,
    MIRAMultiDatasetFactory,
    WRFDataset,
    WRFMultiDataset,
    WRFMultiDatasetFactory,
)

# %%
# get WRF data
MIRA_dataset_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv"))

ncfile1 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov9-12/wrfout_CHPN_d03_2024-11-09_LARGE_SUBGRID.nc"))
NPRK_1 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov9-12/wrfout_CHPN_d03_2024-11-09_NPRK.nc"))

ncfile2 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov11-14/wrfout_CHPN_d03_2024-11-11_LARGE_SUBGRID.nc"))
NPRK_2 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov11-14/wrfout_CHPN_d03_2024-11-11_NPRK.nc"))

ncfile3 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov13-16/wrfout_CHPN_d03_2024-11-13_LARGE_SUBGRID.nc"))
NPRK_3 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov13-16/wrfout_CHPN_d03_2024-11-13_NPRK.nc"))

ncfile4 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov15-18/wrfout_CHPN_d03_2024-11-15_LARGE_SUBGRID.nc"))
NPRK_4 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov15-18/wrfout_CHPN_d03_2024-11-15_NPRK.nc"))

ncfile5 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov17-20/wrfout_CHPN_d03_2024-11-17_LARGE_SUBGRID.nc"))
NPRK_5 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov17-20/wrfout_CHPN_d03_2024-11-17_NPRK.nc"))

ncfile6 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_LARGE_SUBGRID.nc"))
NPRK_6 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_NPRK.nc"))

ncfile7 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_LARGE_SUBGRID.nc"))
NPRK_7 = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_NPRK.nc"))

MIRA = MIRA_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=11, day=10), pd.Timestamp(year=2024, month=11, day=24)
)
# %%

# ncfiles = [ncfile1, ncfile2, ncfile3, ncfile4, ncfile5, ncfile6, ncfile7]
# NPRKs = [NPRK_1, NPRK_2, NPRK_3, NPRK_4, NPRK_5, NPRK_6, NPRK_7]
ncfiles = [ncfile2, ncfile3]
NPRKs = [NPRK_2, NPRK_3]

MIRA_refl = MIRA.refl
for ncfile, NPRK in zip(ncfiles, NPRKs):
    # ZZ_arr = np.zeros((96, 9, 9))
    PHB = ncfile.ncfile.variables["PHB"][0, :, :, :]
    PH = ncfile.ncfile.variables["PH"][0, :, :, :]
    HGT = ncfile.ncfile.variables["HGT"][0, :, :]
    ZZ = (PH + PHB) / 9.81 - HGT
    # Clean up
    del PHB
    del PH
    del HGT

    ZZ = np.diff(ZZ, axis=0) / 2 + ZZ[:-1, :, :]

    fig, ax = plt.subplots(7, 7, figsize=(60, 30), sharex=True, sharey=True)
    times = pd.Series(getvar(ncfile.ncfile, "Times", timeidx=None, meta=False)[288:])
    wrf_mask = (ncfile.times <= times.iloc[-1]) & (ncfile.times >= times.iloc[0])
    MIRA_mask = (MIRA.times <= times.iloc[-1]) & (MIRA.times >= times.iloc[0])

    tick_locs = mdates.drange(times.iloc[0], times.iloc[-1] + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]
    for i in range(7):
        for j in range(7):
            grid_i = i + 1
            grid_j = j + 1
            if grid_i == 4 and grid_j == 4:
                im0 = ax[i, j].pcolormesh(
                    MIRA.times[MIRA_mask][::10],
                    MIRA.range / 1000,
                    MIRA_refl[MIRA_mask, :][::10].T,  # type: ignore
                    vmin=-35,
                    vmax=20,
                    cmap="jet",
                )
                # cbar = plt.colorbar(im0)
                # cbar.set_label("Reflectivity [dBz]")
                continue
            im1 = ax[i, j].pcolormesh(
                ncfile.times[wrf_mask],
                ZZ[:, grid_i, grid_j] / 1000,
                ncfile.ncfile.variables["Zhh_MIRA"][wrf_mask, :, grid_i, grid_j].T,  # type: ignore
                vmin=-35,
                vmax=20,
                cmap="jet",
            )
            ax[i, j].set_ylim(0, 12)
            # cbar = plt.colorbar(im1)
            # cbar.set_label("Reflectivity [dBz]")
