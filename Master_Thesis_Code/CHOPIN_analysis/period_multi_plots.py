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
from utils import get_plot_end_time, get_plot_start_time

# %%


def get_middled_geopotential_height(dataset: Dataset) -> np.ndarray:
    PHB = np.squeeze(dataset.variables["PHB"][0, :])
    PH = np.squeeze(dataset.variables["PH"][0, :])
    HGT = np.squeeze(dataset.variables["HGT"][0])
    ZZ = (PH + PHB) / 9.81 - HGT
    return np.diff(ZZ) / 2 + ZZ[:-1]


def period_multi_plot(
    period_name: str,
    ncfile_path_HAC: Path,
    ncfile_path_NPRK: Path,
    REFL_path: Path,
    REFL_times_path: Path,
    HAC_met_path: Path,
    POI_start: pd.Timestamp | None = None,
    POI_end: pd.Timestamp | None = None,
):
    ncfile_HAC = Dataset(str(ncfile_path_HAC))
    ncfile_NPRK = Dataset(str(ncfile_path_NPRK))
    REFL = np.load(REFL_path)
    REFL_times = pd.Series(np.load(REFL_times_path))
    HAC_met = pd.read_csv(HAC_met_path)
    HAC_met.time = pd.to_datetime(HAC_met.time)
    spin_up_idx = int(24 * 60 / 5)

    temp2m = np.squeeze(ncfile_HAC.variables["T2"][spin_up_idx:])
    rh2m = getvar(ncfile_HAC, "rh2", timeidx=None, meta=False)[spin_up_idx:]  # type: ignore
    pd_times = pd.Series(getvar(ncfile_HAC, "times", timeidx=None, meta=False)[spin_up_idx:])  # type: ignore
    ZZ = get_middled_geopotential_height(ncfile_HAC)
    HGT_NPRK = np.squeeze(ncfile_NPRK.variables["HGT"][0])
    HGT_HAC = np.squeeze(ncfile_HAC.variables["HGT"][0])
    HGT_diff = HGT_HAC - HGT_NPRK
    PBLH = np.squeeze(ncfile_NPRK.variables["PBLH"][spin_up_idx:])

    if POI_start is not None and POI_end is not None:
        mask_ncfile = (pd_times >= POI_start) & (pd_times <= POI_end)

        temp2m = temp2m[mask_ncfile]
        rh2m = rh2m[mask_ncfile]
        PBLH = PBLH[mask_ncfile]
        pd_times = pd_times[mask_ncfile]

    start_time = get_plot_start_time(pd_times.iloc[0])  # type: ignore
    end_time = get_plot_end_time(pd_times.iloc[-1])  # type: ignore
    HAC_mask = (HAC_met.time >= pd_times.iloc[0]) & (HAC_met.time <= pd_times.iloc[-1])
    mask_REFL = (REFL_times >= pd_times.iloc[0]) & (REFL_times < pd_times.iloc[-1])
    REFL = REFL[mask_REFL]
    REFL_times = REFL_times[mask_REFL]

    tick_locs = mdates.drange(start_time, end_time, pd.Timedelta(6, "h"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    im0 = ax[0].pcolormesh(REFL_times, ZZ / 1000, REFL[:, :-1].T, vmin=-35, vmax=20, cmap="jet")
    ax[0].set_xticks(tick_locs)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel("Altitude [km]")
    ax[0].set_ylim(0, 12.5)
    cax = fig.add_axes([0.18, 1.01, 0.7, 0.03])  # type: ignore
    cbar = fig.colorbar(im0, cax=cax, location="top", fraction=0.10)
    cbar.set_label("Ze [dBZ]")

    ax[1].plot(pd_times, temp2m, label="WRF", color="r")
    ax[1].plot(HAC_met.loc[HAC_mask, "time"], HAC_met.loc[HAC_mask, "temp"] + 273.15, label="HAC", color="k")  # type: ignore
    ax[1].set_ylabel("Temperature [K]")
    ax[1].legend()

    ax[2].plot(pd_times, rh2m, label="WRF", color="r")
    ax[2].plot(HAC_met.loc[HAC_mask, "time"], HAC_met.loc[HAC_mask, "rh"], label="HAC", color="k")
    ax[2].set_ylabel("Relative Humidity [%]")
    ax[2].legend()

    step = 1
    ax[3].plot(pd_times[::step], PBLH[::step], label="PBLH")
    ax[3].axhline(2314 - HGT_NPRK, 0, 1, linestyle="--", color="k", label="HAC Altitude")
    ax[3].axhline(HGT_diff, 0, 1, linestyle="--", color="lightgrey", label="HAC WRF Grid cell Altitude")
    ax[3].legend()
    ax[3].set_ylabel("Height above NPRK [m]")

    ax[0].set_xticks(tick_locs)
    ax[0].set_xticklabels([])
    ax[0].set_yticks(np.linspace(0, 12, 7))
    ax[1].set_xticks(tick_locs)
    ax[1].set_xticklabels([])
    ax[2].set_xticks(tick_locs)
    ax[2].set_xticklabels([])
    ax[3].set_xticks(tick_locs)
    ax[3].set_xticklabels(tick_labels)
    ax[3].set_xlabel("Date [UTC]")
    plt.tight_layout()

    fig.suptitle(
        f"{period_name} {pd_times.iloc[0].strftime(r'%Y-%m-%d %H:%M')} to {pd_times.iloc[-1].strftime(r'%Y-%m-%d %H:%M')}",
        y=0.975,
    )


# period_multi_plot(
#     period_name="Envelopment Period",
#     ncfile_path_HAC=Path("data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_HAC.nc"),
#     ncfile_path_NPRK=Path("data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK.nc"),
#     REFL_path=Path("data/envelopment_period_1/radar/REFL_CHPN_2024-10-17.npy"),
#     REFL_times_path=Path("data/envelopment_period_1/radar/REFL_times_CHPN_2024-10-17.npy"),
#     HAC_met_path=Path("data/HAC_meteo.csv"),
#     POI_start=pd.Timestamp(year=2024, month=10, day=19, hour=0),
#     POI_end=pd.Timestamp(year=2024, month=10, day=20, hour=18),
# )

period_multi_plot(
    period_name="Envelopment Period ALL",
    ncfile_path_HAC=Path("data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_HAC.nc"),
    ncfile_path_NPRK=Path("data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK.nc"),
    REFL_path=Path("data/envelopment_period_1/radar/REFL_CHPN_2024-10-17.npy"),
    REFL_times_path=Path("data/envelopment_period_1/radar/REFL_times_CHPN_2024-10-17.npy"),
    HAC_met_path=Path("data/HAC_meteo.csv"),
)

# period_multi_plot(
#     period_name="Rain Period",
#     ncfile_path_HAC=Path("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_YSU.nc"),
#     ncfile_path_NPRK=Path("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_YSU.nc"),
#     REFL_path=Path("data/rain_period_1/radar/REFL_CHPN_2024-10-06_YSU.npy"),
#     REFL_times_path=Path("data/rain_period_1/radar/REFL_times_CHPN_2024-10-06_YSU.npy"),
#     HAC_met_path=Path("data/HAC_meteo.csv"),
#     POI_start=pd.Timestamp(year=2024, month=10, day=7, hour=3),
#     POI_end=pd.Timestamp(year=2024, month=10, day=7, hour=6),
# )

# period_multi_plot(
#     period_name="Rain Period ALL",
#     ncfile_path_HAC=Path("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_YSU.nc"),
#     ncfile_path_NPRK=Path("data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_YSU.nc"),
#     REFL_path=Path("data/rain_period_1/radar/REFL_CHPN_2024-10-06_YSU.npy"),
#     REFL_times_path=Path("data/rain_period_1/radar/REFL_times_CHPN_2024-10-06_YSU.npy"),
#     HAC_met_path=Path("data/HAC_meteo.csv"),
# )

# period_multi_plot(
#     period_name="Clear Period",
#     ncfile_path_HAC=Path("data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27_HAC.nc"),
#     ncfile_path_NPRK=Path("data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27_NPRK.nc"),
#     REFL_path=Path("data/clear_period_1/radar/REFL_CHPN_2024-10-27.npy"),
#     REFL_times_path=Path("data/clear_period_1/radar/REFL_times_CHPN_2024-10-27.npy"),
#     HAC_met_path=Path("data/HAC_meteo.csv"),
# )
