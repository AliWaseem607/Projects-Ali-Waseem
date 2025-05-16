# %%
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import (
    BASTAMultiDataset,
    BASTAMultiDatasetFactory,
    MIRAMultiDataset,
    MIRAMultiDatasetFactory,
    WRFMultiDataset,
    WRFMultiDatasetFactory,
)


def get_LWC(tk: np.ndarray, qvapor: np.ndarray, pressure: np.ndarray, cloud_water: np.ndarray):
    RA = 287.15
    EPS = 0.622
    tv = tk * (EPS + qvapor) / (EPS * (1.0 + qvapor))
    rho = pressure / RA / tv
    lwc = cloud_water * rho * 10**3
    lwc[lwc < 10**-7] = np.nan
    return lwc


def get_LWC_rh(tk: np.ndarray, rh: np.ndarray, pressure: np.ndarray, cloud_water: np.ndarray):
    RA = 287.15
    EPS = 0.622
    TP_T = 273.16
    TP_P = 611.657  # Pa
    Lv = 2260 * 1000  # J/kg
    Rv = 461.51  # JK−1kg−1
    e_sat = TP_P * np.exp(Lv / Rv * (1 / TP_T - 1 / tk))

    e = rh / 100 * e_sat
    qvapor = RA / Rv * e / (pressure * 100 - e)
    tv = tk * (EPS + qvapor) / (EPS * (1.0 + qvapor))
    rho = pressure / RA / tv
    lwc = cloud_water * rho * 10**3
    lwc[lwc < 10**-7] = np.nan
    return lwc


# %%
# get WRF data
wrf_dataset_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))
MIRA_dataset_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv"))
BASTA_dataset_factory = BASTAMultiDatasetFactory(Path("./data/metadata_BASTA.csv"))
# wrf_dataset_factory.describe_data()
# get BASTA data
# get time specific datasets
wrf_1 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=10, day=12),
    pd.Timestamp(year=2024, month=10, day=15),
    station="NPRK",
    mp_phys=10,
    bl_phys=1,
    sip=True,
)

wrf_2 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=10, day=17),
    pd.Timestamp(year=2024, month=10, day=23),
    station="NPRK",
    mp_phys=10,
    bl_phys=1,
    sip=True,
)

wrf_3 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=10, day=25),
    pd.Timestamp(year=2024, month=10, day=27),
    station="NPRK",
    mp_phys=10,
    bl_phys=1,
    sip=True,
)

wrf_4 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=10, day=28, hour=18),
    pd.Timestamp(year=2024, month=11, day=1, hour=18),
    station="NPRK",
    mp_phys=10,
    bl_phys=1,
    sip=True,
)

wrf_5 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=10, day=18, hour=18),
    pd.Timestamp(year=2024, month=10, day=20, hour=18),
    station="NPRK",
    mp_phys=10,
    bl_phys=5,
    sip=True,
)

wrf_6 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=10, day=28, hour=18),
    pd.Timestamp(year=2024, month=10, day=30, hour=18),
    station="NPRK",
    mp_phys=10,
    bl_phys=5,
    sip=True,
)

wrf_7 = wrf_dataset_factory.get_dataset(
    pd.Timestamp(year=2024, month=11, day=10, hour=0),
    pd.Timestamp(year=2024, month=11, day=24, hour=0),
    station="NPRK",
    mp_phys=10,
    bl_phys=5,
    sip=False,
)

ERA5 = Dataset("/work/lapi/waseem/ERA5/ERA5_ncfiles/20241015-20241115.nc")


# %%
def plot_radar_comparison(
    wrf_dataset: WRFMultiDataset,
    BASTA_dataset: BASTAMultiDataset | None = None,
    MIRA_dataset: MIRAMultiDataset | None = None,
    ERA5_ncfile: Dataset | None = None,
    BASTA_radar_step: int = 25,
    MIRA_radar_step: int = 10,
    plot_time: pd.Timedelta = pd.Timedelta(2, "d"),
    save: bool = False,
):
    ylim= 5
    num_subplots = 0
    if BASTA_dataset is not None:
        num_subplots += 2
    if MIRA_dataset is not None:
        num_subplots += 2
    if ERA5_ncfile is not None:
        num_subplots += 2

    wrf_times = pd.Series(wrf_dataset.getvar("Times"))

    if BASTA_dataset is not None:
        wrf_refl_BASTA = wrf_dataset.variables("Zhh_BASTA")

    if MIRA_dataset is not None:
        wrf_refl_MIRA = wrf_dataset.variables("Zhh_MIRA")

    if ERA5_ncfile is not None:
        # NRPK is at lat: 38.007444, ln: 22.196028
        ERA5_times = pd.to_datetime(
            ERA5_ncfile.variables["valid_time"][:], origin=pd.Timestamp(year=1970, month=1, day=1), unit="s"
        )
        ERA5_lat_idx = np.argmin(np.abs(ERA5.variables["latitude"][:] - 38.007444))
        ERA5_lon_idx = np.argmin(np.abs(ERA5.variables["longitude"][:] - 22.196028))

    assert wrf_dataset.start_time is not None
    assert wrf_dataset.end_time is not None

    total_time = wrf_dataset.end_time - wrf_dataset.start_time

    i = 0
    for i in range(int(np.ceil(total_time / plot_time))):
        plot_start = wrf_dataset.start_time + plot_time * i
        plot_end = np.min([wrf_dataset.end_time, wrf_dataset.start_time + plot_time * (i + 1)])  # type: ignore

        tick_locs = mdates.drange(plot_start, plot_end + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
        tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

        wrf_mask = (wrf_times >= plot_start) & (wrf_times <= plot_end)
        if BASTA_dataset is not None:
            BASTA_mask = (BASTA_dataset.times >= plot_start) & (BASTA_dataset.times <= plot_end)
            uncalib_refl = BASTA_dataset.variables("raw_reflectivity")[BASTA_mask][::BASTA_radar_step]
            BASTA_refl = BASTA_dataset.get_calib_refl(uncalib_refl, BASTA_dataset.range)

        if MIRA_dataset is not None:
            MIRA_mask = (MIRA_dataset.times >= plot_start) & (MIRA_dataset.times <= plot_end)

        fig, ax = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots))
        i = 0
        if BASTA_dataset is not None:
            im0 = ax[i].pcolormesh(
                BASTA_dataset.times[BASTA_mask][::BASTA_radar_step],
                BASTA_dataset.range / 1000,
                BASTA_refl.T,  # type: ignore
                vmin=-35,
                vmax=20,
                cmap="jet",
            )
            cbar = plt.colorbar(im0)
            cbar.set_label("Reflectivity [dBz]")
            ax[i].set_ylim(0, ylim)
            ax[i].set_ylabel("Altitude [km]")
            ax[i].set_title("BASTA")

            cs = ax[i + 1].contour(
                wrf_times[wrf_mask],
                wrf_dataset.ZZ / 1000,
                (wrf_dataset.kinetic_temp[wrf_mask, :].T - 273.15),
                levels=np.arange(-70, 10, 5),
                colors="dimgray",
                linewidths=1,
                alpha=0.5,
            )
            ax[i + 1].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
            im1 = ax[i + 1].pcolormesh(
                wrf_times[wrf_mask],
                wrf_dataset.ZZ / 1000,
                wrf_refl_BASTA[wrf_mask, :].T,
                vmin=-35,
                vmax=20,
                cmap="jet",
            )
            cbar = plt.colorbar(im1)
            cbar.set_label("Reflectivity [dBz]")

            ax[i + 1].set_ylim(0, ylim)
            ax[i + 1].set_ylabel("Altitude [km]")
            ax[i + 1].set_title("WRF Simulation BASTA")

            i += 2

        if MIRA_dataset is not None:
            im0 = ax[i].pcolormesh(
                MIRA_dataset.times[MIRA_mask][::MIRA_radar_step],
                MIRA_dataset.range / 1000,
                MIRA_dataset.refl[MIRA_mask][::MIRA_radar_step, :].T,
                vmin=-60,
                vmax=35,
                cmap="jet",
            )
            cbar = plt.colorbar(im0)
            cbar.set_label("Reflectivity [dBz]")
            ax[i].set_ylim(0, ylim)
            ax[i].set_ylabel("Altitude [km]")
            ax[i].set_title("MIRA")

            cs = ax[i + 1].contour(
                wrf_times[wrf_mask],
                wrf_dataset.ZZ / 1000,
                (wrf_dataset.kinetic_temp[wrf_mask, :].T - 273.15),
                levels=np.arange(-70, 10, 5),
                colors="dimgray",
                linewidths=1,
                alpha=0.5,
            )
            ax[i + 1].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
            im1 = ax[i + 1].pcolormesh(
                wrf_times[wrf_mask],
                wrf_dataset.ZZ / 1000,
                wrf_refl_MIRA[wrf_mask, :].T,
                vmin=-60,
                vmax=35,
                cmap="jet",
            )
            cbar = plt.colorbar(im1)
            cbar.set_label("Reflectivity [dBz]")

            ax[i + 1].set_ylim(0, ylim)
            ax[i + 1].set_ylabel("Altitude [km]")
            ax[i + 1].set_title("WRF Simulation MIRA")

            i += 2

        if ERA5_ncfile is not None:
            ERA5_time_mask = (ERA5_times >= plot_start) & (ERA5_times <= plot_end)
            ERA5_ZZ = ERA5_ncfile.variables["z"][ERA5_time_mask, :, ERA5_lat_idx, ERA5_lon_idx]
            levels = np.linspace(-6, 1, 8)
            ERA5_tk = ERA5_ncfile.variables["t"][ERA5_time_mask, :, ERA5_lat_idx, ERA5_lon_idx]
            ERA5_rh = ERA5_ncfile.variables["t"][ERA5_time_mask, :, ERA5_lat_idx, ERA5_lon_idx]
            ERA5_cloud_water = (
                ERA5_ncfile.variables["clwc"][ERA5_time_mask, :, ERA5_lat_idx, ERA5_lon_idx]
                + ERA5_ncfile.variables["crwc"][ERA5_time_mask, :, ERA5_lat_idx, ERA5_lon_idx]
            )
            ERA5_pressure = ERA5_ncfile.variables["pressure_level"][:]
            ERA5_lwc = get_LWC_rh(ERA5_tk, ERA5_rh, ERA5_pressure, ERA5_cloud_water)
            im2 = ax[2].pcolormesh(
                ERA5_times[ERA5_time_mask], ERA5_ZZ[0, :] / 1000, np.log10(ERA5_lwc.T), cmap="viridis"
            )
            cbar = plt.colorbar(im2)
            cbar.set_label("IWC + LWC [log(g/m3)]")
            ax[2].set_ylim(0, ylim)
            ax[2].set_ylabel("Altitude [km]")
            ax[2].set_title("ERA5 IWC and LWC")
            wrf_total_water = np.nansum(
                np.stack([wrf_dataset.iwc[wrf_mask], wrf_dataset.lwc[wrf_mask]], axis=2), axis=2
            )
            wrf_total_water[wrf_total_water < 10**-5] = np.nan
            im3 = ax[3].pcolormesh(
                wrf_times[wrf_mask], wrf_dataset.ZZ / 1000, np.log10(wrf_total_water).T, cmap="viridis"
            )
            ax[3].set_ylim(0, ylim)
            ax[3].set_ylabel("Altitude [km]")
            ax[3].set_title("WRF IWC and LWC")

            cbar = plt.colorbar(im3)
            cbar.set_label("IWC + LWC [log(g/m3)]")

        set_up_dated_x_axis(ax, tick_locs, tick_labels)

        fig.suptitle(
            f"WRF (bl: {wrf_dataset.bl_phys}) BASTA Comparison {plot_start.strftime(r'%m-%d')} to {plot_end.strftime(r'%m-%d')}"
        )
        plt.tight_layout()
        if save:
            plot_title = "WRF"
            if BASTA_dataset is not None:
                plot_title += "vBASTA"
            if MIRA_dataset is not None:
                plot_title += "vMIRA"
            if ERA5_ncfile is not None:
                plot_title += "vERA5"
            plt.savefig(
                f"./figures/radar_comparison/{plot_title}_bl_{wrf_dataset.bl_phys}_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}_zoomed"
            )


def plot_radar_comparison_factory(
    wrf_dataset: WRFMultiDataset,
    MIRA_factory: MIRAMultiDatasetFactory | None = None,
    BASTA_factory: BASTAMultiDatasetFactory | None = None,
    ERA5_ncfile: Dataset | None = None,
    save: bool = False,
):

    MIRA_dataset = None
    BASTA_dataset = None
    if MIRA_factory is not None:
        MIRA_dataset = MIRA_factory.get_dataset(wrf_dataset.start_time, wrf_dataset.end_time)  # type: ignore
    if BASTA_factory is not None:
        BASTA_dataset = BASTA_factory.get_dataset(wrf_dataset.start_time, wrf_dataset.end_time)  # type: ignore

    plot_radar_comparison(
        wrf_dataset=wrf_dataset,
        BASTA_dataset=BASTA_dataset,
        MIRA_dataset=MIRA_dataset,
        ERA5_ncfile=ERA5_ncfile,
        save=save,
    )


# %%
plot_radar_comparison_factory(
    wrf_dataset=wrf_1,
    BASTA_factory=BASTA_dataset_factory,
)
# %%
plot_radar_comparison_factory(
    wrf_dataset=wrf_2,
    BASTA_factory=BASTA_dataset_factory,
    ERA5_ncfile=ERA5,
)
plot_radar_comparison_factory(
    wrf_dataset=wrf_3,
    BASTA_factory=BASTA_dataset_factory,
    ERA5_ncfile=ERA5,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_4,
    BASTA_factory=BASTA_dataset_factory,
    ERA5_ncfile=ERA5,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_5,
    BASTA_factory=BASTA_dataset_factory,
    ERA5_ncfile=ERA5,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_6,
    BASTA_factory=BASTA_dataset_factory,
    ERA5_ncfile=ERA5,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_2,
    BASTA_factory=BASTA_dataset_factory,
)
plot_radar_comparison_factory(
    wrf_dataset=wrf_3,
    BASTA_factory=BASTA_dataset_factory,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_4,
    BASTA_factory=BASTA_dataset_factory,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_5,
    BASTA_factory=BASTA_dataset_factory,
)

plot_radar_comparison_factory(
    wrf_dataset=wrf_6,
    BASTA_factory=BASTA_dataset_factory,
)
# %%

plot_radar_comparison_factory(
    wrf_dataset=wrf_7,
    MIRA_factory=MIRA_dataset_factory,
    save=True
)
