# %%
import sys
from pathlib import Path

import cmasher as cmr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import (
    MIRAMultiDataset,
    MIRAMultiDatasetFactory,
    WRFDataset,
    WRFMultiDataset,
    WRFMultiDatasetFactory,
)

#%%
MIRA_dataset_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv"))
MIRA_dataset = MIRA_dataset_factory.get_dataset(pd.Timestamp(year=2024, month=11, day=9), pd.Timestamp(year=2024, month=12, day=7))

NOV_28_CTRL_NPRK = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_CONTROL_NPRK.nc"))
NOV_28_SIP_NPRK = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_SIP_NPRK.nc"))
# NOV_28_CTRL_HAC = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_CONTROL_HAC.nc"))
# NOV_28_SIP_HAC = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_SIP_HAC.nc"))

NOV_30_CTRL_NPRK = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_CONTROL_NPRK.nc"))
NOV_30_SIP_NPRK = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_SIP_NPRK.nc"))
# NOV_30_CTRL_HAC = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_CONTROL_HAC.nc"))
# NOV_30_SIP_HAC = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_SIP_HAC.nc"))

DEC_3_CTRL_NPRK = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-03_CONTROL_NPRK.nc"))
DEC_3_SIP_NPRK = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-03_SIP_NPRK.nc"))
# DEC_3_CTRL_HAC = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-03_CONTROL_HAC.nc"))
# DEC_3_SIP_HAC = WRFDataset(Path("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-03_SIP_HAC.nc"))

POI_1_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_CONTROL/wrfout_CHPN_d03_2024-11-10_NPRK.nc"))
POI_1_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_SIP/wrfout_CHPN_d03_2024-11-10_NPRK.nc"))
# POI_1_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_CONTROL/wrfout_CHPN_d03_2024-11-10_HAC.nc"))
# POI_1_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_SIP/wrfout_CHPN_d03_2024-11-10_HAC.nc"))

POI_3_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_NPRK.nc"))
POI_3_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_NPRK.nc"))
# POI_3_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_HAC.nc"))
# POI_3_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_HAC.nc"))

NOV_19_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_NPRK.nc"))
NOV_19_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22_SIP/wrfout_CHPN_d03_2024-11-19_NPRK.nc"))
# NOV_19_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_HAC.nc"))
# NOV_19_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22_SIP/wrfout_CHPN_d03_2024-11-19_HAC.nc"))

NOV_21_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_NPRK.nc"))
NOV_21_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24_SIP/wrfout_CHPN_d03_2024-11-21_NPRK.nc"))
# NOV_21_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_HAC.nc"))
# NOV_21_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24_SIP/wrfout_CHPN_d03_2024-11-21_HAC.nc"))


#%%


def plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL: WRFDataset,
    wrf_dataset_SIP: WRFDataset,
    MIRA_dataset: MIRAMultiDataset,
    MIRA_radar_step: int = 10,
    save: bool = False,
    ylim: float = 12
):
    wrf_times = pd.Series(wrf_dataset_CONTROL.getvar("Times"))

    wrf_refl_CONTROL = wrf_dataset_CONTROL.variables("Zhh_MIRA")
    wrf_refl_SIP = wrf_dataset_SIP.variables("Zhh_MIRA")

    assert wrf_dataset_CONTROL.end_time is not None
    assert wrf_dataset_CONTROL.start_time is not None
    assert wrf_dataset_SIP.end_time is not None
    assert wrf_dataset_SIP.start_time is not None


    plot_start = wrf_dataset_CONTROL.start_time
    plot_end = wrf_dataset_CONTROL.end_time

    tick_locs = mdates.drange(plot_start, plot_end + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

    MIRA_mask = (MIRA_dataset.times >= plot_start) & (MIRA_dataset.times <= plot_end)

    fig, ax = plt.subplots(3, 2, figsize=(16, 8))

    im0 = ax[0,0].pcolormesh(
        MIRA_dataset.times[MIRA_mask][::MIRA_radar_step],
        MIRA_dataset.range / 1000,
        MIRA_dataset.refl[MIRA_mask][::MIRA_radar_step, :].T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im0)
    cbar.set_label("Reflectivity [dBz]")
    ax[0,0].set_ylim(0, ylim)
    ax[0,0].set_ylabel("Altitude [km]")
    ax[0,0].set_title("MIRA")

    cs = ax[1,0].contour(
        wrf_dataset_CONTROL.times,
        wrf_dataset_CONTROL.ZZ / 1000,
        (wrf_dataset_CONTROL.kinetic_temp.T - 273.15),
        levels=np.arange(-70, 10, 5),
        colors="dimgray",
        linewidths=1,
        alpha=0.5,
    )
    ax[1,0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
    im1 = ax[1,0].pcolormesh(
        wrf_dataset_CONTROL.times,
        wrf_dataset_CONTROL.ZZ / 1000,
        wrf_refl_CONTROL.T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im1)
    cbar.set_label("Reflectivity [dBz]")

    ax[1,0].set_ylim(0, ylim)
    ax[1,0].set_ylabel("Altitude [km]")
    ax[1,0].set_title("CONTROL")

    cs = ax[2,0].contour(
        wrf_dataset_SIP.times,
        wrf_dataset_SIP.ZZ / 1000,
        (wrf_dataset_SIP.kinetic_temp.T - 273.15),
        levels=np.arange(-70, 10, 5),
        colors="dimgray",
        linewidths=1,
        alpha=0.5,
    )
    ax[2,0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
    im2 = ax[2,0].pcolormesh(
        wrf_dataset_SIP.times,
        wrf_dataset_SIP.ZZ / 1000,
        wrf_refl_SIP.T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im2)
    cbar.set_label("Reflectivity [dBz]")

    ax[2,0].set_ylim(0, ylim)
    ax[2,0].set_ylabel("Altitude [km]")
    ax[2,0].set_title("SIP")

    # downsample MIRA
    # make a new mask so that we can have data always at the end
    new_MIRA_mask = (MIRA_dataset.times >= plot_start-pd.Timedelta(5, "min")) & (MIRA_dataset.times <= plot_end)
    MIRA_resample = pd.DataFrame({"lwp":MIRA_dataset.lwp[new_MIRA_mask], "times":MIRA_dataset.times[new_MIRA_mask]})
    MIRA_resample = MIRA_resample.resample(on="times", rule=pd.Timedelta(5, "min"), origin=plot_start, label="right").mean().reset_index(drop=False)
    
    # ax[0,1].plot(MIRA_dataset.times[MIRA_mask], MIRA_dataset.lwp[MIRA_mask], color="k", label="MIRA", alpha=1)
    ax[0,1].plot(MIRA_resample.times, MIRA_resample.lwp, color="k", label="MIRA", alpha=1)
    ax[0,1].plot(wrf_dataset_CONTROL.times, wrf_dataset_CONTROL.lwp, color="cyan", label="Control", alpha=0.65)
    ax[0,1].plot(wrf_dataset_SIP.times, wrf_dataset_SIP.lwp, color="b", label="SIP", alpha=0.65)
    ax[0,1].set_ylabel("LWP [g/m3]")
    ax[0,1].legend()
    cbar = fig.colorbar(cs, ax=ax[0,1])
    cbar.ax.set_visible(False)

    cmap2 = cmr.get_sub_cmap('Blues', 0.15, 1.0)
    lwc = ax[1,1].contourf(wrf_dataset_CONTROL.times, wrf_dataset_CONTROL.ZZ/1000, wrf_dataset_CONTROL.lwc.T, levels=np.linspace(0, 1, 9), cmap = cmap2)
    ax[1,1].contour(wrf_dataset_CONTROL.times, wrf_dataset_CONTROL.ZZ/1000, wrf_dataset_CONTROL.rimming.T, levels=[1e-5], colors='#FFDB58', linewidths=2)
    ax[1,1].contour(wrf_dataset_CONTROL.times, wrf_dataset_CONTROL.ZZ/1000, wrf_dataset_CONTROL.deposition.T, levels=[1e-5], colors='coral', linewidths=2)
    cs = ax[1,1].contour(
        wrf_dataset_CONTROL.times,
        wrf_dataset_CONTROL.ZZ / 1000,
        (wrf_dataset_CONTROL.kinetic_temp.T - 273.15),
        levels=np.arange(-70, 10, 5),
        colors="dimgray",
        linewidths=1,
        alpha=0.5,
    )
    ax[1,1].clabel(cs, inline=True, fmt=r'%d$^\circ$C', colors='dimgrey')
    ax[1,1].set_ylim(0, ylim)
    ax[1,1].set_ylabel("Altitude [km]")
    ax[1,1].set_title("CONTROL", pad=15)
    cbar = fig.colorbar(lwc,ax=ax[1,1],aspect=15)
    cbar.set_label(r'LWC [$\mathrm{gm^{-3}}$]')
    cbar.set_ticks(np.linspace(0, 1, 5))

    rim_handle = plt.Line2D([], [], color='#FFDB58', linewidth=2) # type: ignore
    dep_handle = plt.Line2D([], [], color='coral', linewidth=2) # type: ignore
    rim_label = r'Riming > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]'
    dep_label = r'Deposition > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]'

    ax[1,1].legend(handles=[rim_handle, dep_handle], labels=[rim_label, dep_label], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.19), frameon=False)

    lwc = ax[2,1].contourf(wrf_dataset_SIP.times, wrf_dataset_SIP.ZZ/1000, wrf_dataset_SIP.lwc.T, levels=np.linspace(0, 1, 9), cmap = cmap2)
    ax[2,1].contour(wrf_dataset_SIP.times, wrf_dataset_SIP.ZZ/1000, wrf_dataset_SIP.rimming.T, levels=[1e-5], colors='#FFDB58', linewidths=2)
    ax[2,1].contour(wrf_dataset_SIP.times, wrf_dataset_SIP.ZZ/1000, wrf_dataset_SIP.deposition.T, levels=[1e-5], colors='coral', linewidths=2)

    cs = ax[2,1].contour(
        wrf_dataset_SIP.times,
        wrf_dataset_SIP.ZZ / 1000,
        (wrf_dataset_SIP.kinetic_temp.T - 273.15),
        levels=np.arange(-70, 10, 5),
        colors="dimgray",
        linewidths=1,
        alpha=0.5,
    )
    ax[2,1].clabel(cs, inline=True, fmt=r'%d$^\circ$C', colors='dimgrey')
    ax[2,1].set_ylim(0, ylim)
    ax[2,1].set_ylabel("Altitude [km]")
    ax[2,1].set_title("SIP", pad=15)
    cbar = fig.colorbar(lwc,ax=ax[2,1], aspect=15)
    cbar.set_label(r'LWC [$\mathrm{gm^{-3}}$]')
    cbar.set_ticks(np.linspace(0, 1, 5)) # type: ignore

    ax[2,1].legend(handles=[rim_handle, dep_handle], labels=[rim_label, dep_label], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.19),frameon=False)


    fig.suptitle(
        f"WRF (bl: {wrf_dataset_CONTROL.bl_phys}) BASTA Comparison {plot_start.strftime(r'%m-%d')} to {plot_end.strftime(r'%m-%d')}"
    )

    set_up_dated_x_axis(ax[:,0], tick_locs, tick_labels)
    set_up_dated_x_axis(ax[:,1], tick_locs, tick_labels)
    plt.tight_layout()
    if save:
        plt.savefig(
            f"./figures/radar_comparison/wrfvsMIRA_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}"
        )

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = POI_1_CTRL_NPRK,
    wrf_dataset_SIP = POI_1_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = POI_3_CTRL_NPRK,
    wrf_dataset_SIP = POI_3_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = NOV_19_CTRL_NPRK,
    wrf_dataset_SIP = NOV_19_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = NOV_21_CTRL_NPRK,
    wrf_dataset_SIP = NOV_21_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = NOV_28_CTRL_NPRK,
    wrf_dataset_SIP = NOV_28_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = NOV_30_CTRL_NPRK,
    wrf_dataset_SIP = NOV_30_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)

plot_MIRA_radar_comparison(
    wrf_dataset_CONTROL = DEC_3_CTRL_NPRK,
    wrf_dataset_SIP = DEC_3_SIP_NPRK,
    MIRA_dataset = MIRA_dataset,
    MIRA_radar_step = 10,
    save = True,
    ylim = 11
)
# %%
