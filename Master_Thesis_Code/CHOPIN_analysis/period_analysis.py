# %%
import sys
from pathlib import Path

import cmasher as cmr
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from netCDF4 import Dataset  # type: ignore

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import MIRAMultiDataset, MIRAMultiDatasetFactory, WRFDataset

# %%


class SIPPeriod:
    """
    This is a class to keep together SIP periods
    """

    def __init__(
        self,
        control_ncfile_path: Path,
        sip_ncfile_path: Path,
        MIRA_factory: MIRAMultiDatasetFactory,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        tick_delta: int = 6,
    ):
        self.NPRK_control = WRFDataset(control_ncfile_path)
        self.NPRK_sip = WRFDataset(sip_ncfile_path)
        self.start_time = start_time
        self.end_time = end_time
        self.MIRA = MIRA_factory.get_dataset(start_time=start_time, end_time=end_time)
        if self.MIRA.clean_mask is not None:
            self.MIRA.refl[self.MIRA.clean_mask] = np.nan
            self.MIRA.skewness[self.MIRA.clean_mask] = np.nan
        self.wrf_mask = (self.NPRK_control.times >= self.start_time) & (
            self.NPRK_control.times <= self.end_time
        )
        self.tick_locs = mdates.drange(
            self.start_time, self.end_time + pd.Timedelta(1, "h"), pd.Timedelta(tick_delta, "hours")
        )
        self.tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in self.tick_locs]

        self.thresholds = {
            "br": 1e-2,  # L-1s-1 break up
            "hm": 1e-4,  # L-1s-1 hallett mossop process (collision break up)
            "br2": 1e-3,  # L-1s-1 break up
            "agg": 1e-5,  # L-1s-1 aggregation
            "agg2": 1e-4,  # L-1s-1 aggreation
            "agg3": 1e-3,  # L-1s-1 aggreation
            "rim": 1e-5,  # gm-3s-1 rimming
            "sb": 1e-4,  # L-1s-1 sublimation break up
            "ds": 1e-5,  # L-1s-1 droplet shattering
            "dep": 1e-5,  # deposition
            "rim": 1e-5,  # Rimming
        }

        self.temp_contour_alpha = 0.8
        self.icnc_levs = np.logspace(-2, 3, 10)

        self.lwc_levs = np.linspace(0, 1, 9)
        self.lwc_cmap = cmr.get_sub_cmap("Blues", 0.15, 1.0)

        self.refl_cmap = "jet"

    def _plot_CONTROL_refl(self, ax: Axes, fig: Figure, ymax: float = 12, temp_level_increment: int = 10):
        temp_levels = np.arange(-80, 10, temp_level_increment)

        cs = ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

        im1 = ax.pcolormesh(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.variables("Zhh_MIRA")[self.wrf_mask, :].T,
            vmin=-60,
            vmax=35,
            cmap=self.refl_cmap,
        )
        cbar = fig.colorbar(im1)
        cbar.set_label("Reflectivity [dBz]")

        ax.set_ylim(0, ymax)
        ax.set_ylabel("Altitude [km]")
        ax.set_title("WRF-CONTROL-MYNN Simulation")

    def _plot_SIP_refl(self, ax: Axes, fig: Figure, ymax: float = 12, temp_level_increment: int = 10):
        temp_levels = np.arange(-80, 10, temp_level_increment)

        cs = ax.contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

        im1 = ax.pcolormesh(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.variables("Zhh_MIRA")[self.wrf_mask, :].T,
            vmin=-60,
            vmax=35,
            cmap=self.refl_cmap,
        )
        cbar = fig.colorbar(im1)
        cbar.set_label("Reflectivity [dBz]")

        ax.set_ylim(0, ymax)
        ax.set_ylabel("Altitude [km]")
        ax.set_title("WRF-SIP-MYNN Simulation")

    def _plot_MIRA_refl(
        self,
        ax: Axes,
        fig: Figure,
        ymax: float = 12,
        temp_level_increment: int = 10,
        MIRA_radar_step: int = 10,
    ):
        temp_levels = np.arange(-80, 10, temp_level_increment)
        cs = ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        elv = self.MIRA.variables("elv")
        elv_mask = elv > 85

        im0 = ax.pcolormesh(
            self.MIRA.times[elv_mask][::MIRA_radar_step],
            self.MIRA.range / 1000,
            self.MIRA.refl[elv_mask][::MIRA_radar_step, :].T,
            vmin=-60,
            vmax=35,
            cmap=self.refl_cmap,
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBz]")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Altitude [km]")
        ax.set_title("MIRA Radar Reflectivity")

    def _plot_CONTROL_icnc(
        self,
        ax: Axes,
        fig: Figure,
        ymax: float = 12,
        temp_level_increment: int = 10,
        plot_hatching: bool = True,
        plot_multi_agg_contour: bool = False,
    ):
        agg = self.thresholds["agg"]
        agg2 = self.thresholds["agg2"]
        agg3 = self.thresholds["agg3"]

        temp_levels = np.arange(-80, 10, temp_level_increment)

        cs = ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        # plot icnc filled contour
        icnc1 = ax.contourf(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.icnc[self.wrf_mask, :].T,
            levels=self.icnc_levs,
            cmap="viridis",
            norm=colors.LogNorm(),
        )
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Altitude [km]")
        ax.set_title("WRF-CONTROL-MYNN Simulation")
        cbar = fig.colorbar(icnc1, ax=ax, aspect=15)
        cbar.ax.set_yscale("log")
        cbar.set_label(r"ICNC [$\mathrm{L^{-1}}$]")

        # Hatching
        if plot_hatching:
            ax.contour(
                self.NPRK_control.times[self.wrf_mask],
                self.NPRK_control.ZZ / 1000,
                self.NPRK_control.RH_I[self.wrf_mask, :].T,
                levels=[100.1],
                colors="k",
                linewidths=0.8,
                linestyles="solid",
            )
            ax.contourf(
                self.NPRK_control.times[self.wrf_mask],
                self.NPRK_control.ZZ / 1000,
                self.NPRK_control.RH_I[self.wrf_mask, :].T,
                levels=[100.1, 150],
                hatches=["////"],
                colors="none",
                linestyles="solid",
            )

        if plot_multi_agg_contour:
            ax.contour(
                self.NPRK_control.times[self.wrf_mask],
                self.NPRK_control.ZZ / 1000,
                self.NPRK_control.aggregation[self.wrf_mask, :].T,
                levels=[agg],
                colors="blue",
                linewidths=2,
            )

            ax.contour(
                self.NPRK_control.times[self.wrf_mask],
                self.NPRK_control.ZZ / 1000,
                self.NPRK_control.aggregation[self.wrf_mask, :].T,
                levels=[agg2],
                colors="cyan",
                linewidths=2,
            )
            ax.contour(
                self.NPRK_control.times[self.wrf_mask],
                self.NPRK_control.ZZ / 1000,
                self.NPRK_control.aggregation[self.wrf_mask, :].T,
                levels=[agg3],
                colors="gold",
                linewidths=2,
            )

        if not plot_multi_agg_contour:
            ax.contour(
                self.NPRK_control.times[self.wrf_mask],
                self.NPRK_control.ZZ / 1000,
                self.NPRK_control.aggregation[self.wrf_mask, :].T,
                levels=[agg],
                colors="red",
                linewidths=2,
            )

    def _plot_SIP_icnc(
        self,
        ax: Axes,
        fig: Figure,
        ymax: float = 12,
        temp_level_increment: int = 10,
        plot_sip_contours: bool = True,
        plot_multi_agg_contours: bool = False,
        plot_hatching: bool = True,
    ):
        agg = self.thresholds["agg"]
        agg2 = self.thresholds["agg2"]
        agg3 = self.thresholds["agg3"]
        br = self.thresholds["br"]
        hm = self.thresholds["hm"]
        br2 = self.thresholds["br2"]
        sb = self.thresholds["sb"]
        ds = self.thresholds["ds"]
        contour_linewidth = 2

        temp_levels = np.arange(-80, 10, temp_level_increment)

        cs = ax.contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        # plot icnc filled contour
        icnc2 = ax.contourf(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.icnc[self.wrf_mask, :].T,
            levels=self.icnc_levs,
            cmap="viridis",
            norm=colors.LogNorm(),
        )
        # Hatching
        if plot_hatching:
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.RH_I[self.wrf_mask, :].T,
                levels=[100.1],
                colors="k",
                linewidths=0.8,
                linestyles="solid",
            )
            ax.contourf(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.RH_I[self.wrf_mask, :].T,
                levels=[100.1, 150],
                hatches=["////"],
                colors="none",
                linestyles="solid",
            )
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Altitude [km]")
        ax.set_title("WRF-SIP-MYNN simulation")
        cbar = fig.colorbar(icnc2, ax=ax, aspect=15)
        cbar.ax.set_yscale("log")
        cbar.set_label(r"ICNC [$\mathrm{L^{-1}}$]")

        if plot_sip_contours:
            # plot SIP contours
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.breakup[self.wrf_mask, :].T,
                levels=[br],
                colors="darkviolet",
                linewidths=contour_linewidth,
            )
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.hallett_mossop[self.wrf_mask, :].T,
                levels=[hm],
                colors="darkcyan",
                linewidths=contour_linewidth,
            )
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.breakup[self.wrf_mask, :].T,
                levels=[br2],
                colors="darkviolet",
                linewidths=contour_linewidth,
                linestyles="dashed",
            )
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.sublimation_breakup[self.wrf_mask, :].T,
                levels=[sb],
                colors="magenta",
                linewidths=contour_linewidth,
            )
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.droplet_shatter[self.wrf_mask, :].T,
                levels=[ds],
                colors="cyan",
                linewidths=contour_linewidth,
            )

        if plot_multi_agg_contours:
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.aggregation[self.wrf_mask, :].T,
                levels=[agg],
                colors="blue",
                linewidths=contour_linewidth,
            )
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.aggregation[self.wrf_mask, :].T,
                levels=[agg2],
                colors="cyan",
                linewidths=contour_linewidth,
            )
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.aggregation[self.wrf_mask, :].T,
                levels=[agg3],
                colors="gold",
                linewidths=contour_linewidth,
            )

        if not plot_multi_agg_contours:
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                self.NPRK_sip.aggregation[self.wrf_mask, :].T,
                levels=[agg],
                colors="red",
                linewidths=contour_linewidth,
            )

    def _plot_MIRA_skewness(self, ax: Axes, fig: Figure, ymax: float = 12, temp_level_increment: int = 10):
        temp_levels = np.arange(-80, 10, temp_level_increment)

        cs = ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

        elv = self.MIRA.variables("elv")
        mask_elv = elv > 85
        MIRA_skewness = self.MIRA.skewness[mask_elv, :]
        skewness = ax.pcolormesh(
            self.MIRA.times[mask_elv],
            self.MIRA.range / 1000,
            MIRA_skewness.T,
            vmin=-1,
            vmax=1,
            cmap="seismic",
        )

        cbar = fig.colorbar(skewness, ax=ax, aspect=15)
        cbar.set_label("Skewness")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Altitude [km]")
        ax.set_title("MIRA Radar Skewness")

    def _plot_CONTROL_lwc(self, ax: Axes, fig: Figure, ymax: float = 12, temp_level_increment: int = 10):

        temp_levels = np.arange(-80, 10, temp_level_increment)
        dep = self.thresholds["dep"]
        rim = self.thresholds["rim"]

        lwc = ax.contourf(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.lwc[self.wrf_mask, :].T,
            levels=self.lwc_levs,
            cmap=self.lwc_cmap,
        )
        ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.rimming[self.wrf_mask, :].T,
            levels=[rim],
            colors="#FFDB58",
            linewidths=2,
        )
        ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.deposition[self.wrf_mask, :].T,
            levels=[dep],
            colors="coral",
            linewidths=2,
        )
        cs = ax.contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        cbar = fig.colorbar(lwc, ax=ax, aspect=15)
        cbar.set_label(r"LWC CONTROL [$\mathrm{gm^{-3}}$]")
        cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(0, ymax)
        ax.set_title("WRF-CONTROL-MYNN Simulation")

    def _plot_SIP_lwc(self, ax: Axes, fig: Figure, ymax: float = 12, temp_level_increment: int = 10):
        temp_levels = np.arange(-80, 10, temp_level_increment)
        dep = self.thresholds["dep"]
        rim = self.thresholds["rim"]

        lwc = ax.contourf(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.lwc[self.wrf_mask, :].T,
            levels=self.lwc_levs,
            cmap=self.lwc_cmap,
        )
        ax.contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.rimming[self.wrf_mask, :].T,
            levels=[rim],
            colors="#FFDB58",
            linewidths=2,
        )
        ax.contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.deposition[self.wrf_mask, :].T,
            levels=[dep],
            colors="coral",
            linewidths=2,
        )
        cs = ax.contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=self.temp_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        cbar = fig.colorbar(lwc, ax=ax, aspect=15)
        cbar.set_label(r"LWC SIP [$\mathrm{gm^{-3}}$]")
        cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(0, ymax)
        ax.set_title("WRF-SIP-MYNN simulation")

    def plot_icnc_skewness(
        self,
        save_path: Path | None = None,
        ymax: float = 12,
        temp_level_increment: int = 10,
        save_id: str | None = None,
        plot_sip_contours: bool = True,
        plot_hatching: bool = True,
        plot_mulit_agg_contours: bool = False,
    ):

        br_label = r"BR$_{rate}$ > 10$^{-2}$ [L$^{-1}$ s$^{-1}$]"
        hm_label = r"HM$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        br_label2 = r"BR$_{rate}$ > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"
        sb_label = r"SUBBR$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        ds_label = r"DS$_{rate}$ > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
        agg_label = r"Aggregation > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
        agg2_label = r"Aggregation > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        agg3_label = r"Aggregation > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"

        fig, axs = plt.subplots(3, 1, figsize=(9, 7))
        for ax in axs:
            ax.set_yticks([0, 2, 4, 6, 8, 10])
        self._plot_CONTROL_icnc(
            axs[0],
            fig,
            ymax=ymax,
            temp_level_increment=temp_level_increment,
            plot_hatching=plot_hatching,
            plot_multi_agg_contour=plot_mulit_agg_contours,
        )
        # adding legend
        br_handle = Line2D([], [], color="darkviolet", linewidth=3)
        hm_handle = Line2D([], [], color="darkcyan", linewidth=3)
        br_handle2 = Line2D([], [], color="darkviolet", linestyle="--", linewidth=3)
        sb_handle = Line2D([], [], color="magenta", linewidth=3)
        ds_handle = Line2D([], [], color="cyan", linewidth=3)
        agg_handle = Line2D([], [], color="red", linewidth=3)
        agg_multi_handle = Line2D([], [], color="blue", linewidth=3)
        agg2_multi_handle = Line2D([], [], color="cyan", linewidth=3)
        agg3_multi_handle = Line2D([], [], color="gold", linewidth=3)

        handles = []
        labels = []
        if plot_sip_contours:
            handles += [br_handle, br_handle2, sb_handle, ds_handle, hm_handle]
            labels += [br_label, br_label2, sb_label, ds_label, hm_label]

        if plot_mulit_agg_contours:
            handles += [agg_multi_handle, agg2_multi_handle, agg3_multi_handle]
            labels += [agg_label, agg2_label, agg3_label]

        if not plot_mulit_agg_contours:
            handles += [agg_handle]
            labels += [agg_label]

        axs[0].legend(
            handles=handles,
            labels=labels,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.15),
            frameon=False,
        )

        self._plot_SIP_icnc(
            axs[1],
            fig,
            ymax=ymax,
            temp_level_increment=temp_level_increment,
            plot_sip_contours=plot_sip_contours,
            plot_multi_agg_contours=plot_mulit_agg_contours,
        )
        self._plot_MIRA_skewness(axs[2], fig, ymax=ymax, temp_level_increment=temp_level_increment)

        set_up_dated_x_axis(axs, self.tick_locs, self.tick_labels)

        if save_path is not None:
            file_name = f"skewness_plot_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"skewness_plot_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_icnc(
        self,
        save_path: Path | None = None,
        ymax: float = 12,
        temp_level_increment: int = 10,
        save_id: str | None = None,
        plot_sip_contours: bool = True,
        plot_hatching: bool = True,
        plot_mulit_agg_contours: bool = False,
    ):

        br_label = r"BR$_{rate}$ > 10$^{-2}$ [L$^{-1}$ s$^{-1}$]"
        hm_label = r"HM$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        br_label2 = r"BR$_{rate}$ > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"
        sb_label = r"SUBBR$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        ds_label = r"DS$_{rate}$ > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
        agg_label = r"Aggregation > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
        agg2_label = r"Aggregation > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        agg3_label = r"Aggregation > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"

        fig, axs = plt.subplots(2, 1, figsize=(9, 4.75))
        for ax in axs:
            ax.set_yticks([0, 2, 4, 6, 8, 10])
        self._plot_CONTROL_icnc(
            axs[0],
            fig,
            ymax=ymax,
            temp_level_increment=temp_level_increment,
            plot_hatching=plot_hatching,
            plot_multi_agg_contour=plot_mulit_agg_contours,
        )
        # adding legend
        br_handle = Line2D([], [], color="darkviolet", linewidth=3)
        hm_handle = Line2D([], [], color="darkcyan", linewidth=3)
        br_handle2 = Line2D([], [], color="darkviolet", linestyle="--", linewidth=3)
        sb_handle = Line2D([], [], color="magenta", linewidth=3)
        ds_handle = Line2D([], [], color="cyan", linewidth=3)
        agg_handle = Line2D([], [], color="red", linewidth=3)
        agg_multi_handle = Line2D([], [], color="blue", linewidth=3)
        agg2_multi_handle = Line2D([], [], color="cyan", linewidth=3)
        agg3_multi_handle = Line2D([], [], color="gold", linewidth=3)

        handles = []
        labels = []
        if plot_sip_contours:
            handles += [br_handle, br_handle2, sb_handle, ds_handle, hm_handle]
            labels += [br_label, br_label2, sb_label, ds_label, hm_label]

        if plot_mulit_agg_contours:
            handles += [agg_multi_handle, agg2_multi_handle, agg3_multi_handle]
            labels += [agg_label, agg2_label, agg3_label]

        if not plot_mulit_agg_contours:
            handles += [agg_handle]
            labels += [agg_label]

        axs[0].legend(
            handles=handles,
            labels=labels,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.15),
            frameon=False,
        )

        self._plot_SIP_icnc(
            axs[1],
            fig,
            ymax=ymax,
            temp_level_increment=temp_level_increment,
            plot_sip_contours=plot_sip_contours,
            plot_multi_agg_contours=plot_mulit_agg_contours,
        )

        set_up_dated_x_axis(axs, self.tick_locs, self.tick_labels)

        if save_path is not None:
            file_name = f"icnc_plot_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"icnc_plot_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_icnc_lwc(
        self, save_path: Path | None = None, ymax: float = 12, temp_level_increment: int = 10, title: str = ""
    ):
        br_label = r"BR$_{rate}$ > 10$^{-2}$ [L$^{-1}$ s$^{-1}$]"
        hm_label = r"HM$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        br_label2 = r"BR$_{rate}$ > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"
        sb_label = r"SUBBR$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
        ds_label = r"DS$_{rate}$ > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
        agg_label = r"Aggregation > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"

        fig, axs = plt.subplots(3, 2, figsize=(23, 10))
        for ax in axs.flatten():
            ax.set_yticks([0, 2, 4, 6, 8, 10])
        self._plot_CONTROL_icnc(axs[0, 0], fig, ymax=ymax, temp_level_increment=temp_level_increment)
        # adding legend
        br_handle = Line2D([], [], color="darkviolet", linewidth=2)
        # hm_handle = Line2D([], [], color='darkcyan', linewidth=2)
        br_handle2 = Line2D([], [], color="darkviolet", linestyle="--", linewidth=2)
        sb_handle = Line2D([], [], color="magenta", linewidth=2)
        ds_handle = Line2D([], [], color="cyan", linewidth=2)
        agg_handle = Line2D([], [], color="red", linewidth=2)
        axs[0, 0].legend(
            handles=[br_handle, br_handle2, sb_handle, ds_handle, agg_handle],
            labels=[br_label, br_label2, sb_label, ds_label, agg_label],
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.375),
            frameon=False,
        )

        self._plot_SIP_icnc(axs[1, 0], fig, ymax=ymax, temp_level_increment=temp_level_increment)
        self._plot_MIRA_skewness(axs[2, 0], fig, ymax=ymax, temp_level_increment=temp_level_increment)

        rim_label = r"Riming > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]"
        dep_label = r"Deposition > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]"
        rim_handle = Line2D([], [], color="#FFDB58", linewidth=2)
        dep_handle = Line2D([], [], color="coral", linewidth=2)

        self._plot_CONTROL_lwc(axs[0, 1], fig, ymax=ymax, temp_level_increment=temp_level_increment)
        axs[0, 1].legend(
            handles=[rim_handle, dep_handle],
            labels=[rim_label, dep_label],
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.36, 1.275),
            frameon=False,
        )

        self._plot_SIP_lwc(axs[1, 1], fig, ymax=ymax, temp_level_increment=temp_level_increment)

        axs[2, 1].axis("off")

        set_up_dated_x_axis(axs[:, 0], self.tick_locs, self.tick_labels)
        set_up_dated_x_axis(axs[:-1, 1], self.tick_locs, self.tick_labels)

        fig.suptitle(title, x=0.45)

        plt.tight_layout()
        if save_path is not None:
            file_name = f"skewness_lwc_plot_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if title != "":
                file_name = f"skewness_lwc_plot_{title}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_lwc(
        self,
        save_path: Path | None = None,
        ymax: float = 12,
        temp_level_increment: int = 10,
        save_id: str | None = None,
    ):
        levs = np.linspace(0, 1, 9)
        temp_contour_alpha = 0.8
        temp_levels = np.arange(-80, 10, temp_level_increment)
        cmap2 = cmr.get_sub_cmap("Blues", 0.15, 1.0)

        dep = 1e-5
        rim = 1e-5

        rim_label = r"Riming > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]"
        dep_label = r"Deposition > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]"

        fig, axs = plt.subplots(3, 1, figsize=(9, 7))
        axs[0].set_yticks([0, 2, 4, 6, 8, 10])
        axs[1].set_yticks([0, 2, 4, 6, 8, 10])
        ### Plot control
        lwc = axs[0].contourf(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.lwc[self.wrf_mask, :].T,
            levels=levs,
            cmap=cmap2,
        )
        axs[0].contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.rimming[self.wrf_mask, :].T,
            levels=[rim],
            colors="#FFDB58",
            linewidths=2,
        )
        axs[0].contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            self.NPRK_control.deposition[self.wrf_mask, :].T,
            levels=[dep],
            colors="coral",
            linewidths=2,
        )
        cs = axs[0].contour(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.ZZ / 1000,
            (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=temp_contour_alpha,
        )
        rim_handle = Line2D([], [], color="#FFDB58", linewidth=3)
        dep_handle = Line2D([], [], color="coral", linewidth=3)
        axs[0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        cbar = fig.colorbar(lwc, ax=axs[0], aspect=15)
        cbar.set_label(r"LWC [$\mathrm{gm^{-3}}$]")
        cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
        axs[0].set_ylabel("Altitude [km]")
        axs[0].set_ylim(0, ymax)
        axs[0].legend(
            handles=[rim_handle, dep_handle],
            labels=[rim_label, dep_label],
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.175),
            frameon=False,
        )
        axs[0].set_title("WRF-CONTROL-MYNN Simulation")

        ###Plot SIP
        lwc = axs[1].contourf(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.lwc[self.wrf_mask, :].T,
            levels=levs,
            cmap=cmap2,
        )
        axs[1].contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.rimming[self.wrf_mask, :].T,
            levels=[rim],
            colors="#FFDB58",
            linewidths=2,
        )
        axs[1].contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            self.NPRK_sip.deposition[self.wrf_mask, :].T,
            levels=[dep],
            colors="coral",
            linewidths=2,
        )
        cs = axs[1].contour(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.ZZ / 1000,
            (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=temp_contour_alpha,
        )
        axs[1].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        cbar = fig.colorbar(lwc, ax=axs[1], aspect=15)
        cbar.set_label(r"LWC [$\mathrm{gm^{-3}}$]")
        cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
        axs[1].set_ylabel("Altitude [km]")
        axs[1].set_ylim(0, ymax)
        axs[1].set_title("WRF-SIP-MYNN simulation")

        # ###Plot MIRA
        # elv = self.MIRA.variables("elv")
        # mask_elv = elv > 85
        # lwc = axs[2].contourf(
        #     self.MIRA.times[mask_elv],
        #     self.MIRA.range / 1000,
        #     self.MIRA.lwc[mask_elv, :].T,
        #     levels=levs,
        #     cmap=cmap2,
        # )
        # cs = axs[2].contour(
        #     self.NPRK_control.times[self.wrf_mask],
        #     self.NPRK_control.ZZ / 1000,
        #     (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
        #     levels=temp_levels,
        #     colors="dimgray",
        #     linewidths=1,
        #     alpha=temp_contour_alpha,
        # )
        # axs[2].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        # cbar = fig.colorbar(lwc, ax=axs[2], aspect=15)
        # cbar.set_label(r"LWC MIRA [$\mathrm{gm^{-3}}$]")
        # cbar.set_ticks(np.linspace(0, 1, 5))
        # axs[2].set_ylabel("Altitude [km]")
        # axs[2].set_ylim(0, ymax)
        # axs[2].set_title("MIRA Radar")

        ###Plot LWP
        # axs[2].plot(self.MIRA.times, self.MIRA.lwp, color="grey", label="MIRA")
        axs[2].plot(
            self.NPRK_control.times[self.wrf_mask],
            self.NPRK_control.lwp[self.wrf_mask],
            color="k",
            label="CONTROL",
        )
        axs[2].plot(
            self.NPRK_sip.times[self.wrf_mask],
            self.NPRK_sip.lwp[self.wrf_mask],
            color="b",
            label="SIP",
        )
        axs[2].set_ylabel("LWP [g m$^{-3}$]")
        axs[2].set_xlim(self.start_time, self.end_time)
        axs[2].legend()
        cbar = fig.colorbar(cs, ax=axs[2])
        cbar.ax.set_visible(False)
        set_up_dated_x_axis(axs, tick_locs=self.tick_locs, tick_labels=self.tick_labels)

        if save_path is not None:
            file_name = f"LWC_plot_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"LWC_plot_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_refl(
        self,
        save_path: Path | None = None,
        ymax: float = 12,
        temp_level_increment: int = 10,
        title: str = "",
        save_id: str | None = None,
    ):
        fig, axs = plt.subplots(3, 1, figsize=(9, 7))
        for ax in axs:
            ax.set_yticks([0, 2, 4, 6, 8, 10])
        self._plot_MIRA_refl(ax=axs[2], fig=fig, ymax=ymax, temp_level_increment=temp_level_increment)
        self._plot_CONTROL_refl(ax=axs[0], fig=fig, ymax=ymax, temp_level_increment=temp_level_increment)
        self._plot_SIP_refl(ax=axs[1], fig=fig, ymax=ymax, temp_level_increment=temp_level_increment)

        fig.suptitle(title, x=0.45)

        set_up_dated_x_axis(axs, self.tick_locs, self.tick_labels)
        plt.tight_layout()
        if save_path is not None:
            file_name = f"Radar_comparison_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"Radar_comparison_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            elif title != "":
                file_name = f"Radar_comparison_{title}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_sip_contours_1(
        self,
        save_path: Path | None = None,
        ymax: float = 12,
        temp_level_increment: int = 10,
        title: str = "",
        save_id: str | None = None,
    ):

        temp_levels = np.arange(-80, 10, temp_level_increment)
        sip_levs = np.logspace(-6, -1, 11)

        fig, axs = plt.subplots(3, 1, figsize=(9, 7))
        for i in range(3):
            axs[i].set_yticks([0, 2, 4, 6, 8, 10])

            cs = axs[i].contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
                levels=temp_levels,
                colors="dimgray",
                linewidths=1,
                alpha=self.temp_contour_alpha,
            )
            axs[i].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
            if i < 2:
                axs[i].contour(
                    self.NPRK_sip.times[self.wrf_mask],
                    self.NPRK_sip.ZZ / 1000,
                    (self.NPRK_sip.icnc[self.wrf_mask, :] > 0.01).T,
                    levels=[0],
                    colors="k",
                    linewidths=2,
                    alpha=1,
                )
            axs[i].set_ylim(0, ymax)
            axs[i].set_ylabel("Altitude [km]")
            # plot icnc filled contour

        for ax, sip_process, sip_cmap, cbar_label, ax_title in zip(
            axs,
            [self.NPRK_sip.breakup, self.NPRK_sip.sublimation_breakup],
            ["Purples", "RdPu"],
            [r"BR$_{rate}$ [L$^{-1}$ s$^{-1}$]", r"SUBRp$_{rate}$ [L$^{-1}$ s$^{-1}$]"],
            ["Break-Up", "Sublimation Break-Up"],
        ):
            sip = ax.contourf(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                sip_process[self.wrf_mask, :].T,
                levels=sip_levs,
                cmap=sip_cmap,
                norm=colors.LogNorm(),
            )

            cbar = fig.colorbar(sip, ax=ax, aspect=15)
            cbar.ax.set_yscale("log")
            cbar.set_label(cbar_label)

            ax.set_title(ax_title)

        self._plot_MIRA_skewness(axs[2], fig, ymax=ymax, temp_level_increment=temp_level_increment)
        set_up_dated_x_axis(axs, self.tick_locs, self.tick_labels)
        plt.tight_layout()
        if save_path is not None:
            file_name = f"sip_contours_1_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"sip_contours_1_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            elif title != "":
                file_name = f"sip_contours_1_{title}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_sip_contours_2(
        self,
        save_path: Path | None = None,
        ymax: float = 12,
        temp_level_increment: int = 10,
        title: str = "",
        save_id: str | None = None,
    ):

        temp_levels = np.arange(-80, 10, temp_level_increment)
        sip_levs = np.logspace(-6, -1, 11)

        fig, axs = plt.subplots(2, 1, figsize=(9, 4.75))
        for ax in axs:
            ax.set_yticks([0, 2, 4, 6, 8, 10])

            cs = ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
                levels=temp_levels,
                colors="dimgray",
                linewidths=1,
                alpha=self.temp_contour_alpha,
            )
            ax.clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
            ax.contour(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                (self.NPRK_sip.icnc[self.wrf_mask, :] > 0.01).T,
                levels=[0],
                colors="k",
                linewidths=1,
                alpha=1,
            )
            ax.set_ylim(0, ymax)
            ax.set_ylabel("Altitude [km]")
            # plot icnc filled contour

        for ax, sip_process, sip_cmap, cbar_label, ax_title in zip(
            axs,
            [self.NPRK_sip.droplet_shatter, self.NPRK_sip.hallett_mossop],
            ["Blues", "Greens"],
            [r"DS$_{rate}$ [L$^{-1}$ s$^{-1}$]", r"HM$_{rate}$ [L$^{-1}$ s$^{-1}$]"],
            ["Droplet Shatter", "Hallett Mossop"],
        ):
            sip = ax.contourf(
                self.NPRK_sip.times[self.wrf_mask],
                self.NPRK_sip.ZZ / 1000,
                sip_process[self.wrf_mask, :].T,
                levels=sip_levs,
                cmap=sip_cmap,
                norm=colors.LogNorm(),
            )

            cbar = fig.colorbar(sip, ax=ax, aspect=15)
            cbar.ax.set_yscale("log")
            cbar.set_label(cbar_label)

            ax.set_title(ax_title)

        set_up_dated_x_axis(axs, self.tick_locs, self.tick_labels)
        plt.tight_layout()
        if save_path is not None:
            file_name = f"sip_contours_2_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"sip_contours_2_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            elif title != "":
                file_name = f"sip_contours_2_{title}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    def plot_sip_rates(
        self,
        save_path: Path | None = None,
        title: str = "",
        save_id: str | None = None,
    ):
        hm = self.NPRK_sip.hallett_mossop[self.wrf_mask, :].copy()
        ds = self.NPRK_sip.droplet_shatter[self.wrf_mask, :].copy()
        br = self.NPRK_sip.breakup[self.wrf_mask, :].copy()
        subbr = self.NPRK_sip.sublimation_breakup[self.wrf_mask, :].copy()

        # hm[hm == 0] = np.nan
        # ds[ds == 0] = np.nan
        # br[br == 0] = np.nan
        # subbr[subbr == 0] = np.nan

        # hm_time = np.nanmedian(hm, axis=1)
        # ds_time = np.nanmedian(ds, axis=1)
        # br_time = np.nanmedian(br, axis=1)
        # subbr_time = np.nanmedian(subbr, axis=1)

        hm_time = np.sum(hm, axis=1)
        ds_time = np.sum(ds, axis=1)
        br_time = np.sum(br, axis=1)
        subbr_time = np.sum(subbr, axis=1)

        plt.figure(figsize=(9, 4))
        plt.yscale("log")
        plt.ylim(0.00001, 15)
        plt.yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
        plt.grid(axis="y")
        plt.plot(self.NPRK_sip.times[self.wrf_mask], br_time, color="tab:purple", label="Break-Up")
        plt.plot(self.NPRK_sip.times[self.wrf_mask], subbr_time, color="tab:pink", label="Sublim. Break-Up")
        plt.plot(self.NPRK_sip.times[self.wrf_mask], ds_time, color="tab:cyan", label="Droplet Shatter")
        plt.plot(self.NPRK_sip.times[self.wrf_mask], hm_time, color="tab:green", label="Hallet Mossop")
        plt.legend(ncols=4, bbox_to_anchor=(0.5, 1.02), loc="lower center")

        plt.xticks(self.tick_locs, self.tick_labels)
        plt.xlabel("Date [UTC]")
        plt.ylabel("SIP rate [L$^{-1}$ s$^{-1}$]")

        plt.tight_layout()
        if save_path is not None:
            file_name = f"sip_rates_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            if save_id is not None:
                file_name = f"sip_rates_{save_id}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            elif title != "":
                file_name = f"sip_rates_{title}_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
            file_save_path = Path(save_path, file_name)
            plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
            plt.close()

    # def _plot_icnc_skewness(
    #     self, save_path: Path | None = None, ymax: float = 12, temp_level_increment: int = 10
    # ):
    #     levs = np.logspace(-2, 3, 10)
    #     br = 1e-2  # L-1s-1 break up
    #     hm = 1e-4  # L-1s-1 hallett mossop process (collision break up)
    #     br2 = 1e-3  # L-1s-1 break up
    #     agg = 1e-5  # L-1s-1 aggregation
    #     rim = 1e-5  # gm-3s-1 rimming
    #     sb = 1e-4  # L-1s-1 sublimation break up
    #     ds = 1e-5  # L-1s-1 droplet shattering

    #     br_label = r"BR$_{rate}$ > 10$^{-2}$ [L$^{-1}$ s$^{-1}$]"
    #     hm_label = r"HM$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
    #     br_label2 = r"BR$_{rate}$ > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"
    #     sb_label = r"SUBBR$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
    #     ds_label = r"DS$_{rate}$ > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
    #     agg_label = r"Aggregation > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"

    #     temp_contour_alpha = 0.8
    #     temp_levels = np.arange(-80, 10, temp_level_increment)

    #     fig, axs = plt.subplots(3, 1, figsize=(14, 10))

    #     ### Plot control
    #     # plot temperature contour

    #     cs = axs[0].contour(
    #         self.NPRK_control.times[self.wrf_mask],
    #         self.NPRK_control.ZZ / 1000,
    #         (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
    #         levels=temp_levels,
    #         colors="dimgray",
    #         linewidths=1,
    #         alpha=temp_contour_alpha,
    #     )
    #     axs[0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
    #     # plot icnc filled contour
    #     icnc1 = axs[0].contourf(
    #         self.NPRK_control.times[self.wrf_mask],
    #         self.NPRK_control.ZZ / 1000,
    #         self.NPRK_control.icnc[self.wrf_mask, :].T,
    #         levels=levs,
    #         cmap="viridis",
    #         norm=colors.LogNorm(),
    #     )
    #     axs[0].set_ylim(0, ymax)
    #     axs[0].set_ylabel("Altitude [km]")
    #     axs[0].set_title("WRF-CONTROL-MYNN Simulation")
    #     cbar = fig.colorbar(icnc1, ax=axs[0], aspect=15)
    #     cbar.ax.set_yscale("log")
    #     cbar.set_label(r"ICNC CONTROL [$\mathrm{L^{-1}}$]")

    #     # SIP contours
    #     axs[0].contour(
    #         self.NPRK_control.times[self.wrf_mask],
    #         self.NPRK_control.ZZ / 1000,
    #         self.NPRK_control.aggregation[self.wrf_mask, :].T,
    #         levels=[agg],
    #         colors="red",
    #         linewidths=2,
    #     )
    #     axs[0].contour(
    #         self.NPRK_control.times[self.wrf_mask],
    #         self.NPRK_control.ZZ / 1000,
    #         self.NPRK_control.RH_I[self.wrf_mask, :].T,
    #         levels=[100.1],
    #         colors="k",
    #         linewidths=0.8,
    #         linestyles="solid",
    #     )
    #     axs[0].contourf(
    #         self.NPRK_control.times[self.wrf_mask],
    #         self.NPRK_control.ZZ / 1000,
    #         self.NPRK_control.RH_I[self.wrf_mask, :].T,
    #         levels=[100.1, 150],
    #         hatches=["////"],
    #         colors="none",
    #         linestyles="solid",
    #     )

    #     # adding legend
    #     br_handle = Line2D([], [], color="darkviolet", linewidth=2)
    #     # hm_handle = Line2D([], [], color='darkcyan', linewidth=2)
    #     br_handle2 = Line2D([], [], color="darkviolet", linestyle="--", linewidth=2)
    #     sb_handle = Line2D([], [], color="magenta", linewidth=2)
    #     ds_handle = Line2D([], [], color="cyan", linewidth=2)
    #     agg_handle = Line2D([], [], color="red", linewidth=2)
    #     axs[0].legend(
    #         handles=[br_handle, br_handle2, sb_handle, ds_handle, agg_handle],
    #         labels=[br_label, br_label2, sb_label, ds_label, agg_label],
    #         ncol=3,
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, 1.375),
    #         frameon=False,
    #     )

    #     ### Plot SIP
    #     # plot temperature contour
    #     cs = axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         (self.NPRK_sip.kinetic_temp[self.wrf_mask, :].T - 273.15),
    #         levels=temp_levels,
    #         colors="dimgray",
    #         linewidths=1,
    #         alpha=temp_contour_alpha,
    #     )
    #     axs[1].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
    #     # plot icnc filled contour
    #     icnc2 = axs[1].contourf(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.icnc[self.wrf_mask, :].T,
    #         levels=levs,
    #         cmap="viridis",
    #         norm=colors.LogNorm(),
    #     )
    #     axs[1].set_ylim(0, ymax)
    #     axs[1].set_ylabel("Altitude [km]")
    #     axs[1].set_title("WRF-SIP-MYNN simulation")
    #     cbar = fig.colorbar(icnc2, ax=axs[1], aspect=15)
    #     cbar.ax.set_yscale("log")
    #     cbar.set_label(r"ICNC SIP [$\mathrm{L^{-1}}$]")
    #     # plot SIP contours
    #     axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.aggregation[self.wrf_mask, :].T,
    #         levels=[agg],
    #         colors="red",
    #         linewidths=2,
    #     )
    #     axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.breakup[self.wrf_mask, :].T,
    #         levels=[br],
    #         colors="darkviolet",
    #         linewidths=2,
    #     )
    #     axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.hallett_mossop[self.wrf_mask, :].T,
    #         levels=[hm],
    #         colors="darkcyan",
    #         linewidths=2,
    #     )
    #     axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.breakup[self.wrf_mask, :].T,
    #         levels=[br2],
    #         colors="darkviolet",
    #         linewidths=2,
    #         linestyles="dashed",
    #     )
    #     axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.sublimation_breakup[self.wrf_mask, :].T,
    #         levels=[sb],
    #         colors="magenta",
    #         linewidths=2,
    #     )
    #     axs[1].contour(
    #         self.NPRK_sip.times[self.wrf_mask],
    #         self.NPRK_sip.ZZ / 1000,
    #         self.NPRK_sip.droplet_shatter[self.wrf_mask, :].T,
    #         levels=[ds],
    #         colors="cyan",
    #         linewidths=2,
    #     )

    #     ### Plot skewness
    #     # give same temperature contour as control
    #     cs = axs[2].contour(
    #         self.NPRK_control.times[self.wrf_mask],
    #         self.NPRK_control.ZZ / 1000,
    #         (self.NPRK_control.kinetic_temp[self.wrf_mask, :].T - 273.15),
    #         levels=temp_levels,
    #         colors="dimgray",
    #         linewidths=1,
    #         alpha=temp_contour_alpha,
    #     )
    #     axs[2].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

    #     elv = self.MIRA.variables("elv")
    #     mask_elv = elv > 85
    #     MIRA_skewness = self.MIRA.skewness[mask_elv, :]
    #     skewness = axs[2].pcolormesh(
    #         self.MIRA.times[mask_elv],
    #         self.MIRA.range / 1000,
    #         MIRA_skewness.T,
    #         vmin=-1,
    #         vmax=1,
    #         cmap="seismic",
    #     )

    #     cbar = fig.colorbar(skewness, ax=axs[2], aspect=15)
    #     cbar.set_label("Skewness")
    #     axs[2].set_ylim(0, ymax)
    #     axs[2].set_ylabel("Altitude [km]")
    #     axs[2].set_title("MIRA Radar Skewness")

    #     set_up_dated_x_axis(axs, tick_locs=self.tick_locs, tick_labels=self.tick_labels)

    #     if save_path is not None:
    #         file_name = f"skewness_plot_{self.start_time.strftime(r'%Y%m%dH%H')}-{self.end_time.strftime(r'%Y%m%dH%H')}.png"
    #         file_save_path = Path(save_path, file_name)
    #         plt.savefig(file_save_path, bbox_inches="tight", dpi=300)
    #         plt.close()


# %%
if __name__ == "__main__":
    MIRA_dataset_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
    wrf_metadata = pd.read_csv(
        "./data/metadata.csv", parse_dates=["start_time", "end_time", "true_start_time"]
    )
    legend_size = 11
    small_size = 13
    medium_size = 14
    large_size = 14
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=legend_size)  # legend fontsize
    plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
    plt.rc("axes", titlesize=medium_size)
    sip_mask = (
        (wrf_metadata.sip == True)
        & (wrf_metadata.bl == 5)
        & (wrf_metadata.mp == 10)
        & (wrf_metadata.station == "NPRK")
    )
    control_mask = (
        (wrf_metadata.sip == False)
        & (wrf_metadata.bl == 5)
        & (wrf_metadata.mp == 10)
        & (wrf_metadata.station == "NPRK")
    )

    sip_runs = wrf_metadata.loc[sip_mask]
    control_runs = wrf_metadata.loc[control_mask]

    sip_start = set(sip_runs.start_time)
    sip_end = set(sip_runs.end_time)
    control_start = set(control_runs.start_time)
    control_end = set(control_runs.end_time)

    start_intersection = sip_start.intersection(control_start)
    end_intersection = sip_end.intersection(control_end)

    sip_overlapping_runs_mask = sip_runs.apply(
        lambda x: x.start_time in start_intersection and x.end_time in end_intersection, axis=1
    )

    control_overlapping_runs_mask = control_runs.apply(
        lambda x: x.start_time in start_intersection and x.end_time in end_intersection, axis=1
    )

    sip_periods = sip_runs.loc[sip_overlapping_runs_mask].copy()
    sip_periods.sort_values("start_time", inplace=True)
    sip_periods.reset_index(drop=True, inplace=True)
    control_periods = control_runs.loc[control_overlapping_runs_mask].copy()
    control_periods.sort_values("start_time", inplace=True)
    control_periods.reset_index(drop=True, inplace=True)

    assert len(sip_periods) == len(control_periods)

    # %%
    save_path = Path("/home/waseem/CHOPIN_analysis/figures/period_plots_3")
    for idx in range(len(sip_periods)):
        assert sip_periods.loc[idx, "true_start_time"] == control_periods.loc[idx, "true_start_time"]
        print(f"Starting {sip_periods.loc[idx, 'true_start_time']}")
        period = SIPPeriod(
            control_ncfile_path=Path(control_periods.loc[idx, "file_path"]),  # type: ignore
            sip_ncfile_path=Path(sip_periods.loc[idx, "file_path"]),  # type: ignore
            MIRA_factory=MIRA_dataset_factory,
            start_time=sip_periods.loc[idx, "true_start_time"],  # type: ignore
            end_time=sip_periods.loc[idx, "end_time"],  # type: ignore
        )
        period.plot_refl(save_path=save_path, ymax=10)
        period.plot_icnc(
            save_path=save_path,
            ymax=10,
            plot_hatching=False,
            plot_mulit_agg_contours=True,
            plot_sip_contours=False,
        )
        period.plot_lwc(save_path=save_path, ymax=10)
        period.plot_sip_contours_1(save_path=save_path, ymax=10)
        period.plot_sip_contours_2(save_path=save_path, ymax=10)
        period.plot_sip_rates(save_path=save_path)
        del period

    # period1 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/pois/wrfout_CHPN_d03_2024-11-10_poi_1_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/pois/wrfout_CHPN_d03_2024-11-10_poi_1_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv")),
    #     start_time=pd.Timestamp(year=2024, month=11, day=11),
    #     end_time=pd.Timestamp(year=2024, month=11, day=13),
    # )

    # period2 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/pois/wrfout_CHPN_d03_2024-11-14_poi_3_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/pois/wrfout_CHPN_d03_2024-11-14_poi_3_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv")),
    #     start_time=pd.Timestamp(year=2024, month=11, day=15, hour=6),
    #     end_time=pd.Timestamp(year=2024, month=11, day=16, hour=0),
    # )

    # period3 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv")),
    #     start_time=pd.Timestamp(year=2024, month=11, day=30, hour=0),
    #     end_time=pd.Timestamp(year=2024, month=11, day=30, hour=18),
    # )

    # period4 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv")),
    #     start_time=pd.Timestamp(year=2024, month=12, day=1, hour=12),
    #     end_time=pd.Timestamp(year=2024, month=12, day=2, hour=18),
    # )

    # period5 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-03_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-03_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv")),
    #     start_time=pd.Timestamp(year=2024, month=12, day=5, hour=0),
    #     end_time=pd.Timestamp(year=2024, month=12, day=6, hour=6),
    # )

    # period6 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-13_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-13_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv")),
    #     start_time=pd.Timestamp(year=2024, month=12, day=15, hour=0),
    #     end_time=pd.Timestamp(year=2024, month=12, day=16, hour=0),
    # )

    # save_folder = Path("/home/waseem/CHOPIN_analysis/figures/skewness")
    # period1.plot_icnc_lwc(
    #     save_path=save_folder,
    # )
    # period2.plot_icnc_lwc(save_path=save_folder, ymax=9, temp_level_increment=5)
    # period3.plot_icnc_lwc(save_path=save_folder, ymax=9, temp_level_increment=5)
    # period4.plot_icnc_lwc(save_path=save_folder, ymax=9, temp_level_increment=5)
    # period5.plot_icnc_lwc(save_path=save_folder, ymax=9)
    # period6.plot_icnc_lwc(save_path=save_folder, temp_level_increment=5)

    # %%
    # Plot singleton

    save_path = Path("/home/waseem/CHOPIN_analysis/figures/period_plots_3")
    period1 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-13_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-13_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv")),
        start_time=pd.Timestamp(year=2024, month=12, day=14, hour=18),
        end_time=pd.Timestamp(year=2024, month=12, day=16, hour=0),
        tick_delta=6,
    )

    # period1.plot_refl(ymax=10, save_id="final")
    # period1.plot_icnc_skewness(
    #     ymax=10,
    #     save_id="final",
    #     plot_hatching=False,
    #     plot_mulit_agg_contours=True,
    #     plot_sip_contours=False,
    # )
    # period1.plot_lwc(ymax=10, save_id="final")
    # period1.plot_sip_contours_1(ymax=10, save_id="final")
    # period1.plot_sip_rates()
    # period1.plot_icnc(
    #     plot_hatching=False,
    #     plot_mulit_agg_contours=True,
    #     plot_sip_contours=False,
    # )

    # period1.plot_refl(save_path=save_path, ymax=10, save_id="final")
    # period1.plot_icnc(
    #     save_path=save_path,
    #     ymax=10,
    #     save_id="final",
    #     plot_hatching=False,
    #     plot_mulit_agg_contours=True,
    #     plot_sip_contours=False,
    # )
    # period1.plot_lwc(save_path=save_path, ymax=10, save_id="final")
    # period1.plot_sip_contours_1(save_path=save_path, ymax=10, save_id="final")
    # period1.plot_sip_contours_2(save_path=save_path, ymax=10, save_id="final")
    # period1.plot_sip_rates(save_path=save_path, save_id="final")

    period2 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-22_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-22_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv")),
        start_time=pd.Timestamp(year=2024, month=12, day=23, hour=6),
        end_time=pd.Timestamp(year=2024, month=12, day=24, hour=12),
        tick_delta=6,
    )

    # period2.plot_icnc(
    #     # save_path=save_path,
    #     ymax=10,
    #     save_id="final_hatching",
    #     plot_hatching=True,
    #     plot_mulit_agg_contours=True,
    #     plot_sip_contours=False,
    # )
    # period2.plot_refl(save_path=save_path, ymax=10, save_id="final")
    # period2.plot_icnc(
    #     save_path=save_path,
    #     ymax=10,
    #     save_id="final",
    #     plot_hatching=False,
    #     plot_mulit_agg_contours=True,
    #     plot_sip_contours=False,
    # )
    # period2.plot_icnc(
    #     save_path=save_path,
    #     ymax=10,
    #     save_id="final_hatching",
    #     plot_hatching=True,
    #     plot_mulit_agg_contours=True,
    #     plot_sip_contours=False,
    # )
    # period2.plot_lwc(save_path=save_path, ymax=10, save_id="final")
    # period2.plot_sip_contours_1(save_path=save_path, ymax=10, save_id="final")
    # period2.plot_sip_contours_2(save_path=save_path, ymax=10, save_id="final")
    # period2.plot_sip_rates(save_path=save_path, save_id="final")

    # %%
    hm_exists = np.any(period1.NPRK_sip.hallett_mossop > 0.00001, axis=1)
    ds_exists = np.any(period1.NPRK_sip.droplet_shatter > 0.00001, axis=1)

    print(np.sum(hm_exists))
    print(np.sum(hm_exists) / hm_exists.shape)
    print(np.sum(ds_exists))
    print(np.sum(ds_exists) / ds_exists.shape)
    print()

    hm_exists = np.any(period2.NPRK_sip.hallett_mossop > 0.00001, axis=1)
    ds_exists = np.any(period2.NPRK_sip.droplet_shatter > 0.00001, axis=1)

    print(np.sum(hm_exists))
    print(np.sum(hm_exists) / hm_exists.shape)
    print(np.sum(ds_exists))
    print(np.sum(ds_exists) / ds_exists.shape)

    # %%
    # drop periods without clouds and

    sip_intense = sip_periods.drop([1, 2, 6, 13, 15])
    control_intense = control_periods.drop([1, 2, 6, 13, 15])

    # drop periods without significant SIP
    sip_intense.drop([0, 4, 5], inplace=True)
    control_intense.drop([0, 4, 5], inplace=True)

    # drop duplicate
    sip_intense.drop([7], inplace=True)
    control_intense.drop([7], inplace=True)

    # reset_index
    sip_intense.reset_index(drop=True, inplace=True)
    control_intense.reset_index(drop=True, inplace=True)

    ds = []
    hm = []
    br = []
    subbr = []
    hm_rates = []
    for _, row in sip_intense.iterrows():
        dataset = WRFDataset(ncfile_path=Path(row.file_path))
        hm_rates.append(dataset.hallett_mossop[dataset.hallett_mossop > 0])
        hm_exists = np.any(dataset.hallett_mossop > 0.000001, axis=1)
        ds_exists = np.any(dataset.droplet_shatter > 0.000001, axis=1)
        br_exists = np.any(dataset.breakup > 0.000001, axis=1)
        subbr_exists = np.any(dataset.sublimation_breakup > 0.000001, axis=1)
        hm.append(np.sum(hm_exists) / hm_exists.shape)
        ds.append(np.sum(ds_exists) / ds_exists.shape)
        br.append(np.sum(br_exists) / br_exists.shape)
        subbr.append(np.sum(subbr_exists) / subbr_exists.shape)

    print(np.mean(hm))
    print(np.mean(ds))
    print(np.mean(br))
    print(np.mean(subbr))
