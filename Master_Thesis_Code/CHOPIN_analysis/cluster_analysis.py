# %%
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import (
    MIRAMultiDataset,
    MIRAMultiDatasetFactory,
    WRFMultiDatasetFactory,
)

# %%
mapping_21 = {5: 0, 2: 1, 4: 2, 0: 3, 1: 4, 3: 5}

clusters = pd.read_csv("./figures/cluster_analysis_21/data.csv", parse_dates=["start_time", "end_time"])
label_mapping = mapping_21
clusters["label_standardized"] = -1
for key, val in label_mapping.items():
    clusters.loc[clusters.label == key, "label_standardized"] = val

clusters.sort_values("start_time", inplace=True)
clusters.reset_index(inplace=True, drop=True)
clusters.drop("Unnamed: 0", axis=1, inplace=True)

wrf_dataset_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))
MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
# %%
# # # start = pd.Timestamp(year=2024, month=10, day=21, hour=12)
# # # end = pd.Timestamp(year=2024, month=10, day=22)
# # # wrf_dataset_factory.get_setups(start, end)


# # # # %%
# # # start_time = clusters.loc[0, "start_time"]
# # # end_time = clusters.loc[0, "end_time"]
# # # hours = 0
# # # for idx in range(len(clusters) - 1):
# # #     groups = set(wrf_dataset_factory.get_setups(start_time, end_time))

# # #     row_cluster = clusters.loc[idx, "label_standardized"]
# # #     row_start_time = clusters.loc[idx, "start_time"]
# # #     row_end_time = clusters.loc[idx, "end_time"]

# # #     next_row_cluster = clusters.loc[idx + 1, "label_standardized"]
# # #     next_row_start_time = clusters.loc[idx + 1, "start_time"]
# # #     next_row_end_time = clusters.loc[idx + 1, "end_time"]

# # #     print(idx)
# # #     print(len(groups))
# # #     if len(groups) == 0:
# # #         start_time = next_row_start_time
# # #         end_time = next_row_end_time
# # #         continue

# # #     print(row_cluster)
# # #     print(next)
# # #     print(groups)
# # #     print(set(wrf_dataset_factory.get_setups(start_time, next_row_end_time)))
# # #     print(groups == set(wrf_dataset_factory.get_setups(start_time, next_row_end_time)))
# # #     print()
# # #     if (row_cluster == next_row_cluster) & (
# # #         groups == set(wrf_dataset_factory.get_setups(start_time, next_row_end_time))
# # #     ):
# # #         hours += (next_row_start_time - row_start_time).total_seconds() / 3600
# # #         end_time = next_row_end_time
# # #         if hours < 40:
# # #             continue

# # #     print("plotting:")
# # #     print(start_time, end_time)

# # #     total_hours = (end_time - start_time).total_seconds() / 3600

# # #     interval = 3
# # #     if total_hours > 12:
# # #         interval = 6
# # #     if total_hours > 24:
# # #         interval = 12

# # #     tick_locs = mdates.drange(start_time, end_time + pd.Timedelta(1, "h"), pd.Timedelta(interval, "hours"))
# # #     tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

# # #     MIRA = MIRA_factory.get_dataset(start_time, end_time)
# # #     fig, axs = plt.subplots(len(groups) + 1, 1, figsize=(6, 2 * (len(groups) + 1)))
# # #     i = 0

# # #     for group in groups:
# # #         print(group)
# # #         mp, bl, sip = group
# # #         wrf_dataset = wrf_dataset_factory.get_dataset(
# # #             start_time, end_time, station="NPRK", mp_phys=mp, bl_phys=bl, sip=sip
# # #         )

# # #         temp_levels = np.arange(-80, 10, 10)

# # #         cs = axs[i].contour(
# # #             wrf_dataset.times,
# # #             wrf_dataset.ZZ / 1000,
# # #             (wrf_dataset.kinetic_temp.T - 273.15),
# # #             levels=temp_levels,
# # #             colors="dimgray",
# # #             linewidths=1,
# # #             alpha=0.8,
# # #         )
# # #         axs[i].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

# # #         im1 = axs[i].pcolormesh(
# # #             wrf_dataset.times,
# # #             wrf_dataset.ZZ / 1000,
# # #             wrf_dataset.variables("Zhh_MIRA").T,
# # #             vmin=-60,
# # #             vmax=35,
# # #             cmap="jet",
# # #         )
# # #         cbar = fig.colorbar(im1)
# # #         cbar.set_label("Reflectivity [dBz]")

# # #         axs[i].set_ylim(0, 12)
# # #         axs[i].set_ylabel("Altitude [km]")
# # #         axs[i].set_title(f"mp: {mp}, bl: {bl}, sip:{sip}")

# # #         i += 1

# # #     cs = axs[i].contour(
# # #         wrf_dataset.times,
# # #         wrf_dataset.ZZ / 1000,
# # #         (wrf_dataset.kinetic_temp.T - 273.15),
# # #         levels=temp_levels,
# # #         colors="dimgray",
# # #         linewidths=1,
# # #         alpha=0.8,
# # #     )
# # #     axs[i].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

# # #     elv = MIRA.variables("elv")
# # #     elv_mask = elv > 85
# # #     MIRA.refl[MIRA.clean_mask] = np.nan

# # #     im0 = axs[i].pcolormesh(
# # #         MIRA.times[elv_mask],
# # #         MIRA.range / 1000,
# # #         MIRA.refl[elv_mask].T,
# # #         vmin=-60,
# # #         vmax=35,
# # #         cmap="jet",
# # #     )
# # #     cbar = plt.colorbar(im0)
# # #     cbar.set_label("Reflectivity [dBz]")
# # #     axs[i].set_ylim(0, 12)
# # #     axs[i].set_ylabel("Altitude [km]")
# # #     axs[i].set_title("MIRA")

# # #     set_up_dated_x_axis(axs, tick_locs, tick_labels)
# # #     plt.tight_layout()
# # #     plt.savefig(
# # #         f"./figures/cluster_plots_2/refl_cluster_{row_cluster}_{start_time.strftime(r'%Y%m%dH%H')}-{end_time.strftime(r'%Y%m%dH%H')}.png"
# # #     )
# # #     plt.close()

# # #     print("----------------------------")
# # #     print()
# # #     start_time = next_row_start_time
# # #     end_time = next_row_end_time
# # #     hours = 0


# %%
start_time = clusters.loc[0, "start_time"]
end_time = clusters.loc[0, "end_time"]
for i in range(len(clusters)):
    start_time = clusters.loc[i, "start_time"]
    end_time = clusters.loc[i, "end_time"]
    groups = set(wrf_dataset_factory.get_setups(start_time, end_time))
    tick_locs = mdates.drange(start_time, end_time + pd.Timedelta(1, "h"), pd.Timedelta(3, "hours"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]
    row_cluster = clusters.loc[i, "label_standardized"]
    for group in groups:
        mp, bl, sip = group
        if sip == False:
            continue

        print(start_time)

        fig, axs = plt.subplots(2, 1)


        wrf_dataset = wrf_dataset_factory.get_dataset(
            start_time, end_time, station="NPRK", mp_phys=mp, bl_phys=bl, sip=sip
        )
        MIRA = MIRA_factory.get_dataset(start_time, end_time)

        temp_levels = np.arange(-80, 10, 10)

        cs = axs[0].contour(
            wrf_dataset.times,
            wrf_dataset.ZZ / 1000,
            (wrf_dataset.kinetic_temp.T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=0.8,
        )
        axs[0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

        MP = (wrf_dataset.icnc>0.01) & (~np.isnan(wrf_dataset.lwc))
        MP = MP.astype(float)
        MP[MP==1] = 20
        MP[MP==0] = np.nan
        im1 = axs[0].pcolormesh(
            wrf_dataset.times,
            wrf_dataset.ZZ / 1000,
            MP.T,
            vmin=0,
            vmax=1,
            cmap="jet",
        )
        cbar = fig.colorbar(im1)
        cbar.set_label("Reflectivity [dBz]")

        axs[0].set_ylim(0, 12)
        axs[0].set_ylabel("Altitude [km]") 
        axs[0].set_title(f"mp: {mp}, bl: {bl}, sip:{sip}")

        cs = axs[1].contour(
            wrf_dataset.times,
            wrf_dataset.ZZ / 1000,
            (wrf_dataset.kinetic_temp.T - 273.15),
            levels=temp_levels,
            colors="dimgray",
            linewidths=1,
            alpha=0.8,
        )
        axs[1].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")

        elv = MIRA.variables("elv")
        elv_mask = elv > 85
        MIRA.refl[MIRA.clean_mask] = np.nan

        im0 = axs[1].pcolormesh(
            MIRA.times[elv_mask],
            MIRA.range / 1000,
            MIRA.refl[elv_mask].T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBz]")
        axs[1].set_ylim(0, 12)
        axs[1].set_ylabel("Altitude [km]")
        axs[1].set_title("MIRA")
        plt.tight_layout()
        plt.savefig(f"./figures/last_minute/MP_{row_cluster}_{start_time.strftime(r'%Y%m%dH%H')}-{end_time.strftime(r'%Y%m%dH%H')}.png")
        plt.close("all")
        del MIRA
        del wrf_dataset
        del MP
        
    


# %%

# # # legend_size = 13
# # # small_size = 12
# # # medium_size = 13
# # # large_size = 15
# # # plt.rc("font", size=small_size)  # controls default text sizes
# # # plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
# # # plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
# # # plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
# # # plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
# # # plt.rc("legend", fontsize=legend_size)  # legend fontsize
# # # plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
# # # plt.rc("axes", titlesize=medium_size)


# # # def plot_period(MIRA_factory: MIRAMultiDatasetFactory, start_time, end_time, ax: Axes, title):

# # #     MIRA = MIRA_factory.get_dataset(start_time, end_time)

# # #     tick_locs = mdates.drange(start_time, end_time + pd.Timedelta(1, "h"), pd.Timedelta(4, "hours"))
# # #     tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

# # #     if MIRA.clean_mask is not None:
# # #         MIRA.refl[MIRA.clean_mask] = np.nan
# # #     elv = MIRA.variables("elv")
# # #     elv_mask = elv > 85

# # #     im0 = ax.pcolormesh(
# # #         MIRA.times[elv_mask],
# # #         MIRA.range / 1000,
# # #         MIRA.refl[elv_mask, :].T,
# # #         vmin=-60,
# # #         vmax=35,
# # #         cmap="jet",
# # #     )
# # #     cbar = plt.colorbar(im0, ticks=[-60, -40, -20, 0, 20])
# # #     cbar.set_label("Refl. [dBz]")
# # #     ax.set_ylim(0, 10)
# # #     ax.set_yticks([0, 2, 4, 6, 8, 10])
# # #     # ax.set_ylabel("Altitude [km]")
# # #     ax.set_xticks(tick_locs, tick_labels)
# # #     ax.set_title(title)


# # # fig, axs = plt.subplots(5, 2, figsize=(9, 10))
# # # MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))

# # # start = pd.Timestamp(year=2024, month=12, day=11, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[0, 0], "Cluster 2")

# # # start = pd.Timestamp(year=2024, month=10, day=28, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[0, 1], "Cluster 2")

# # # start = pd.Timestamp(year=2024, month=12, day=19, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[1, 0], "Cluster 3")

# # # start = pd.Timestamp(year=2024, month=12, day=19, hour=12)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[1, 1], "Cluster 3")

# # # start = pd.Timestamp(year=2024, month=10, day=12, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[2, 0], "Cluster 4")

# # # start = pd.Timestamp(year=2024, month=12, day=6, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[2, 1], "Cluster 4")

# # # start = pd.Timestamp(year=2024, month=12, day=24, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[3, 0], "Cluster 5")

# # # start = pd.Timestamp(year=2024, month=11, day=20, hour=0)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[3, 1], "Cluster 5")

# # # start = pd.Timestamp(year=2024, month=12, day=1, hour=12)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[4, 0], "Cluster 6")

# # # start = pd.Timestamp(year=2024, month=11, day=12, hour=12)
# # # plot_period(MIRA_factory, start, start + pd.Timedelta(12, "h"), axs[4, 1], "Cluster 6")

# # # fig.supylabel("Altitude [km]")
# # # fig.supxlabel("Date [UTC]")
# # # plt.tight_layout()

# # # plt.savefig("./figures/final_plots/cluster_samples.png", dpi=300, bbox_inches="tight")
