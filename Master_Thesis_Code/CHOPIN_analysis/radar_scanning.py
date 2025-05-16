# %%
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import MIRAMultiDataset, MIRAMultiDatasetFactory

# %%


def plot_regular(
    start_time: pd.Timestamp, end_time: pd.Timestamp, MIRA_factory: MIRAMultiDatasetFactory, save: bool = True
):
    total_time = end_time - start_time
    plot_time = pd.Timedelta(2, "d")
    for i in range(int(np.ceil(total_time / plot_time))):
        plot_start = start_time + plot_time * i
        plot_end = np.min([end_time, start_time + plot_time * (i + 1)])  # type: ignore
        MIRA = MIRA_factory.get_dataset(start_time=plot_start, end_time=plot_end + pd.Timedelta(1, "h"))

        if save:
            if Path(
                f"./figures/radar_scanning/MIRA_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}.png"
            ).exists():
                continue

        tick_locs = mdates.drange(plot_start, plot_end + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
        tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

        MIRA_mask = (MIRA.times >= plot_start) & (MIRA.times <= plot_end)
        if np.all(MIRA_mask == False):
            continue
        plt.figure(figsize=(10, 3))
        im0 = plt.pcolormesh(
            MIRA.times[MIRA_mask][::10],
            MIRA.range / 1000,
            MIRA.refl[MIRA_mask][::10, :].T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBZ]")
        plt.ylim(0, 12)
        plt.ylabel("Altitude [km]")
        plt.title("MIRA")
        plt.xticks(tick_locs, tick_labels)

        if save:
            plt.savefig(
                f"./figures/radar_scanning/MIRA_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}"
            )
        plt.close()


def plot_cross(start_time, end_time, MIRA_factory: MIRAMultiDatasetFactory, save: bool = True):
    total_time = end_time - start_time
    plot_time = pd.Timedelta(2, "d")
    for i in range(int(np.ceil(total_time / plot_time))):

        plot_start = start_time + plot_time * i
        plot_end = np.min([end_time, start_time + plot_time * (i + 1)])  # type: ignore
        MIRA = MIRA_factory.get_dataset(start_time=plot_start, end_time=plot_end + pd.Timedelta(1, "h"))

        if save:
            if Path(
                f"./figures/radar_scanning_cross/MIRA_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}.png"
            ).exists():
                continue

        tick_locs = mdates.drange(plot_start, plot_end + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
        tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

        MIRA_mask = (MIRA.times >= plot_start) & (MIRA.times <= plot_end)
        if np.all(MIRA_mask == False):
            continue
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        im0 = ax[0].pcolormesh(
            MIRA.times[MIRA_mask],
            MIRA.range / 1000,
            MIRA.refl[MIRA_mask].T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBZ]")
        ax[0].set_ylim(0, 12)
        ax[0].set_ylabel("Altitude [km]")
        ax[0].set_title("MIRA Reflectivitsy")

        im0 = ax[1].pcolormesh(
            MIRA.times[MIRA_mask],
            MIRA.range / 1000,
            MIRA.cross_refl[MIRA_mask].T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBZ]")
        ax[1].set_ylim(0, 12)
        ax[1].set_ylabel("Altitude [km]")
        ax[1].set_title(f"MIRA Cross Reflectivity")

        set_up_dated_x_axis(ax, tick_locs, tick_labels)
        if save:
            plt.savefig(
                f"./figures/radar_scanning_cross/MIRA_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}.png"
            )
        plt.close()


def plot_clean(start_time, end_time, MIRA_factory: MIRAMultiDatasetFactory, save: bool = True, mode:int=1):
    total_time = end_time - start_time
    plot_time = pd.Timedelta(2, "d")
    for i in range(int(np.ceil(total_time / plot_time))):

        plot_start = start_time + plot_time * i
        plot_end = np.min([end_time, start_time + plot_time * (i + 1)])  # type: ignore
        MIRA = MIRA_factory.get_dataset(start_time=plot_start, end_time=plot_end + pd.Timedelta(1, "h"))
        assert MIRA.clean_mask is not None

        tick_locs = mdates.drange(plot_start, plot_end + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
        tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

        MIRA_mask = (MIRA.times >= plot_start) & (MIRA.times <= plot_end)
        if np.all(MIRA_mask == False):
            continue
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        im0 = ax[0].pcolormesh(
            MIRA.times[MIRA_mask],
            MIRA.range / 1000,
            MIRA.refl[MIRA_mask].T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBZ]")
        ax[0].set_ylim(0, 12)
        ax[0].set_ylabel("Altitude [km]")
        ax[0].set_title("MIRA")

        clean_refl = MIRA.refl[MIRA_mask].copy()
        clean_mask = MIRA.clean_mask[MIRA_mask]
        clean_refl[clean_mask] = np.nan

        im0 = ax[1].pcolormesh(
            MIRA.times[MIRA_mask],
            MIRA.range / 1000,
            clean_refl.T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Reflectivity [dBZ]")
        ax[1].set_ylim(0, 12)
        ax[1].set_ylabel("Altitude [km]")
        if mode == 1:
            ax[1].set_title(f"MIRA cleaned: {np.sum(clean_mask)} points removed")
        if mode == 2:
            ax[1].set_title(f"MIRA cleaned")

        if mode ==1:
            clean_refl[:, :] = np.nan
            clean_refl[clean_mask] = 100
            im0 = ax[1].pcolormesh(
                MIRA.times[MIRA_mask],
                MIRA.range / 1000,
                clean_refl.T,
                vmin=-60,
                vmax=35,
                cmap="jet",
            )

        set_up_dated_x_axis(ax, tick_locs, tick_labels)
        if save:
            file_name = f"MIRA_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}.png"
            if mode == 1:
                plt.savefig(
                    f"./figures/radar_clean_comparison_diff/{file_name}", bbox_inches="tight", dpi=200
                )
            if mode ==2:
                plt.savefig(
                    f"./figures/radar_clean_comparison/{file_name}", bbox_inches="tight", dpi=200
                )
        plt.close()


def main(mode: str = "regular"):
    start_time = pd.Timestamp(year=2024, month=10, day=12)
    end_time = pd.Timestamp(year=2024, month=12, day=30)

    if mode == "regular":
        MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv"))
        plot_regular(start_time=start_time, end_time=end_time, MIRA_factory=MIRA_factory)
    elif mode == "cross":
        MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv"))
        plot_cross(start_time=start_time, end_time=end_time, MIRA_factory=MIRA_factory)
    elif mode == "clean":
        MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
        plot_clean(start_time=start_time, end_time=end_time, MIRA_factory=MIRA_factory)
    elif mode == "clean_2":
        MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
        plot_clean(start_time=start_time, end_time=end_time, MIRA_factory=MIRA_factory, mode=2)
    else:
        raise NotImplementedError(f"{mode} has not been implemented")


# %%


# %%

if __name__ == "__main__":
    tyro.cli(main)
