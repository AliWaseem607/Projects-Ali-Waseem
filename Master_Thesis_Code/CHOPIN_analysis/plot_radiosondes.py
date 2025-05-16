#%%
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import img_tiles
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes._axes import Axes
from matplotlib.pyplot import Circle
from netCDF4 import Dataset
from wrf import destagger, getvar, rh

sys.path.append("./")
from scalebar import scale_bar


def plot_radiosonde_temp_comparison(ax: Axes, radiosonde: pd.DataFrame, wrf: bool = True, ERA5: bool = True):
    im0 = ax.scatter(
        radiosonde["T"],
        radiosonde.geopot_height / 1000,
        c=radiosonde.fly_time / 60,
        cmap="jet",
        s=1,
        label="Radiosonde",
    )
    if wrf:
        ax.plot(
            radiosonde["wrf_T"],
            radiosonde.geopot_height / 1000,
            color="saddlebrown",
            label="WRF Simulation",
            linestyle="-",
        )
    if ERA5:
        ax.plot(
            radiosonde["ERA5_T"],
            radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )
    cbar = plt.colorbar(im0)
    cbar.set_label("Fly time [min]")
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Geopotential Height [km]")


def plot_radiosonde_RH_comparison(ax: Axes, radiosonde: pd.DataFrame, wrf: bool = True, ERA5: bool = True):
    im0 = ax.scatter(
        radiosonde["RH"],
        radiosonde.geopot_height / 1000,
        c=radiosonde.fly_time / 60,
        cmap="jet",
        s=1,
        label="Radiosonde",
    )
    if wrf:
        ax.plot(
            radiosonde["wrf_RH"],
            radiosonde.geopot_height / 1000,
            color="saddlebrown",
            label="WRF Simulation",
            linestyle="-",
        )
    if ERA5:
        ax.plot(
            radiosonde["ERA5_RH"],
            radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    cbar = plt.colorbar(im0)
    cbar.set_label("Fly time [min]")
    ax.legend()
    ax.set_xlabel("Relative Humidity [%]")
    ax.set_ylabel("Geopotential Height [km]")
    ax.set_xlim(-5,105)


def plot_radiosonde_wind_speed_comparison(
    ax: Axes, radiosonde: pd.DataFrame, wrf: bool = True, ERA5: bool = True
):
    im0 = ax.scatter(
        radiosonde["wind_speed"],
        radiosonde.geopot_height / 1000,
        c=radiosonde.fly_time / 60,
        cmap="jet",
        s=1,
        label="Radiosonde",
    )
    if wrf:
        ax.plot(
            radiosonde["wrf_wind_speed"],
            radiosonde.geopot_height / 1000,
            color="saddlebrown",
            label="WRF Simulation",
            linestyle="-",
        )
    if ERA5:
        ax.plot(
            radiosonde["ERA5_wind_speed"],
            radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    cbar = plt.colorbar(im0)
    cbar.set_label("Fly time [min]")
    ax.legend()
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Geopotential Height [km]")


def plot_radiosonde_wind_dir_comparison(
    ax: Axes, radiosonde: pd.DataFrame, fontsize: float = 11, wrf: bool = True, ERA5: bool = True
):
    max_range = radiosonde.geopot_height.max() / 1000
    plot_range = int(np.ceil(max_range))
    ax.set_ylim(-plot_range, plot_range)
    ax.set_xlim(-plot_range, plot_range)
    ax.axvline(0, ymin=-plot_range, ymax=plot_range, color="k", alpha=0.75)
    ax.axhline(0, xmin=-plot_range, xmax=plot_range, color="k", alpha=0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    last_tic_range = np.floor(plot_range / 2.5) * 2.5
    tic_steps = int((last_tic_range - 2.5) / 2.5 + 1)
    circles_radi = np.linspace(2.5, last_tic_range, tic_steps)
    circles = [Circle((0, 0), x, fill=False, color="grey", alpha=0.75) for x in circles_radi]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for circ in circles:
        ax.add_patch(circ)

    ax.add_patch(Circle((0, 0), plot_range, fill=False, color="k"))
    ax.set_yticks(-circles_radi, [str(x) for x in circles_radi])

    ax.text(-0.75, plot_range + 1, "N", fontdict={"size": fontsize})
    ax.text(-0.75, -plot_range - 2, "S", fontdict={"size": fontsize})

    ax.text(plot_range + 0.5, -1, "E", fontdict={"size": fontsize})
    ax.text(-plot_range - 3, -1, "W", fontdict={"size": fontsize})

    radiosonde_x = np.cos(radiosonde.wind_dir / 180 * np.pi)
    radiosonde_y = np.sin(radiosonde.wind_dir / 180 * np.pi)
    im0 = ax.scatter(
        radiosonde_x * radiosonde.geopot_height / 1000,
        radiosonde_y * radiosonde.geopot_height / 1000,
        c=radiosonde.fly_time / 60,
        cmap="jet",
        s=1,
        label="Radiosonde",
    )
    if wrf:
        wrf_wind_dir = radiosonde.wrf_wind_dir.to_numpy(dtype="float") / 180 * np.pi
        wrf_x = np.cos(wrf_wind_dir)
        wrf_y = np.sin(wrf_wind_dir)
        ax.plot(
            wrf_x * radiosonde.geopot_height / 1000,
            wrf_y * radiosonde.geopot_height / 1000,
            color="saddlebrown",
            label="WRF Simulation",
            linestyle="-",
        )

    if ERA5:
        era5_wind_dir = radiosonde.ERA5_wind_dir.to_numpy(dtype="float") / 180 * np.pi
        era5_x = np.cos(era5_wind_dir)
        era5_y = np.sin(era5_wind_dir)
        ax.plot(
            era5_x * radiosonde.geopot_height / 1000,
            era5_y * radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    cbar = plt.colorbar(im0)
    cbar.set_label("Flight Time [min]")
    ax.set_ylabel("Radius is Geopotential Height [km]")
    ax.set_xlabel("Wind Direction", labelpad=30)
    ax.legend()


def plot_radiosonde_flight_path(ax: GeoAxes, radiosonde: pd.DataFrame):
    ax.scatter(radiosonde.lon, radiosonde.lat, c=radiosonde.fly_time, cmap="jet", s=0.5)
    ax.scatter(
        radiosonde.iloc[0]["lon"], radiosonde.iloc[0]["lat"], marker="s", color="blue", s=45, label="Start"
    )
    ax.scatter(
        radiosonde.iloc[-1]["lon"], radiosonde.iloc[-1]["lat"], marker="o", color="red", s=45, label="End"
    )
    ax.legend()
    buffer = 0.05
    min_lat = radiosonde.lat.min() - buffer / 2
    max_lat = radiosonde.lat.max() + buffer / 2
    min_lon = radiosonde.lon.min() - buffer
    max_lon = radiosonde.lon.max() + buffer
    tiler = img_tiles.GoogleTiles()
    dlat = max_lat - min_lat
    dlon = max_lon - min_lon
    min_axis = np.min([dlat, dlon])
    zoom_level = 10
    if min_axis < 0.2:
        zoom_level = 11
    if min_axis < 0.15:
        zoom_level = 12
    ax.add_image(tiler, zoom_level)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_title("Flight Path")


def plot_radiosonde(df:pd.DataFrame, bl:int, sip: int, save=True):
    
        fig = plt.figure(figsize=(9, 10.5))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())
        plot_radiosonde_temp_comparison(ax1, df, ERA5=False)
        plot_radiosonde_RH_comparison(ax2, df, ERA5=False)
        plot_radiosonde_wind_speed_comparison(ax3, df, ERA5=False)
        plot_radiosonde_wind_dir_comparison(ax4, df, ERA5=False)
        plot_radiosonde_flight_path(ax5, df)
        scale_bar(ax5, location=(0.2, 0.05))

        start_time = df.time.min()
        if sip == 0:
            ver = "CONTROL"
        elif sip == 1:
            ver = "SIP"
        else:
            raise RuntimeError("incorrect SIP value")
        plt.tight_layout()
        fig.suptitle(f"{start_time.strftime(r'%Y%m%d %H')} bl: {bl} {ver}", y=1.01)
        if save:
            save_name = f"radiosonde_plot_{start_time.strftime(r'%Y%m%d %H')}_bl_{bl}.png"
            plt.savefig(f"figures/radiosonde_comparison/{save_name}", bbox_inches="tight")
            plt.close()

# %%
# plt.rcdefaults()
legend_size = 9
small_size = 10
medium_size = 11
large_size = 14
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=legend_size)  # legend fontsize
plt.rc("figure", titlesize=large_size)  # fontsize of the figure title

metadata = pd.read_csv("./data/metadata_radiosondes.csv")

for _, row in metadata.iterrows():
    df = pd.read_csv(row.file_path, parse_dates=["time"])
    plot_radiosonde(df, bl=row.bl, sip=row.sip, save=True)