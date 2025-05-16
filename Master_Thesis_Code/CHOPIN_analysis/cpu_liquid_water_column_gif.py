import concurrent.futures
import glob
import logging
import sys
import time
from pathlib import Path

import cartopy.crs as crs
import matplotlib.pyplot as plt
import numpy as np
import tyro
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset  # type: ignore
from PIL import Image
from wrf import cartopy_xlim, cartopy_ylim, get_cartopy, getvar, latlon_coords, to_np


def get_image_number(png_path: str) -> int:
    return int(png_path.split("_")[-1].split(".")[0])


def make_gif(gif_folder: Path, save_path: Path):

    glob_string = str(gif_folder.absolute()) + "/*.png"
    files = glob.glob(glob_string)
    images = {}
    for file in files:
        image_number = get_image_number(file)
        images[image_number] = Image.open(file)

    image_array = []
    min_image = np.min(list(images.keys()))
    max_image = np.max(list(images.keys()))
    for i in range(min_image, max_image):
        try:
            image_array.append(images[i])
        except:
            continue

    image_array[0].save(
        str(save_path),
        format="GIF",
        append_images=image_array[0:],
        save_all=True,
        duration=200,
        loop=0,
        optimize=False,
    )


def get_LWP(ncfile, timestep):
    ### WRF constants
    RA = 287.15
    RD = 287.0
    CP = 1004.5
    P1000MB = 100000.0
    EPS = 0.622
    pres = getvar(ncfile, "P", timeidx=timestep, meta=False) + getvar(
        ncfile, "PB", timeidx=timestep, meta=False
    )
    thet = getvar(ncfile, "T", timeidx=timestep, meta=False) + 300.0
    qv = getvar(ncfile, "QVAPOR", timeidx=timestep, meta=False)
    tk = (pres / P1000MB) ** (RD / CP) * thet
    tv = tk * (EPS + qv) / (EPS * (1.0 + qv))
    rho = pres / RA / tv
    lwc = (
        (
            getvar(ncfile, "QCLOUD", timeidx=timestep, meta=False)
            + getvar(ncfile, "QRAIN", timeidx=timestep, meta=False)
        )
        * rho
        * 10**3
    )

    zstag = getvar(ncfile, "zstag", timeidx=timestep, meta=False)
    dz = np.diff(zstag, axis=0)
    lwc_arr = to_np(lwc)
    return np.sum(lwc_arr * dz, axis=0)


def plot_liquid_water_path(ncfile, cart_proj, slp, save_path, timeidx, time_string, lats, lons):
    logging.info(f"Plotting: {timeidx}")
    column_LWP = get_LWP(ncfile, timeidx)
    column_LWP[column_LWP < 10**-5] = np.nan
    # Create a figure
    fig = plt.figure(figsize=(7, 6))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)

    coast = NaturalEarthFeature(category="physical", scale="10m", facecolor="none", name="coastline")
    ax.add_feature(
        coast,
        linewidth=1,
        edgecolor="black",
    )

    levels = np.linspace(-4, 3, 15)
    cont0 = plt.contourf(
        to_np(lons), to_np(lats), np.log10(column_LWP).T, levels=levels, transform=crs.PlateCarree(), cmap="jet"
    )
    # Add a color bar
    cb = plt.colorbar(cont0)
    cb.set_label("log[g/m3]")

    # Add the gridlines
    ax.gridlines(color="black", linestyle="dotted")
    plt.scatter(167, 153, color="k", marker="*", s=60)

    plt.title(f"LWP {time_string}")
    # Set the map bounds
    ax.set_xlim(cartopy_xlim(slp))
    ax.set_ylim(cartopy_ylim(slp))
    plt.tight_layout()
    plot_name = f"LWP_gif_{timeidx:03}.png"
    plt.savefig(str(Path(save_path, plot_name)))
    logging.info(f"Saved {str(Path(save_path, plot_name))}")
    plt.close()

def main(file_path: Path, temp_path: Path, gif_save_path: Path):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("Running Main")

    temp_path.mkdir(exist_ok=True)

    ncfile = Dataset(str(file_path))

    wrf_times = getvar(ncfile, "times", timeidx=None, meta=False)
    wrf_times = np.datetime_as_string(wrf_times, unit="m")

    slp = getvar(ncfile, "slp")
    lats, lons = latlon_coords(slp, as_np=True)

    start_time_idx = int(24 * 60 / 5)  # hours * minutes/hour / minute/reading
    end_time_idx = wrf_times.shape[0]
    cart_proj = get_cartopy(slp)

    for timeidx in range(start_time_idx, end_time_idx, 2):
        plot_liquid_water_path(
                time_string=wrf_times[timeidx],
                timeidx=timeidx,
                ncfile=ncfile,
                save_path=str(temp_path),
                lats=lats,
                lons=lons,
                slp=slp,
                cart_proj=cart_proj
            )

    logging.info("Creating gif...")
    make_gif(temp_path, gif_save_path)


tyro.cli(main)
