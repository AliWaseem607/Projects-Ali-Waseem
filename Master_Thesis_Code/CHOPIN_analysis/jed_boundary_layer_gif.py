import concurrent.futures
import glob
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from netCDF4 import Dataset  # type: ignore
from PIL import Image
from wrf import CoordPair, getvar, interpline, to_np, vertcross


def get_image_number(png_path: str) -> int:
    return int(png_path.split("_")[-1].split(".")[0])

def make_gif(gif_folder:Path, save_path:Path):
     
    glob_string = str(gif_folder.absolute()) +"/*.png"
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

    image_array[0].save(str(save_path), format='GIF', 
               append_images=image_array[0:], save_all=True, duration=200, 
               loop=0, optimize=False)


def plot_cross_section_load_one(pblh:np.ndarray, height_line:np.ndarray, time_string:str, timeidx:int, ncfile_path:str, z:np.ndarray, save_path:str):
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()
    logging.info(f"Plotting: {timeidx}")
    ncfile = Dataset(ncfile_path)
    start_point = CoordPair(lat=38.173750, lon=21.512966)
    end_point = CoordPair(lat=37.841686, lon=22.779232)

    wspd = getvar(ncfile, "uvmet_wspd_wdir", timeidx=timeidx)[0,:]
    # Compute the vertical cross-section interpolation.  Also, include the
    # lat/lon points along the cross-section.
    wspd_cross = vertcross(wspd, z, wrfin=ncfile, start_point=start_point,
                        end_point=end_point, latlon=True, meta=True)


    # Make the contour plot
    ax.set_facecolor("black")
    wspd_contours = ax.contourf(to_np(wspd_cross), cmap="viridis")

    # Add the color bar
    cbar = plt.colorbar(wspd_contours, ax=ax)
    cbar.set_label("wind speed m/s")
    # Set the x-ticks to use latitude and longitude labels.
    coord_pairs = to_np(wspd_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [pair.latlon_str(fmt="{:.2f}, {:.2f}")
                for pair in to_np(coord_pairs)]
    ax.set_xticks(x_ticks[::20])
    ax.set_xticklabels(x_labels[::20], rotation=45, fontsize=8)

    # Set the y-ticks to be height.
    vert_vals = to_np(wspd_cross.coords["vertical"])
    v_ticks = np.arange(vert_vals.shape[0])
    ax.set_yticks(v_ticks[::4])
    ax.set_yticklabels(vert_vals[::4], fontsize=8)

    # Set the x-axis and  y-axis labels
    ax.set_xlabel("Latitude, Longitude", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    pbhl_line_graph_space = np.interp(pblh, to_np(wspd_cross.coords["vertical"]), np.arange(100))
    height_line_graph_space = np.interp(height_line, to_np(wspd_cross.coords["vertical"]), np.arange(100))
    ax.plot(pbhl_line_graph_space, color='magenta', linewidth=2.5, label="PBLH")
    # ax.plot(height_line_graph_space, color="orange", linewidth=2)
    ax.fill_between(np.arange(height_line.shape[0]), height_line_graph_space, color="saddlebrown", label="Terrain")
    plt.legend()
    plt.title(time_string)
    ax.set_ylim(0,20)

    plot_name = f"PBLH_gif_{timeidx:03}.png"
    plt.savefig(str(Path(save_path, plot_name)))
    logging.info(f"Saved {str(Path(save_path, plot_name))}")
    plt.close()

def main(file_path: Path, temp_path: Path, gif_save_path:Path, cpus: int)-> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("Running Main")
    
    temp_path.mkdir(exist_ok=True)

    ncfile = Dataset(str(file_path))

    pblh_arr = getvar(ncfile, "PBLH", timeidx = None,) + getvar(ncfile, "HGT", timeidx=None)
    z = getvar(ncfile, "z", units="m")
    height_arr = getvar(ncfile, "HGT")
    wrf_times = getvar(ncfile, "times", timeidx=None, meta=False)
    wrf_times = np.datetime_as_string(wrf_times, unit="m")


    start_idx = int(24*60/5) #hours * minutes/hour / minute/reading

    end_time_idx = pblh_arr.shape[0]

    start_point = CoordPair(lat=38.173750, lon=21.512966)
    end_point = CoordPair(lat=37.841686, lon=22.779232)

    pblh_line = interpline(pblh_arr, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True)
    height_line = interpline(height_arr, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True)

    pblh_line_np = to_np(pblh_line)
    height_line_np = to_np(height_line)
    z_np = to_np(z)

    del pblh_arr
    del height_arr
    logging.info("Starting ProcessPool")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=cpus)

    futures = []
    logging.info(f"Submitting jobs {start_idx} to {end_time_idx}")
    for timeidx in range(start_idx, end_time_idx, 2):
        
        futures.append(
            executor.submit(
                plot_cross_section_load_one, 
                pblh=pblh_line_np[timeidx,:], 
                height_line=height_line_np, 
                time_string=wrf_times[timeidx], 
                timeidx=timeidx,
                ncfile_path=str(file_path), 
                z=z_np, 
                save_path=str(temp_path)
            )
        )
    
    
    
    logging.info("Shut down called...")
    executor.shutdown(wait=True, cancel_futures=False)
    logging.info("Shutting down...")

    logging.info("Checking all futures done and no exceptions raised...")
    continue_process = True
    for future in futures:
        if future.done() == False:
            print("Something is wrong")
            continue_process = False
        if future.exception() != None:
            print("exception was raised")
            continue_process = False
            
    if continue_process is True:
        logging.info("Creating gif...")
        make_gif(temp_path, gif_save_path)

tyro.cli(main)