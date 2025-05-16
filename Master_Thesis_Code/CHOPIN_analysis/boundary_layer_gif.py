from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # type: ignore
from wrf import CoordPair, getvar, interpline, to_np, vertcross

save_path = Path("figures/PBLH_gif/")
save_path.mkdir(exist_ok=True)

filename = '/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d03_2024-10-08_00:00:00.nc'
ncfile = Dataset(filename)

pblh_arr = getvar(ncfile, "PBLH", timeidx = None,) + getvar(ncfile, "HGT", timeidx=None)
z = getvar(ncfile, "z", units="m")
height_arr = getvar(ncfile, "HGT")
wrf_times = getvar(ncfile, "times", timeidx=None, meta=False)
wrf_times = np.datetime_as_string(wrf_times, unit="m")


start_idx = int(24*60/5) #hours * minutes/hour / minute/reading

end_time_idx = pblh_arr.shape[0]

start_point = CoordPair(lat=38.173750, lon=21.512966)
end_point = CoordPair(lat=37.64947, lon=22.66818)
NPRK = CoordPair(lat=38.00745, lon=22.196031)

pblh_line = interpline(pblh_arr, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True)
height_line = interpline(height_arr, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True)

del pblh_arr
del height_arr


for timeidx in range(start_idx, end_time_idx, 3):
    print(f"Plotting Time: {timeidx}")
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()
    wspd = getvar(ncfile, "uvmet_wspd_wdir", timeidx=timeidx)[0,:]
    pblh = pblh_line[timeidx, :]

    start_point = CoordPair(lat=38.173750, lon=21.512966)
    end_point = CoordPair(lat=37.841686, lon=22.779232)
    

    # Compute the vertical cross-section interpolation.  Also, include the
    # lat/lon points along the cross-section.
    wspd_cross = vertcross(wspd, z, wrfin=ncfile, start_point=start_point,
                        end_point=end_point, latlon=True, meta=True)
    # Create the figure
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()

    # Make the contour plot
    ax.set_facecolor("black")
    wspd_contours = ax.contourf(to_np(wspd_cross), cmap="viridis")


    # Add the color bar
    plt.colorbar(wspd_contours, ax=ax)
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
    plt.title(wrf_times[timeidx])
    ax.set_ylim(0,20)

    plot_name = f"PBLH_gif_{timeidx:03}.png"
    plt.savefig(str(Path(save_path, plot_name)))
    plt.close()