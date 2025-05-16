# %%
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
from wrf import get_cartopy, getvar

sys.path.append("./")
from scalebar import scale_bar

#%%

# d01_max_lon =
# d01_min_lon =
# d01_max_lat =
# d01_min_lat = 

# d02_max_lon =
# d02_min_lon =
# d02_max_lat =
# d02_min_lat = 

# d03_max_lon = 23.994019
# d03_min_lon = 20.362518
# d03_max_lat = 
# d03_min_lat = 

ncfile_path_d01 = Path("/scratch/waseem/CHOPIN_dec21-23_CONTROL/wrfout_CHPN_d01_2024-12-21_00:00:00.nc")
ncfile_d01 = Dataset(ncfile_path_d01)
lats_d01 = ncfile_d01.variables["XLAT"][0, :, :]
lons_d01 = ncfile_d01.variables["XLONG"][0, :, :]

ncfile_path_d02 = Path("/scratch/waseem/CHOPIN_dec21-23_CONTROL/wrfout_CHPN_d02_2024-12-21_00:00:00.nc")
ncfile_d02 = Dataset(ncfile_path_d02)
lats_d02 = ncfile_d02.variables["XLAT"][0, :, :]
lons_d02 = ncfile_d02.variables["XLONG"][0, :, :]

ncfile_path_d03 = Path("/scratch/waseem/CHOPIN_dec21-23_CONTROL/wrfout_CHPN_d03_2024-12-21_00:00:00.nc")
ncfile_d03 = Dataset(ncfile_path_d03)
lats_d03 = ncfile_d03.variables["XLAT"][0, :, :]
lons_d03 = ncfile_d03.variables["XLONG"][0, :, :]
# Create a figure

# Set the GeoAxes to the projection used by WRF
cart_proj = get_cartopy(getvar(ncfile_d01, "slp"))
fig = plt.figure(figsize=(28, 10))
gs = fig.add_gridspec(1, 2)
ax = fig.add_subplot(gs[0], projection=cart_proj)
ax2 = fig.add_subplot(gs[1], projection=cart_proj)
# ax = plt.axes(projection=cart_proj)

colours = ["blue", "orange", "tab:red"]
i = 0
for lons, lats in zip([lons_d01, lons_d02, lons_d03], [lats_d01, lats_d02, lats_d03]):
    ax.plot(lons[:,0], lats[:,0], color=colours[i], transform=ccrs.PlateCarree(), linewidth=4)
    ax.plot(lons[:,-1], lats[:,-1], color=colours[i], transform=ccrs.PlateCarree(), linewidth=4)
    ax.plot(lons[0,:], lats[0,:], color=colours[i], transform=ccrs.PlateCarree(), linewidth=4)
    ax.plot(lons[-1,:], lats[-1,:], color=colours[i], transform=ccrs.PlateCarree(), linewidth=4)
    i +=1
# ax.scatter(22.196028, 38.007444, marker ="s", label = "Mt Helmos", color="lawngreen", s= 200)

max_lat = np.max(lats_d01)
min_lat = np.min(lats_d01)
max_lon = np.max(lons_d01)
min_lon = np.min(lons_d01)

# lon lon, lat lat
ax.set_extent([-5, 45, 27, 55], crs=ccrs.PlateCarree())

ax.text(5, 42, "d01", color="blue", fontsize=22, transform=ccrs.PlateCarree(), weight="bold" )
ax.text(13, 35, "d02", color="orange", fontsize=22, transform=ccrs.PlateCarree(), weight="bold" )
ax.text(20, 35, "d03", color="tab:red", fontsize=22, transform=ccrs.PlateCarree(), weight="bold" )

# ax.add_image(terrain_tiler, 3)

tiler = img_tiles.GoogleTiles()

ax.add_image(tiler, 4)

# tiler = img_tiles.OSM()
# ax.add_image(tiler, 4)

# tiler = img_tiles.QuadtreeTiles()
# ax.add_image(tiler, 4)



# ax2.plot(lons_d03[:,0], lats_d03[:,0], color="r", transform=ccrs.PlateCarree())
# ax2.plot(lons_d03[:,-1], lats_d03[:,-1], color="r", transform=ccrs.PlateCarree())
# ax2.plot(lons_d03[0,:], lats_d03[0,:], color="r", transform=ccrs.PlateCarree())
# ax2.plot(lons_d03[-1,:], lats_d03[-1,:], color="r", transform=ccrs.PlateCarree())
tiler = img_tiles.GoogleTiles( style="satellite")
for spine in ax2.spines.values():
        spine.set_edgecolor('tab:red')
        spine.set_linewidth(6)

ax2.set_extent([20.42, 23.92, 36.44, 39.25], crs=ccrs.PlateCarree())
ax2.add_image(tiler, 8)
ax2.scatter(22.196028, 38.007444, marker ="s", label = "Mt Helmos", color="lawngreen", s= 350)
ax2.legend(fontsize=20, loc="upper left")
# fig.line()

# plt.tight_layout()

plt.savefig("figures/maps/domains.png")