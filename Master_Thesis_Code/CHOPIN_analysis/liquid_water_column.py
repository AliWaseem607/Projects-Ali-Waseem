# %%
import cartopy.crs as crs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from wrf import cartopy_xlim, cartopy_ylim, get_cartopy, getvar, latlon_coords, to_np

# %%
# Open the NetCDF file
ncfile = Dataset("/scratch/waseem/CHOPIN_rain_oct6-9/wrfout_CHPN_d03_2024-10-06_00:00:00.nc")

times = getvar(ncfile, "Times", timeidx=None, meta=False)

# %%
toi = np.datetime64("2024-10-07T04:00")
time_idx = np.argwhere(times == toi)[0][0]


# Get the LWC
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
            getvar(ncfile, "QCLOUD", timeidx=timestep)
            + getvar(ncfile, "QRAIN", timeidx=timestep)
        )
        * rho
        * 10**3
    )

    zstag = getvar(ncfile, "zstag", timeidx=timestep, meta=False)
    dz = np.diff(zstag, axis=0)
    lwc_arr = to_np(lwc)
    return np.sum(lwc_arr * dz, axis=0)


column_LWP = get_LWP(ncfile, time_idx)
column_LWP[column_LWP < 10**-5] = np.nan

# Smooth the sea level pressure since it tends to be noisy near the
# mountains


qvap = getvar(ncfile, "QVAPOR")
# Get the latitude and longitude points
lats, lons = latlon_coords(qvap, as_np=True)

# Get the cartopy mapping object
cart_proj = get_cartopy(qvap)

# Create a figure
fig = plt.figure(figsize=(7, 6))
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

states = NaturalEarthFeature(category="physical", scale="10m", facecolor="none", name="coastline")
ax.add_feature(
    states,
    linewidth=1,
    edgecolor="black",
)
# ax.coastlines('10m', linewidth=0.8)

# Make the contour outlines and filled contours for the smoothed sea level
# pressure.
# plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), 10, colors="black",
#             transform=crs.PlateCarree())
levels = np.linspace(-4, 3, 15)
cont0 = plt.contourf(
    to_np(lons), to_np(lats), np.log10(column_LWP).T, levels=levels, transform=crs.PlateCarree(), cmap="jet"
)

# Add a color bar
cb = plt.colorbar(cont0)
# cb.set_ticks(np.linspace(-4,1,7))
cb.set_label("log[g/m3]")

# Set the map bounds
ax.set_xlim(cartopy_xlim(slp))
ax.set_ylim(cartopy_ylim(slp))

# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")
plt.scatter(167, 153, color="k", marker="*", s=60)

plt.title(f"LWP {times[time_idx]}")

plt.show()
# %%
# plot as gif
