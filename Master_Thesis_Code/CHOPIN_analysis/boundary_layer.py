# %%
import datetime
import sys
import time

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import wrf
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.cm import get_cmap
from netCDF4 import Dataset  # type: ignore
from wrf import CoordPair, destagger, getvar, interplevel, interpline, to_np, vertcross, ALL_TIMES

sys.path.append("/home/waseem/ali_plots/")
from utils import get_wrf_times


def get_plot_start_time(first_time:pd.Timestamp) -> pd.Timestamp:
    year = first_time.year
    month = first_time.month
    day = first_time.day
    hour = first_time.hour
    start_time_hour = 0
    if hour >=6:
        start_time_hour = 6
    if hour >= 12:
        start_time_hour = 12
    if hour >= 18:
        start_time_hour = 18
    
    return pd.Timestamp(year=year, month=month, day=day, hour=start_time_hour)

def get_plot_end_time(last_time:pd.Timestamp) -> pd.Timestamp:
    year = last_time.year
    month = last_time.month
    day = last_time.day
    hour = last_time.hour
    end_time_hour = 0
    add_day = False
    if hour < 18:
        end_time_hour = 18
    if hour < 12:
        end_time_hour = 12
    if hour < 6:
        end_time_hour = 6
    if hour >=18:
        end_time_hour = 0
        add_day = True
    
    end_time = pd.Timestamp(year=year, month=month, day=day, hour=end_time_hour)
    if add_day:
        end_time += pd.Timedelta(1, "D")
    return end_time
# %%
#############################################################
# Plot PBLH over time at NPRK and compare met data at HAC
#############################################################
wrfout_path_ysu_NPRK = "/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_YSU.nc"
wrfout_path_keps_NPRK = "/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_KEPS.nc"
wrfout_path_myj_NPRK = "/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_MYJ.nc"
ysu_NPRK = Dataset(wrfout_path_ysu_NPRK)
keps_NPRK = Dataset(wrfout_path_keps_NPRK)
myj_NPRK = Dataset(wrfout_path_myj_NPRK)

wrfout_path_ysu_HAC = "/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_YSU.nc"
wrfout_path_keps_HAC = "/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_KEPS.nc"
wrfout_path_myj_HAC = "/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_HAC_MYJ.nc"
ysu_HAC = Dataset(wrfout_path_ysu_HAC)
keps_HAC = Dataset(wrfout_path_keps_HAC)
myj_HAC = Dataset(wrfout_path_myj_HAC)

spin_up_idx = int(24*60/5)
times = getvar(ysu_NPRK, "Times", meta=False)[spin_up_idx:]
assert isinstance(times, np.ndarray)

pd_times = pd.Series(times) + pd.Timedelta(2, "h")

start_time = get_plot_start_time(pd_times.iloc[0]) # type: ignore
end_time = get_plot_end_time(pd_times.iloc[-1]) # type: ignore

# ysu_T_wrfpy = np.zeros((len(pd_times), 96)) 
# myj_T_wrfpy = np.zeros((len(pd_times), 96)) 
# keps_T_wrfpy = np.zeros((len(pd_times), 96)) 
# ysu_rh_wrfpy = np.zeros((len(pd_times), 96)) 
# myj_rh_wrfpy = np.zeros((len(pd_times), 96)) 
# keps_rh_wrfpy = np.zeros((len(pd_times), 96)) 

# svp1=611.2
# svp2=17.67
# svp3=29.65
# svpt0=273.15
# eps = 0.622
# rh = 1.E2 * (p*q/(q*(1.-eps) + eps))/(svp1*exp(svp2*(t-svpt0)/(T-svp3)))

# j = 0
# for i in range(spin_up_idx, len(times)+spin_up_idx):
#     # ysu_T_wrfpy[j, :] = getvar(ysu_HAC, "temp", units="degC", meta=False, timeidx=i)
#     # myj_T_wrfpy[j, :] = getvar(myj_HAC, "temp", units="degC", meta=False, timeidx=i)
#     # keps_T_wrfpy[j, :] = getvar(keps_HAC, "temp", units="degC", meta=False, timeidx=i)
#     ysu_rh_wrfpy[j, :] = getvar(ysu_HAC, "rh", meta=False, timeidx=i)
#     myj_rh_wrfpy[j, :] = getvar(myj_HAC, "rh", meta=False, timeidx=i)
#     keps_rh_wrfpy[j, :] = getvar(keps_HAC, "rh", meta=False, timeidx=i)
#     j+=1

# %%




HGT_NPRK = np.squeeze(ysu_NPRK.variables["HGT"][0])
HGT_HAC = np.squeeze(ysu_HAC.variables["HGT"][0])
HGT_diff = HGT_HAC - HGT_NPRK
PBLH_YSU = np.squeeze(ysu_NPRK.variables["PBLH"][-len(times):])
PBLH_MYJ = np.squeeze(myj_NPRK.variables["PBLH"][-len(times):])
PBLH_KEPS = np.squeeze(keps_NPRK.variables["PBLH"][-len(times):])

tick_locs = mdates.drange(start_time, end_time, datetime.timedelta(hours=6))
tick_labels = [mdates.num2date(t).strftime('%d/%m'+'\n'+ '%H:%M') for t in tick_locs]


step =1
plt.figure()
plt.plot(pd_times[::step], PBLH_YSU[::step], label="YSU")
plt.plot(pd_times[::step], PBLH_MYJ[::step], label="MYJ")
plt.plot(pd_times[::step], PBLH_KEPS[::step], label="KEPS")
# plt.hlines(HGT_diff, tick_locs[0], tick_locs[-1], linestyle="--", color= "lightgrey", label="HAC grid cell")
plt.hlines(2314-HGT_NPRK, tick_locs[0], tick_locs[-1], linestyle="--", color= "k", label="HAC Altitude")
lgnd= plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
          ncol=1)
plt.title(f"BLH from {pd_times.iloc[0].strftime('%d/%m %H:%M')} to {pd_times.iloc[-1].strftime('%d/%m %H:%M')}")
plt.xticks(tick_locs, labels=tick_labels)
plt.xlabel("Date [Local]")
plt.ylabel("Height Above Ground [m]")
# fig.savefig('samplefigure', bbox_extra_artists=(lgnd,), bbox_inches='tight')

#%%

meteo_path = "/home/waseem/CHOPIN_analysis/data/HAC_meteo.csv"
meteo = pd.read_csv(meteo_path)
meteo["time"] = pd.to_datetime(meteo["time"])
mask = (meteo.time>=pd_times.iloc[0]) & (meteo.time<=pd_times.iloc[-1])
meteo_slice = meteo.loc[mask]
plt.figure()
fig, ax = plt.subplots(3, 1, figsize=(8,9))
level = 0
z = getvar(ysu_HAC, "zstag", meta=False)
# ysu_temp = np.zeros([len(times), 96])
RA=287.15
RD=287.0
CP=1004.5
P1000MB=100000.0
EPS=0.622
# tkCONTROL = ((presCONTROL / P1000MB)**(RD/CP) * thetCONTROL)

ysu_theta = np.squeeze(ysu_HAC.variables["T"][spin_up_idx:]) + 300
myj_theta = np.squeeze(myj_HAC.variables["T"][spin_up_idx:]) + 300
keps_theta = np.squeeze(keps_HAC.variables["T"][spin_up_idx:]) + 300
ysu_pres = np.squeeze(ysu_HAC.variables["P"][spin_up_idx:] + ysu_HAC.variables["PB"][spin_up_idx:])
myj_pres = np.squeeze(myj_HAC.variables["P"][spin_up_idx:] + myj_HAC.variables["PB"][spin_up_idx:])
keps_pres = np.squeeze(keps_HAC.variables["P"][spin_up_idx:] + keps_HAC.variables["PB"][spin_up_idx:])

ysu_tk = ((ysu_pres/P1000MB)**(RD/CP) * ysu_theta)
myj_tk = ((myj_pres/P1000MB)**(RD/CP) * myj_theta)
keps_tk = ((keps_pres/P1000MB)**(RD/CP) * keps_theta)

ysu_qv = np.squeeze(ysu_HAC.variables["QVAPOR"][spin_up_idx:])
myj_qv = np.squeeze(myj_HAC.variables["QVAPOR"][spin_up_idx:])
keps_qv = np.squeeze(keps_HAC.variables["QVAPOR"][spin_up_idx:])


ysu_tv = ysu_tk * (EPS + ysu_qv) / (EPS * (1.0 + ysu_qv))
myj_tv = myj_tk * (EPS + myj_qv) / (EPS * (1.0 + myj_qv))
keps_tv = keps_tk * (EPS + keps_qv) / (EPS * (1.0 + keps_qv))

ysu_t2 = np.squeeze(ysu_HAC.variables["T2"][spin_up_idx:])
myj_t2 = np.squeeze(myj_HAC.variables["T2"][spin_up_idx:])
keps_t2 = np.squeeze(keps_HAC.variables["T2"][spin_up_idx:])


ax[0].plot(meteo_slice.time, meteo_slice.temp, color= 'k', label= "Measured")
ax[0].plot(pd_times, ysu_t2- 273.15, label="ysu")
ax[0].plot(pd_times, myj_t2- 273.15, label="myj")
ax[0].plot(pd_times, keps_t2- 273.15, label="keps")
ax[0].legend()
ax[0].set_title("2m Temperature")
ax[0].set_xticks([])
ax[0].set_ylabel("Temperature [$^{\circ}$C]")

ax[1].plot(meteo_slice.time, meteo_slice.rh, color= 'k', label= "Measured")
ax[1].plot(pd_times, ysu_rh_wrfpy[:,level], label="ysu")
ax[1].plot(pd_times, myj_rh_wrfpy[:,level], label="myj")
ax[1].plot(pd_times, keps_rh_wrfpy[:,level], label="keps")
ax[1].legend()
ax[1].set_title("Relative Humidity")
ax[1].set_xticks([])
ax[1].set_ylabel("RH [%]]")

ax[2].plot(pd_times, PBLH_YSU, label="YSU")
ax[2].plot(pd_times, PBLH_MYJ, label="MYJ")
ax[2].plot(pd_times, PBLH_KEPS, label="KEPS")
# plt.hlines(HGT_diff, tick_locs[0], tick_locs[-1], linestyle="--", color= "lightgrey", label="HAC grid cell")
ax[2].axhline(2314-HGT_NPRK, linestyle="--", color= "k", label="HAC Altitude")
# lgnd= ax[2].legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
#           ncol=1)
ax[2].legend()
ax[2].set_title(f"Boundary Layer Height")
ax[2].set_xticks(tick_locs, labels=tick_labels)
ax[2].set_ylabel("Height above NPRK Station [m]")
plt.xlabel("Date [Local]")


#%%
# Error in temperature, error in humdiity
RMSE_temp_ysu = np.mean(((ysu_t2-273.15) - meteo_slice.temp.to_numpy())**2)**0.5
RMSE_temp_mjy = np.mean(((myj_t2-273.15) - meteo_slice.temp.to_numpy())**2)**0.5
RMSE_temp_keps = np.mean(((keps_t2-273.15) - meteo_slice.temp.to_numpy())**2)**0.5

RMSE_rh_ysu = np.mean(((ysu_rh_wrfpy[:,level]) - meteo_slice.rh.to_numpy())**2)**0.5
RMSE_rh_mjy = np.mean(((myj_rh_wrfpy[:,level]) - meteo_slice.rh.to_numpy())**2)**0.5
RMSE_rh_keps = np.mean(((keps_rh_wrfpy[:,level]) - meteo_slice.rh.to_numpy())**2)**0.5

print(f" RMSE error Temperature [C] YSU: {RMSE_temp_ysu:.2f}")
print(f" RMSE error Temperature [C] MYJ: {RMSE_temp_mjy:.2f}")
print(f" RMSE error Temperature [C] KEPS: {RMSE_temp_keps:.2f}")
print()
print(f" RMSE error Temperature [%] YSU: {RMSE_rh_ysu:.2f}")
print(f" RMSE error Temperature [%] MYJ: {RMSE_rh_mjy:.2f}")
print(f" RMSE error Temperature [%] KEPS: {RMSE_rh_keps:.2f}")


# %%
#############################################################
# Plot PBLH over time
#############################################################
wrfout_path_1 = "/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d03_2024-10-08_NPRK.nc"
wrfout_path_2 = "/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d03_2024-10-08_HAC_test_stag_2.nc"
dataset_1 = Dataset(wrfout_path_1)
dataset_2 = Dataset(wrfout_path_2)

times = get_wrf_times("/scratch/waseem/CHOPIN_oct8-10_allsip/namelist.input", np.timedelta64(1, "D"))


PBLH_1 = np.squeeze(dataset_1.variables["PBLH"])
PBLH_2 = np.squeeze(dataset_2.variables["PBLH"])

HGT_1 = np.squeeze(dataset_1.variables["HGT"])
HGT_2 = np.squeeze(dataset_2.variables["HGT"])

pd_times = pd.Series(times) + 
start_time = get_plot_start_time(pd_times.iloc[0])
end_time = get_plot_end_time(pd_times.iloc[-1])

tick_locs = mdates.drange(start_time, end_time, datetime.timedelta(hours=6))
tick_labels = [mdates.num2date(t).strftime('%d/%m'+'\n'+ '%H:%M') for t in tick_locs]

plt.figure()
plt.plot(pd_times, PBLH_1[-len(times):], label="HAC")
plt.plot(pd_times, PBLH_2[-len(times):], label="VL")
plt.legend()
plt.title(f"BLH from {pd_times.iloc[0].strftime('%d/%m %H:%M')} to {pd_times.iloc[-1].strftime('%d/%m %H:%M')}")
plt.xticks(tick_locs, labels=tick_labels)
plt.xlabel("Date [UTC]")
plt.ylabel("Height Above Ground [m]")

# %%
#############################################################
# Plot the domains
#############################################################

def get_plot_element(infile):
    rootgroup = nc.Dataset(infile, 'r')
    p = wrf.getvar(rootgroup, 'RAINNC')
    lats, lons = wrf.latlon_coords(p)
    cart_proj = wrf.get_cartopy(p)
    xlim = wrf.cartopy_xlim(p)
    ylim = wrf.cartopy_ylim(p)
    rootgroup.close()
    return cart_proj, xlim, ylim, lats, lons
 
infile_d01 = '/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d01_2024-10-08_00:00:00.nc'
cart_proj, xlim_d01, ylim_d01, lats_d01, lons_d01 = get_plot_element(infile_d01)
 
infile_d02 = '/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d02_2024-10-08_00:00:00.nc'
_, xlim_d02, ylim_d02, lats_d02, lons_d02 = get_plot_element(infile_d02)
 
infile_d03 = '/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d03_2024-10-08_00:00:00.nc'
_, xlim_d03, ylim_d03, lats_d03, lons_d03 = get_plot_element(infile_d03)
 
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=cart_proj)
 
ax.coastlines('50m', linewidth=0.8)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# d01
ax.set_xlim([xlim_d01[0]-(xlim_d01[1]-xlim_d01[0])/15, xlim_d01[1]+(xlim_d01[1]-xlim_d01[0])/15])
ax.set_ylim([ylim_d01[0]-(ylim_d01[1]-ylim_d01[0])/15, ylim_d01[1]+(ylim_d01[1]-ylim_d01[0])/15])
 
# d01 box
ax.add_patch(mpl.patches.Rectangle((xlim_d01[0], ylim_d01[0]), xlim_d01[1]-xlim_d01[0], ylim_d01[1]-ylim_d01[0],
             fill=None, lw=3, edgecolor='blue', zorder=10))
ax.text(xlim_d01[0]+(xlim_d01[1]-xlim_d01[0])*0.05, ylim_d01[0]+(ylim_d01[1]-ylim_d01[0])*0.9, 'D01',
        size=15, color='blue', zorder=10)
 
# d02 box
ax.add_patch(mpl.patches.Rectangle((xlim_d02[0], ylim_d02[0]), xlim_d02[1]-xlim_d02[0], ylim_d02[1]-ylim_d02[0],
             fill=None, lw=3, edgecolor='black', zorder=10))
ax.text(xlim_d02[0]+(xlim_d02[1]-xlim_d02[0])*0.05, ylim_d02[0]+(ylim_d02[1]-ylim_d02[0])*1.1, 'D02',
        size=15, color='black', zorder=10)
 
# d03 box
ax.add_patch(mpl.patches.Rectangle((xlim_d03[0], ylim_d03[0]), xlim_d03[1]-xlim_d03[0], ylim_d03[1]-ylim_d03[0],
             fill=None, lw=3, edgecolor='red', zorder=10))
ax.text(xlim_d03[0]+(xlim_d03[1]-xlim_d03[0])*0.1, ylim_d03[0]+(ylim_d03[1]-ylim_d03[0])*0.8, 'D03',
        size=15, color='red', zorder=10)
# ax.gridlines(color="black", linestyle="dotted", draw_labels=True)

# %%
#############################################################
# Now we are going to try a vertical cross section
#############################################################

filename = '/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d03_2024-10-08_00:00:00.nc'
ncfile = Dataset(filename)
# %%

height_arr = getvar(ncfile, "HGT")
pblh_arr = getvar(ncfile, "PBLH", timeidx=None) + height_arr
z = getvar(ncfile, "z", units="m")
start_point = CoordPair(lat=38.173750, lon=21.512966)
end_point = CoordPair(lat=37.841686, lon=22.779232)
NPRK = CoordPair(lat=38.00745, lon=22.196031)
HAC = CoordPair(lat = 37.9843, lon = 22.1963)
lon_range = []
# pblh_line = interpline(pblh_arr, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True, timeidx=None)
# height_line = interpline(height_arr, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True, timeidx=None)
pblh_line = interpline(pblh_arr, wrfin=ncfile, pivot_point=HAC, angle=-2, latlon=True, timeidx=None)
height_line = interpline(height_arr, wrfin=ncfile, pivot_point=HAC, angle=-2, latlon=True, timeidx=None)

wrf_times = getvar(ncfile, "times", timeidx=None, meta=False)
coord_pairs = to_np(pblh_line.coords["xy_loc"])
line_lonlats = np.array([[coord.lat, coord.lon] for coord in coord_pairs])

#%%
times = []
times.append(time.time())
timeidx = 600

pblh = pblh_line[timeidx, :]
height = height_line
wrf_times = np.datetime_as_string(wrf_times, unit="m")
times.append(time.time()) # flag 1: get vars


# Compute the vertical cross-section interpolation.  Also, include the
# lat/lon points along the cross-section.
wspd = getvar(ncfile, "uvmet_wspd_wdir", timeidx=timeidx)[0,:]
wspd_cross = vertcross(wspd, z, wrfin=ncfile, start_point=start_point,
                       end_point=end_point, latlon=True, meta=True)
# %%
# Create the figure
fig = plt.figure(figsize=(12,6))
ax = plt.axes()

# Make the contour plot
ax.set_facecolor("black")
times.append(time.time()) # flag 5: set face colour
wspd_contours = ax.contourf(to_np(wspd_cross), cmap=get_cmap("viridis"))
times.append(time.time()) # flag 5: plot contour


# Add the color bar
plt.colorbar(wspd_contours, ax=ax)
# Set the x-ticks to use latitude and longitude labels.

x_ticks = np.arange(coord_pairs.shape[0])
x_labels = [pair.latlon_str(fmt="{:.2f}, {:.2f}")
            for pair in to_np(coord_pairs)]
ax.set_xticks(x_ticks[::20])
ax.set_xticklabels(x_labels[::20], rotation=45, fontsize=8)
times.append(time.time()) # flag 6: set xticks

# Set the y-ticks to be height.
vert_vals = to_np(wspd_cross.coords["vertical"])
v_ticks = np.arange(vert_vals.shape[0])
ax.set_yticks(v_ticks[::4])
ax.set_yticklabels(vert_vals[::4], fontsize=8)

# Set the x-axis and  y-axis labels
ax.set_xlabel("Latitude, Longitude", fontsize=12)
ax.set_ylabel("Height (m)", fontsize=12)
times.append(time.time()) # flag 7: set y ticks
pbhl_line_graph_space = np.interp(pblh, vert_vals, np.arange(100))
times.append(time.time()) # flag 8: interp pblh to coords
height_line_graph_space = np.interp(height, vert_vals, np.arange(100))
times.append(time.time()) # flag 9: interp height
ax.plot(pbhl_line_graph_space, color='magenta', linewidth=2.5, label="PBLH")
times.append(time.time()) # flag 10: plot pblh
# ax.plot(height_line_graph_space, color="orange", linewidth=2)
ax.fill_between(np.arange(height.shape[0]), height_line_graph_space, color="saddlebrown", label="Terrain")
times.append(time.time()) # flag 11: plot height
plt.legend()
plt.title(wrf_times[timeidx])
ax.set_ylim(0,30)


# %%
