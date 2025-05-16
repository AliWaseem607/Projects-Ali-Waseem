from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from wrf import cartopy_xlim, cartopy_ylim, get_cartopy, getvar, latlon_coords, to_np


def get_LWC_WRF_grid(ncfile):
    ### WRF constants
    RA = 287.15
    RD = 287.0
    CP = 1004.5
    P1000MB = 100000.0
    EPS = 0.622
    pres = getvar(ncfile, "P", timeidx=None, meta=False) + getvar(
        ncfile, "PB", timeidx=None, meta=False
    )
    thet = getvar(ncfile, "T", timeidx=None, meta=False) + 300.0
    qv = getvar(ncfile, "QVAPOR", timeidx=None, meta=False)
    tk = (pres / P1000MB) ** (RD / CP) * thet
    tv = tk * (EPS + qv) / (EPS * (1.0 + qv))
    rho = pres / RA / tv
    lwc = (
        (
            getvar(ncfile, "QCLOUD", timeidx=None, meta=False)
            + getvar(ncfile, "QRAIN", timeidx=None, meta=False)
        )
        * rho
        * 10**3
    )

    return lwc

def get_LWC_ERA5_grid(ncfile, lon_ind, lat_ind):
    RA = 287.15
    EPS = 0.622
    tk = getvar(ncfile, "t", meta=False, timeidx=None)[:,:,lon_ind,lat_ind]
    SH = getvar(ncfile, "q", meta=False, timeidx=None)[:,:,lon_ind,lat_ind]
    pres = getvar(ncfile, "z", meta=False, timeidx=None)[:,:,lon_ind,lat_ind]
    tv = tk * (EPS + SH) / (EPS * (1.0 + SH))
    rho = pres / RA / tv
    lwc = (
        (
            (getvar(ncfile, "clwc", timeidx=None, meta=False)[:,:,lon_ind,lat_ind])
            + (getvar(ncfile, "crwc", timeidx=None, meta=False)[:,:,lon_ind,lat_ind])
        )
        * rho
        * 10**3
    )
    return lwc


def get_wrf_LWP(ncfile, timestep):
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

def get_era5_LWP_all(ncfile):
    RA = 287.15
    EPS = 0.622
    tk = getvar(ncfile, "t", meta=False, timeidx=None)
    SH = getvar(ncfile, "q", meta=False, timeidx=None)
    pres = getvar(ncfile, "z", meta=False, timeidx=None)
    tv = tk * (EPS + SH) / (EPS * (1.0 + SH))
    rho = pres / RA / tv
    lwc = (
        (
            getvar(ncfile, "clwc", timeidx=None, meta=False)
            + getvar(ncfile, "crwc", timeidx=None, meta=False)
        )
        * rho
        * 10**3
    )
    geopotential_height = pres/9.81
    min_geo_height = geopotential_height[:,0:1,:,:] - np.diff(geopotential_height[:,0:2, :, :], axis=1)
    dgeo_height = np.diff(pres / 9.81 - min_geo_height, axis=1)

    column = lwc[:,:-1, :, :] * dgeo_height
    column = np.sum(column, axis=1)
    return column

def plot_ERA5_vs_WRF_LWC_at_grib_point(era5_ncfile:Dataset, wrf_ncfile:Dataset, save_path: Path):
    times = getvar(era5_ncfile, "valid_time", meta=False, timeidx=None).filled()
    era5_times = pd.Series(map(lambda x: pd.Timestamp("1970-01-01") + pd.Timedelta(x, "s"), times))
    wrf_times = pd.Series(getvar(wrf_ncfile, "Times", meta=False, timeidx=None))

    min_time = np.min((era5_times.iloc[0], wrf_times.iloc[288])) # 288 is 24h spinup for 5 min resolution
    max_time = np.min((era5_times.iloc[-1], wrf_times.iloc[-1]))

    lon_ind = 9
    lat_ind = 8
    era5_lwc = get_LWC_ERA5_grid(era5_ncfile, lon_ind, lat_ind)
    era5_pres = getvar(era5_ncfile, "z", meta=False, timeidx=None)[:,:,lon_ind,lat_ind]

    HGT = getvar(wrf_ncfile, "HGT", meta=False)

    wrf_lwc = get_LWC_WRF_grid(wrf_ncfile)
    wrf_lwc[wrf_lwc<10**-5] = np.nan
    ZZ_wrf =(getvar(wrf_ncfile,"PH",meta=False, timeidx=None)+getvar(wrf_ncfile,"PHB",meta=False, timeidx=None))/9.81-HGT
    
    ZZ_wrf_middle = np.diff(ZZ_wrf, axis=1)/2 + ZZ_wrf[:,:-1]

    ZZ_era5 = era5_pres/9.81 - HGT
    # ZZmiddle = np.diff(ZZ, axis=1)/2 + ZZ[:,:-1]
    era5_lwc[era5_lwc<10**-5] = np.nan

    wrf_time_mask = (wrf_times>=min_time) & (wrf_times<=max_time)
    era5_time_mask = (era5_times>=min_time) & (era5_times<=max_time)

    tick_locs =  mdates.drange(min_time, max_time, pd.Timedelta(6, "h"))
    tick_labels = [mdates.num2date(t).strftime('%d/%m'+'\n'+ '%H:%M') for t in tick_locs]

    fig, ax = plt.subplots(2,1, figsize=(7,6))
    levels = np.linspace(-6, 1, 9)


    im0 = ax[0].contourf(era5_times[era5_time_mask], ZZ_era5[0,:]/1000, np.log10(era5_lwc[era5_time_mask,:]).T, levels=levels)
    ax[0].contourf(era5_times[era5_time_mask], ZZ_era5[0,:]/1000, np.log10(era5_lwc[era5_time_mask,:]).T, levels=levels)
    ax[0].set_ylim((0,10))
    ax[1].set_ylim((0,10))
    cbar = plt.colorbar(im0, ax=ax[0])
    cbar.set_label("LWC g/m3")
    ax[0].set_xticks(tick_locs)
    ax[0].set_xticklabels([])
    fig.supylabel("Distance [km]")
    fig.supxlabel("Date [UTC]")

    ax[1].contourf(wrf_times[wrf_time_mask], ZZ_wrf_middle[288,:]/1000, np.log10(wrf_lwc[wrf_time_mask,:].T), levels=levels)
    cbar = plt.colorbar(im0, ax=ax[1])
    cbar.set_label("LWC g/m3")
    plt.tight_layout()

    ax[1].set_xticks(tick_locs)
    ax[1].set_xticklabels(tick_labels)

    ax[0].set_title("LWC from GRIB data")
    ax[1].set_title("LWC from WRF (same location as GRIB)")
    plt.savefig(Path(save_path, "ERA5_vs_WRF_LWC.png"))

def plot_ERA5_vs_WRF_LWP(era5_ncfile, wrf_ncfile, save_path):
    
    # wrf = Dataset("/scratch/waseem/helmos_chopin_test/wrfout_Helmos_d03_2021-12-17_00:00:00.nc")
    # era5 = Dataset("carmelle.nc")
    # save_path = "/home/waseem/CHOPIN_analysis/figures/carmelle_comparison/"
    # plot_comparison("/home/waseem/CHOPIN_analysis/figures/carmelle_comparison/", era5, wrf)


    # def plot_comparison(save_path:Path, era5:Dataset, wrf:Dataset):
    wrf_times = pd.Series(getvar(wrf_ncfile, "times", timeidx=None, meta=False))
    era5_times_temp = getvar(era5_ncfile, "valid_time", meta=False, timeidx=None).filled()
    era5_times = pd.Series(map(lambda x: pd.Timestamp("1970-01-01") + pd.Timedelta(x, "s"), era5_times_temp))
    era5_column = get_era5_LWP_all(era5_ncfile)
    era5_column[era5_column < 10**-5] = np.nan

    era5_lats = getvar(era5_ncfile, "latitude", meta=False, timeidx=None)
    era5_lons = getvar(era5_ncfile, "longitude", meta=False, timeidx=None)
    levels = np.linspace(-4, 4, 17)
    qvap = getvar(wrf_ncfile, "QVAPOR")
    # Get the latitude and longitude points
    wrf_lats, wrf_lons = latlon_coords(qvap, as_np=True)
    cart_proj = get_cartopy(qvap)
    for era5_idx in range(len(era5_times)):
        time = era5_times[era5_idx]
        wrf_idx = wrf_times[wrf_times == time].index # type: ignore
        if wrf_idx[0] == None:
            continue
        print(f"strating time {wrf_times[wrf_idx]}")
        wrf_column = get_wrf_LWP(wrf_ncfile, wrf_idx[0])
        wrf_column[wrf_column < 10**-5] = np.nan

        # Plot grib
        
        # Create a figure
        fig = plt.figure(figsize = (11,7))
        ax0 = fig.add_subplot(1,2,1, projection= cart_proj)

        ax0.coastlines()
        cont0 = ax0.contourf(era5_lons, era5_lats, np.log10(era5_column[era5_idx,:,:]).T, levels=levels, cmap="jet", transform=ccrs.PlateCarree())
        # Add a color bar
        cb = plt.colorbar(cont0, ax=ax0, fraction=0.046, pad=0.04)
        # cb.set_ticks(levels)
        cb.set_label("log[g/m3]")
        # latitude = 37.9995 longitude = 22.19329
        ax0.scatter(22.19329, 37.9995, color="k", marker="*", s=80, label="VL Radar", transform=ccrs.PlateCarree())
        ax0.scatter(22.25,38.00, color="r", marker="s", label="GRIB data point", transform=ccrs.PlateCarree())
        # ax0.set_xticks(era5_lons[::2])
        # ax0.set_yticks(era5_lats[::2])
        # ax0.title("Cloud and rain water content at 2024-10-07 04:00: UTC")
        gl = ax0.gridlines(color="black", linestyle="dotted", draw_labels = True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'rotation': 0, 'ha':"center"}
        ax0.set_extent([20.5,23.5,36.5,39])
        ax0.legend()

        ax1 = fig.add_subplot(1,2,2, projection= cart_proj)
        
        coast = NaturalEarthFeature(category="physical", scale="10m", facecolor="none", name="coastline")
        ax1.add_feature(
            coast,
            linewidth=1,
            edgecolor="black",
        )
        cont0 = ax1.contourf(
            wrf_lons, wrf_lats, np.log10(wrf_column).T, levels=levels, transform=ccrs.PlateCarree(), cmap="jet"
        )

        # Add a color bar
        cb = plt.colorbar(cont0, ax=ax1, shrink=0.95, fraction=0.046, pad=0.04)
        # cb.set_ticks(np.linspace(-4,1,7))
        cb.set_label("log[g/m3]")

        # Set the map bounds
        ax1.set_xlim(cartopy_xlim(qvap))
        ax1.set_ylim(cartopy_ylim(qvap))

        # Add the gridlines
        gl = ax1.gridlines(color="black", linestyle="dotted", draw_labels = True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'rotation': 0, 'ha':"center"}
        ax1.scatter(167, 153, color="k", marker="*", s=60)
        ax1.set_extent([20.5,23.5,36.5,39])

        ax0.set_title(f"ERA5 LWP at {time.strftime('%Y-%m-%d %H:%M')}")
        ax0.set_title(f"WRF LWP at {time.strftime('%Y-%m-%d %H:%M')}") 
        plt.tight_layout()


        plt.savefig(Path(save_path, f"ERA5_vs_WRF_LWP_{time.strftime('%Y-%m-%dH%H')}.png"))
        plt.close()


def main():
    era5_ncfile = Dataset("/work/lapi/waseem/ERA5/carmelle.nc")
    wrf_gridcell_ncfile = Dataset("/scratch/waseem/helmos_chopin_test/wrfout_Helmos_d03_2021-12-17_VL.nc")
    wrf_ncfile = Dataset("/scratch/waseem/helmos_chopin_test/wrfout_Helmos_d03_2021-12-17_00:00:00.nc")

    save_path = Path("/home/waseem/CHOPIN_analysis/figures/carmelle_comparison/")
    plot_ERA5_vs_WRF_LWC_at_grib_point(era5_ncfile, wrf_gridcell_ncfile, save_path)

    plot_ERA5_vs_WRF_LWP(era5_ncfile, wrf_ncfile, save_path)


if __name__ == "__main__":
    main()