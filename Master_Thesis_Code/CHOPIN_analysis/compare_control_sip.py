# %%
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from pybaselines import Baseline
from wrf import getvar

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import (
    MIRAMultiDataset,
    MIRAMultiDatasetFactory,
    WRFDataset,
    WRFMultiDatasetFactory,
)


def add_time_skip_nans(df:pd.DataFrame, interval:int, time_col:str="time"):
    time_jumps = np.where(df.time.diff().apply(lambda x: x.total_seconds()) > interval)[0]
    df_fixed = df.copy()
    j = 0

    for idx in time_jumps:
        end_idx = idx + j
        new_row = {
            x: df_fixed.loc[end_idx-1, time_col] +pd.Timedelta(interval, "seconds") if x == time_col else np.nan for x in df.columns
        }
        df_fixed = pd.concat([df_fixed.loc[:end_idx-1],pd.DataFrame([new_row]), df_fixed.loc[end_idx:]]).reset_index(drop=True)
        j+=1

    return df_fixed
#%%

HAC_meteo_raw = pd.read_csv("data/insitu_measurements/HAC_meteo_20241211.csv", parse_dates=["time"])


HAC_meteo = add_time_skip_nans(HAC_meteo_raw, 3600)


ammonia_raw = pd.read_csv("./data/insitu_measurements/Ammonia_20241211_hourly.csv")

ammonia = pd.DataFrame()
ammonia["time"] = pd.to_datetime(ammonia_raw["Date"], format=r"%Y-%m-%d %H:%M")
ammonia["NH3"] = ammonia_raw["NH3 ppb (- replaced fo LOD/sqr2)"]
ammonia = ammonia.sort_values("time").reset_index(drop=True)

ammonia = add_time_skip_nans(ammonia, int(60*60))


POI_1_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_CONTROL/wrfout_CHPN_d03_2024-11-10_NPRK.nc"))
POI_1_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_SIP/wrfout_CHPN_d03_2024-11-10_NPRK.nc"))

POI_1_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_CONTROL/wrfout_CHPN_d03_2024-11-10_HAC.nc"))
POI_1_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_1_SIP/wrfout_CHPN_d03_2024-11-10_HAC.nc"))

POI_2_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_2_CONTROL/wrfout_CHPN_d03_2024-11-12_NPRK.nc"))
POI_2_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_2_CONTROL/wrfout_CHPN_d03_2024-11-12_HAC.nc"))

POI_3_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_NPRK.nc"))
POI_3_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_NPRK.nc"))

POI_3_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_HAC.nc"))
POI_3_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_HAC.nc"))

NOV_19_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_NPRK.nc"))
NOV_19_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22_SIP/wrfout_CHPN_d03_2024-11-19_NPRK.nc"))

NOV_19_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_HAC.nc"))
NOV_19_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov19-22_SIP/wrfout_CHPN_d03_2024-11-19_HAC.nc"))

NOV_21_CTRL_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_NPRK.nc"))
NOV_21_SIP_NPRK = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24_SIP/wrfout_CHPN_d03_2024-11-21_NPRK.nc"))

NOV_21_CTRL_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_HAC.nc"))
NOV_21_SIP_HAC = WRFDataset(Path("/scratch/waseem/CHOPIN_nov21-24_SIP/wrfout_CHPN_d03_2024-11-21_HAC.nc"))

MIRA_dataset_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA.csv"))

#%%
MIRA = MIRA_dataset_factory.get_dataset(pd.Timestamp(year=2024, month=11, day=10), pd.Timestamp(year=2024, month=11, day=24))

def plot_CONTROL_vs_SIP(
        MIRA:MIRAMultiDataset, 
        CONTROL_NPRK:WRFDataset, 
        CONTROL_HAC:WRFDataset, 
        SIP_NPRK:WRFDataset, 
        SIP_HAC:WRFDataset, 
        ammonia:pd.DataFrame, 
        HAC_meteo:pd.DataFrame,
        save:bool = False,
        ):
    fig, ax = plt.subplots(4, 2, figsize=(16,8))
    spinup_idx = int(24*60/5)
    control_times = pd.Series(CONTROL_NPRK.getvar("Times")[spinup_idx:])
    sip_times = pd.Series(SIP_NPRK.getvar("Times")[spinup_idx:])
    
    start_time = control_times.iloc[0]
    end_time = control_times.iloc[-1]

    MIRA_mask = (MIRA.times <= end_time) & (MIRA.times >= start_time)
    HAC_meteo_mask = (HAC_meteo.time <= end_time) & (HAC_meteo.time >= start_time)
    ammonia_mask = (ammonia.time <= end_time) & (ammonia.time >= start_time)

    tick_locs = mdates.drange(start_time, end_time + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

    NPRK_height = CONTROL_NPRK.variables("HGT")[0]
    HAC_height = CONTROL_HAC.variables("HGT")[0] -  NPRK_height

    control_rh2m = CONTROL_HAC.getvar("rh2")[spinup_idx:]
    control_t2m = CONTROL_HAC.variables("T2")[spinup_idx:] - 273.15
    control_blh = CONTROL_NPRK.variables("PBLH")[spinup_idx:]

    sip_rh2m = SIP_HAC.getvar("rh2")[spinup_idx:]
    sip_t2m = SIP_HAC.variables("T2")[spinup_idx:] - 273.15
    sip_blh = SIP_NPRK.variables("PBLH")[spinup_idx:]

    intersect_times = set(HAC_meteo.loc[HAC_meteo_mask, "time"]).intersection(set(control_times))
    HAC_RMSE_mask = [x in intersect_times for x in HAC_meteo.loc[HAC_meteo_mask, "time"]]
    wrf_RMSE_mask = [x in intersect_times for x in control_times]

    control_RMSE_t2m = np.sqrt(np.mean((control_t2m[wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "temp"][HAC_RMSE_mask].to_numpy())**2))
    control_RMSE_rh2m = np.sqrt(np.mean((control_rh2m[wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "rh"][HAC_RMSE_mask].to_numpy())**2))

    sip_RMSE_t2m = np.sqrt(np.mean((sip_t2m[wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "temp"][HAC_RMSE_mask].to_numpy())**2))
    sip_RMSE_rh2m = np.sqrt(np.mean((sip_rh2m[wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "rh"][HAC_RMSE_mask].to_numpy())**2))

    mira_step=10
    im0=ax[0,0].pcolormesh(
        MIRA.times[MIRA_mask][::mira_step],
        MIRA.range / 1000,
        MIRA.refl[MIRA_mask][::mira_step, :].T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im0)
    cbar.set_label("Reflectivity [dBz]")
    ax[0,0].set_ylim(1,12)
    ax[0,0].set_title("MIRA Radar")
    ax[0,0].set_ylabel("Altitude [km]")

    cs = ax[1,0].contour(
        control_times,
        CONTROL_NPRK.ZZ / 1000,
        (CONTROL_NPRK.kinetic_temp[spinup_idx:, :].T - 273.15),
        levels=np.arange(-70, 10, 10),
        colors="dimgray",
        linewidths=1,
        alpha=0.5,
    )
    ax[1,0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
    im1 = ax[1,0].pcolormesh(
        control_times,
        CONTROL_NPRK.ZZ / 1000,
        CONTROL_NPRK.variables("Zhh_MIRA")[spinup_idx:, :].T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im1)
    cbar.set_label("Reflectivity [dBz]")
    ax[1,0].set_ylim(1,12)
    ax[1,0].set_title("CONTROL simulation")
    ax[1,0].set_ylabel("Altitude [km]")

    cs = ax[2,0].contour(
        sip_times,
        SIP_NPRK.ZZ / 1000,
        (SIP_NPRK.kinetic_temp[spinup_idx:, :].T - 273.15),
        levels=np.arange(-70, 10, 10),
        colors="dimgray",
        linewidths=1,
        alpha=0.5,
    )
    ax[2,0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
    im1 = ax[2,0].pcolormesh(
        sip_times,
        SIP_NPRK.ZZ / 1000,
        SIP_NPRK.variables("Zhh_MIRA")[spinup_idx:, :].T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im1)
    cbar.set_label("Reflectivity [dBz]")
    ax[2,0].set_ylim(1,12)
    ax[2,0].set_title("SIP simulation")
    ax[2,0].set_ylabel("Altitude [km]")
    
    ax[3,0].plot(MIRA.times[MIRA_mask], MIRA.lwp[MIRA_mask], color="grey", label="MIRA")
    ax[3,0].plot(control_times, CONTROL_NPRK.lwp[spinup_idx:], color="k", label="Control")
    ax[3,0].plot(sip_times, SIP_NPRK.lwp[spinup_idx:], color="b", label="SIP")
    ax[3,0].set_ylabel("LWP [g/m3]")
    ax[3,0].legend()
    cbar = fig.colorbar(cs, ax=ax[3,0])
    cbar.ax.set_visible(False)


    


    ax[1,1].plot(HAC_meteo.loc[HAC_meteo_mask, "time"], HAC_meteo.loc[HAC_meteo_mask, "temp"], label="HAC", color="r")
    ax[1,1].plot(control_times, control_t2m, label="WRF Control", color="k")
    ax[1,1].plot(sip_times, sip_t2m, label="WRF SIP", color="b")
    ax[1,1].set_xticks(tick_locs)
    ax[1,1].set_xticklabels([])
    ax[1,1].set_ylabel("2m Temp [C]")
    ax[1,1].text(ax[1,1].get_xlim()[1] +0.02, ax[1,1].get_ylim()[0]+0.5, f"RMSE ctrl: {control_RMSE_t2m:.2f} C", rotation=90)
    ax[1,1].text(ax[1,1].get_xlim()[1] +0.07, ax[1,1].get_ylim()[0]+0.5, f"RMSE SIP: {sip_RMSE_t2m:.2f} C", rotation=90)
    ax[1,1].legend()
    
    ax[2,1].plot(HAC_meteo.loc[HAC_meteo_mask, "time"], HAC_meteo.loc[HAC_meteo_mask, "rh"], label="HAC", color="r")
    ax[2,1].plot(control_times, control_rh2m, label="WRF Control", color="k")
    ax[2,1].plot(sip_times, sip_rh2m, label="WRF SIP", color="b")
    ax[2,1].set_xticks(tick_locs)
    ax[2,1].set_xticklabels([])
    ax[2,1].set_ylabel("2m RH [%]")
    ax[2,1].set_ylim(1,105)
    ax[2,1].text(ax[2,1].get_xlim()[1] +0.02, ax[2,1].get_ylim()[0]+5, f"RMSE ctrl: {control_RMSE_rh2m:.2f} %", rotation=90)
    ax[2,1].text(ax[2,1].get_xlim()[1] +0.07, ax[2,1].get_ylim()[0]+5, f"RMSE sip: {sip_RMSE_rh2m:.2f} %", rotation=90)
    ax[2,1].legend()
      
    ax32 = ax[3,1].twinx()
    ln1=ax32.plot(ammonia.loc[ammonia_mask, "time"], ammonia.loc[ammonia_mask, "NH3"], label="NH3", color="r", alpha = 0.3)
    ax32.spines['right'].set_color('red')
    ax32.yaxis.label.set_color('red')
    ax32.tick_params(axis='y', colors='red')
    ax32.set_ylabel("NH3 [ppb]")
    
    ln2=ax[3,1].plot(control_times, control_blh, label="WRF Control PBLH", color='k', alpha=0.75)
    ln3=ax[3,1].plot(control_times, sip_blh, label="WRF SIP PBLH", color='b', alpha=0.75)
    ln4=ax[3,1].axhline(HAC_height, 0,1, color="purple", label="HAC height", linewidth=2, linestyle="dashed")
    ax[3,1].set_ylabel("Altitude [m]")
    ax[3,1].set_xticks(tick_locs)
    ax[3,1].set_xticklabels(tick_labels)
    
    lns = ln1+ln2+ln3+[ln4]
    labs = [l.get_label() for l in lns]
    ax[3,1].legend(lns, labs, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax[3,1].set_xlabel("Date [UTC]",labelpad=30)

    ax[0,1].axis("off")

    set_up_dated_x_axis(ax[:,0], tick_locs, tick_labels)
    set_up_dated_x_axis(ax[1:,1], tick_locs, tick_labels)
    plt.tight_layout()
    
    if save:
        plt.savefig(f"./figures/control_sip_comparison/{start_time.strftime(r'%Y%m%d')}-{end_time.strftime(r'%Y%m%d')}")


plot_CONTROL_vs_SIP(
    MIRA=MIRA, 
    CONTROL_NPRK=POI_1_CTRL_NPRK, 
    CONTROL_HAC=POI_1_CTRL_HAC, 
    SIP_NPRK=POI_1_SIP_NPRK, 
    SIP_HAC=POI_1_SIP_HAC,
    ammonia=ammonia,
    HAC_meteo=HAC_meteo,
    save=True
    )

plot_CONTROL_vs_SIP(
    MIRA=MIRA, 
    CONTROL_NPRK=POI_3_CTRL_NPRK, 
    CONTROL_HAC=POI_3_CTRL_HAC, 
    SIP_NPRK=POI_3_SIP_NPRK, 
    SIP_HAC=POI_3_SIP_HAC,
    ammonia=ammonia,
    HAC_meteo=HAC_meteo,
    save=True
    )

plot_CONTROL_vs_SIP(
    MIRA=MIRA, 
    CONTROL_NPRK=NOV_19_CTRL_NPRK, 
    CONTROL_HAC=NOV_19_CTRL_HAC, 
    SIP_NPRK=NOV_19_SIP_NPRK, 
    SIP_HAC=NOV_19_SIP_HAC,
    ammonia=ammonia,
    HAC_meteo=HAC_meteo,
    save=True
    )

plot_CONTROL_vs_SIP(
    MIRA=MIRA, 
    CONTROL_NPRK=NOV_21_CTRL_NPRK, 
    CONTROL_HAC=NOV_21_CTRL_HAC, 
    SIP_NPRK=NOV_21_SIP_NPRK, 
    SIP_HAC=NOV_21_SIP_HAC,
    ammonia=ammonia,
    HAC_meteo=HAC_meteo,
    save=True
    )

#%%

# test redo

start_time = pd.Timestamp(year=2024, month=11, day=13)
end_time = pd.Timestamp(year=2024, month=11, day=15)
MIRA = MIRA_dataset_factory.get_dataset(start_time, end_time)
wrf_dataset_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))
old_WRF_NPRK = wrf_dataset_factory.get_dataset(
    start_time=start_time, 
    end_time=end_time, 
    station="NPRK", 
    mp_phys=10, 
    bl_phys=5, 
    sip=False,
    )

old_WRF_HAC = wrf_dataset_factory.get_dataset(
    start_time=start_time, 
    end_time=end_time, 
    station="HAC", 
    mp_phys=10, 
    bl_phys=5, 
    sip=False,
    )
wrf_times = pd.Series(old_WRF_HAC.times)
tick_locs = mdates.drange(start_time, end_time + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

fig, ax = plt.subplots(6, 1, figsize = (9,10))

ax[0].pcolormesh(
    MIRA.times[::10], 
    MIRA.range/1000,
    MIRA.refl[::10,:].T,
    vmin=-60,
    vmax=35,
    cmap="jet",
)
ax[0].set_title("MIRA radar")
ax[0].set_ylim(1,12)
ax[0].set_ylabel("Altitude [km]")

ax[1].pcolormesh(
    old_WRF_NPRK.times, 
    old_WRF_NPRK.ZZ/1000, 
    old_WRF_NPRK.variables("Zhh_MIRA").T,
    vmin=-60,
    vmax=35,
    cmap="jet",
    )
ax[1].set_ylim(1,12)
ax[1].set_title("Old simulation")
ax[1].set_ylabel("Altitude [km]")
ax[1].axvline(pd.Timestamp(year=2024, month=11, day=14), 0, 12, color="magenta", linewidth=3)

ax[2].pcolormesh(
    POI_2_CTRL_NPRK.times[288:], 
    POI_2_CTRL_NPRK.ZZ/1000, 
    POI_2_CTRL_NPRK.variables("Zhh_MIRA")[288:, :].T,
    vmin=-60,
    vmax=35,
    cmap="jet",
    )
ax[2].set_ylim(1,12)
ax[2].set_title("re-run simulation")
ax[2].set_ylabel("Altitude [km]")


old_rh2m = old_WRF_HAC.getvar("rh2")
old_t2m = old_WRF_HAC.variables("T2") - 273.15
old_blh = old_WRF_NPRK.variables("PBLH")

new_rh2m = POI_2_CTRL_HAC.getvar("rh2")
new_t2m = POI_2_CTRL_HAC.variables("T2") - 273.15
new_blh = POI_2_CTRL_NPRK.variables("PBLH")
HAC_meteo_mask = (HAC_meteo.time<=end_time) & (HAC_meteo.time>=start_time)
ammonia_mask = (ammonia.time<=end_time) & (ammonia.time>=start_time)

intersect_times = set(HAC_meteo.loc[HAC_meteo_mask, "time"]).intersection(set(wrf_times))
HAC_RMSE_mask = [x in intersect_times for x in HAC_meteo.loc[HAC_meteo_mask, "time"]]
wrf_RMSE_mask = [x in intersect_times for x in wrf_times]

old_RMSE_t2m = np.sqrt(np.nanmean((old_t2m[wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "temp"][HAC_RMSE_mask].to_numpy())**2))
old_RMSE_rh2m = np.sqrt(np.nanmean((old_rh2m[wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "rh"][HAC_RMSE_mask].to_numpy())**2))
new_RMSE_t2m = np.sqrt(np.nanmean((new_t2m[288:][wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "temp"][HAC_RMSE_mask].to_numpy())**2))
new_RMSE_rh2m = np.sqrt(np.nanmean((new_rh2m[288:][wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "rh"][HAC_RMSE_mask].to_numpy())**2))

HAC_height = POI_2_CTRL_HAC.variables("HGT")[0] - POI_2_CTRL_NPRK.variables("HGT")[0]


ax[3].plot(HAC_meteo.loc[HAC_meteo_mask, "time"], HAC_meteo.loc[HAC_meteo_mask, "temp"], label="HAC", color="r")
ax[3].plot(wrf_times, old_t2m, label="Old Simulations", color="k")
ax[3].plot(POI_2_CTRL_HAC.times[288:], new_t2m[288:], label="Re-run Simulaion", color="b")
ax[3].set_xticks(tick_locs)
ax[3].set_xticklabels([])
ax[3].set_ylabel("2m Temp [C]")
ax[3].text(ax[3].get_xlim()[1] +0.02, ax[3].get_ylim()[0]+0.5, f"RMSE old: {old_RMSE_t2m:.2f} C", rotation=90)
ax[3].text(ax[3].get_xlim()[1] +0.07, ax[3].get_ylim()[0]+0.5, f"RMSE new: {new_RMSE_t2m:.2f} C", rotation=90)
ax[3].legend()

ax[4].plot(HAC_meteo.loc[HAC_meteo_mask, "time"], HAC_meteo.loc[HAC_meteo_mask, "rh"], label="HAC", color="r")
ax[4].plot(wrf_times, old_rh2m, label="Old_simulations", color="k")
ax[4].plot(POI_2_CTRL_HAC.times[288:], new_rh2m[288:], label="Re-run Simulation", color="b")
ax[4].set_xticks(tick_locs)
ax[4].set_xticklabels([])
ax[4].set_ylabel("RH [%]")
ax[4].set_ylim(1,105)
ax[4].text(ax[4].get_xlim()[1] +0.02, ax[4].get_ylim()[0]+5, f"RMSE old: {old_RMSE_rh2m:.2f} %", rotation=90)
ax[4].text(ax[4].get_xlim()[1] +0.07, ax[4].get_ylim()[0]+5, f"RMSE new: {new_RMSE_rh2m:.2f} %", rotation=90)
ax[4].legend()
    
ax32 = ax[5].twinx()
ln1=ax32.plot(ammonia.loc[ammonia_mask, "time"], ammonia.loc[ammonia_mask, "NH3"], label="NH3", color="r", alpha = 0.3)
ax32.spines['right'].set_color('red')
ax32.yaxis.label.set_color('red')
ax32.tick_params(axis='y', colors='red')
ax32.set_ylabel("NH3 [ppb]")

ln2=ax[5].plot(wrf_times, old_blh, label="Old Simulation PBLH", color='k', alpha=0.75)
ln3=ax[5].plot(POI_2_CTRL_HAC.times[288:], new_blh[288:], label="Re-run Simulation PBLH", color='b', alpha=0.75)
ln4=ax[5].axhline(HAC_height, 1,1, color="purple", label="HAC height", linewidth=2, linestyle="dashed")
ax[5].set_ylabel("Height above NPRK [m]")
ax[5].set_xticks(tick_locs)
ax[5].set_xticklabels(tick_labels)

lns = ln1+ln2+ln3+[ln4]
labs = [l.get_label() for l in lns]
ax[5].legend(lns, labs, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.3))
ax[5].set_xlabel("Date [UTC]",labelpad=35)

set_up_dated_x_axis(ax, tick_locs, tick_labels)
plt.tight_layout()