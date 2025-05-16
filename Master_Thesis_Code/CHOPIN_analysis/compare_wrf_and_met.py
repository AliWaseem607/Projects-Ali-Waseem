# %%
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from pybaselines import Baseline

sys.path.append("./")
from WRFMultiDataset import (
    BASTAMultiDataset,
    BASTAMultiDatasetFactory,
    MIRAMultiDataset,
    MIRAMultiDatasetFactory,
    WRFMultiDataset,
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
# Load data
wrf_dataset_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))
wrf_dataset_HAC = wrf_dataset_factory.get_dataset(
    start_time=pd.Timestamp(year=2024, month=11, day=10),
    end_time=pd.Timestamp(year=2024, month=11, day=24),
    mp_phys=10,
    bl_phys=5,
    sip=False,
    station="HAC"
)

wrf_dataset_NPRK = wrf_dataset_factory.get_dataset(
    start_time=pd.Timestamp(year=2024, month=11, day=10),
    end_time=pd.Timestamp(year=2024, month=11, day=24),
    mp_phys=10,
    bl_phys=5,
    sip=False,
    station="NPRK"
)

HAC_meteo_raw = pd.read_csv("data/insitu_measurements/HAC_meteo_20241211.csv", parse_dates=["time"])


HAC_meteo = add_time_skip_nans(HAC_meteo_raw, 3600)


ammonia_raw = pd.read_csv("./data/insitu_measurements/Ammonia_20241211.csv")
ammonia = pd.DataFrame()
ammonia["time"] = pd.to_datetime(ammonia_raw["DateTime"], format=r"%d-%m-%Y %H:%M:%S")
ammonia["NH3"] = ammonia_raw["NH3 [ppb] original"]
ammonia = ammonia.sort_values("time").reset_index(drop=True)

ammonia_resample = ammonia.set_index("time").resample("15min").mean().reset_index()

ammonia_resample = add_time_skip_nans(ammonia_resample, int(15*60))

#%%
def plot_met_comparion(
        wrf_dataset_HAC:WRFMultiDataset, 
        wrf_dataset_NPRK:WRFMultiDataset, 
        HAC_meteo:pd.DataFrame, 
        ammonia:pd.DataFrame,
        plot_time:pd.Timedelta = pd.Timedelta(2, "d"),
        save:bool = False
        ) -> None:

    wrf_times_HAC = pd.Series(wrf_dataset_HAC.getvar("Times"))
    wrf_times_NPRK = pd.Series(wrf_dataset_NPRK.getvar("Times"))

    NPRK_height = wrf_dataset_NPRK.variables("HGT")[0]
    HAC_height = wrf_dataset_HAC.variables("HGT")[0] - NPRK_height
    


    assert np.all(wrf_times_HAC == wrf_times_NPRK)
    assert wrf_dataset_HAC.start_time is not None
    assert wrf_dataset_HAC.end_time is not None
    assert wrf_dataset_NPRK.start_time is not None
    assert wrf_dataset_NPRK.end_time is not None

    wrf_times = wrf_times_HAC
    del wrf_times_HAC
    del wrf_times_NPRK
    start_time = wrf_dataset_HAC.start_time
    end_time = wrf_dataset_HAC.end_time

    total_time = end_time - start_time

    rh2m = wrf_dataset_HAC.getvar("rh2")
    t2m = wrf_dataset_HAC.variables("T2") - 273.15
    blh = wrf_dataset_NPRK.variables("PBLH")
    wrf_refl_MIRA = wrf_dataset_NPRK.variables("Zhh_MIRA")

    # ammonia_drop_na = ammonia.dropna()
    # ammonia_overall_mask = (ammonia_drop_na.time >=start_time) & (ammonia_drop_na.time <= end_time)
    # ammonia_slice = ammonia_drop_na[ammonia_overall_mask]
    # baseline_fitter = Baseline(x_data=[(x-ammonia_slice["time"].iloc[0]).total_seconds() for x in ammonia_slice["time"]])
    # asls, params = baseline_fitter.asls(ammonia_slice["NH3"], lam=1e5, p=0.3)


    for i in range(int(np.ceil(total_time / plot_time))):
        plot_start = start_time + plot_time * i
        plot_end = np.min([end_time, start_time + plot_time * (i + 1)])  # type: ignore

        tick_locs = mdates.drange(plot_start, plot_end + pd.Timedelta(1, "h"), pd.Timedelta(6, "hours"))
        tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

        wrf_mask = (wrf_times >= plot_start) & (wrf_times <= plot_end)
        HAC_meteo_mask = (HAC_meteo.time >=plot_start) & (HAC_meteo.time <= plot_end)
        ammonia_mask = (ammonia.time >=plot_start) & (ammonia.time <= plot_end)
        # ammonia_slice_mask = (ammonia_slice.time >=plot_start) & (ammonia_slice.time <= plot_end)

        intersect_times = set(HAC_meteo.loc[HAC_meteo_mask, "time"]).intersection(set(wrf_times[wrf_mask]))
        HAC_RMSE_mask = [x in intersect_times for x in HAC_meteo.loc[HAC_meteo_mask, "time"]]
        wrf_RMSE_mask = [x in intersect_times for x in wrf_times[wrf_mask]]

        RMSE_t2m = np.sqrt(np.mean((t2m[wrf_mask][wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "temp"][HAC_RMSE_mask])**2))
        RMSE_rh2m = np.sqrt(np.mean((rh2m[wrf_mask][wrf_RMSE_mask] - HAC_meteo.loc[HAC_meteo_mask, "rh"][HAC_RMSE_mask])**2))

        fig, ax = plt.subplots(4, 1, figsize = (8,7))

        cs = ax[0].contour(
                wrf_times[wrf_mask],
                wrf_dataset_NPRK.ZZ / 1000,
                (wrf_dataset_NPRK.kinetic_temp[wrf_mask, :].T - 273.15),
                levels=np.arange(-70, 10, 5),
                colors="dimgray",
                linewidths=1,
                alpha=0.5,
            )
        ax[0].clabel(cs, inline=True, fontsize=12, fmt=r"%d$^\circ$C", colors="dimgrey")
        im1 = ax[0].pcolormesh(
            wrf_times[wrf_mask],
            wrf_dataset_NPRK.ZZ / 1000,
            wrf_refl_MIRA[wrf_mask, :].T,
            vmin=-60,
            vmax=35,
            cmap="jet",
        )
        # cbar = plt.colorbar(im1)
        # cbar.set_label("Reflectivity [dBz]")

        ax[0].set_ylim(0, 12)
        ax[0].set_ylabel("Altitude [km]")
        ax[0].set_title(f"{plot_start.strftime(r'%d-%m')} to {plot_end.strftime(r'%d-%m')}, bl: {wrf_dataset_NPRK.bl_phys}")
        ax[0].set_xticks(tick_locs)
        ax[0].set_xticklabels([])
        ax[0].text(ax[0].get_xlim()[0]+0.05, ax[0].get_ylim()[1]-2, "WRF Prediction", bbox={"facecolor":"white", "alpha":0.3})



        ax[1].plot(HAC_meteo.loc[HAC_meteo_mask, "time"], HAC_meteo.loc[HAC_meteo_mask, "temp"], label="HAC", color="r")
        ax[1].plot(wrf_times[wrf_mask], t2m[wrf_mask], label="WRF", color="k")
        ax[1].set_xticks(tick_locs)
        ax[1].set_xticklabels([])
        ax[1].set_ylabel("2m Temp [C]")
        ax[1].text(ax[1].get_xlim()[1] +0.05, ax[1].get_ylim()[0]+1.5, f"RMSE: {RMSE_t2m:.2f} C", rotation=90)
        ax[1].legend()
        
        
        ax[2].plot(HAC_meteo.loc[HAC_meteo_mask, "time"], HAC_meteo.loc[HAC_meteo_mask, "rh"], label="HAC", color="r")
        ax[2].plot(wrf_times[wrf_mask], rh2m[wrf_mask], label="WRF", color="k")
        ax[2].set_xticks(tick_locs)
        ax[2].set_xticklabels([])
        ax[2].set_ylabel("RH [%]")
        ax[2].set_ylim(0,105)
        ax[2].text(ax[1].get_xlim()[1] +0.05, ax[1].get_ylim()[0]+15, f"RMSE: {RMSE_rh2m:.2f} %", rotation=90)
        ax[2].legend()
        
        

        ax32 = ax[3].twinx()
        ln1=ax32.plot(ammonia.loc[ammonia_mask, "time"], ammonia.loc[ammonia_mask, "NH3"], label="NH3", color="r", alpha = 0.3)
        ax32.spines['right'].set_color('red')
        ax32.yaxis.label.set_color('red')
        ax32.tick_params(axis='y', colors='red')
        ax32.set_ylabel("NH3 [ppb]")
        
        ln2=ax[3].plot(wrf_times[wrf_mask], blh[wrf_mask], label="WRF PBLH", color='k')
        ln3=ax[3].axhline(HAC_height, 0,1, color="b", label="HAC height", linewidth=2, linestyle="dashed")
        ax[3].set_ylabel("Height above NPRK [m]")
        ax[3].set_xticks(tick_locs)
        ax[3].set_xticklabels(tick_labels)
        
        lns = ln1+ln2+[ln3]
        labs = [l.get_label() for l in lns]
        ax[3].legend(lns, labs, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
        ax[3].set_xlabel("Date [UTC]",labelpad=25)

        plt.tight_layout()
        if save:
            plt.savefig(f"./figures/met_comparison/temp_rh_blh_bl{wrf_dataset_NPRK.bl_phys}_{plot_start.strftime(r'%Y%m%d')}-{plot_end.strftime(r'%Y%m%d')}")

plot_met_comparion(
    wrf_dataset_HAC=wrf_dataset_HAC, 
    wrf_dataset_NPRK=wrf_dataset_NPRK, 
    HAC_meteo=HAC_meteo, 
    ammonia=ammonia_resample,
    save=True
    )
# %%
