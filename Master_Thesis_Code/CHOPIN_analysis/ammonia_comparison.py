# %%

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar

sys.path.append("./")
from utils import get_middled_geopotential_height
from WRFMultiDataset import WRFDataset

# %%
# load ammonia data

# ammonia_raw = pd.read_csv("./data/insitu_measurements/Ammonia.csv")
# ammonia = pd.DataFrame()
# ammonia["time"] = pd.to_datetime(ammonia_raw["DateTime.1"], format=r"%d-%m-%Y %H:%M:%S")
# ammonia["NH3"] = ammonia_raw["NH3 [ppb] original"]
# ammonia = ammonia.sort_values("time").reset_index(drop=False)

# ammonia_resample = ammonia.set_index("time").resample("15min").mean().reset_index()
dtypes = {
    "temp": "float",
    "temp_std": "float",
    "rh": "float",
    "rh_std": "float",
    "wind_dir": "float",
    "wind_dir_std": "float",
    "wind_speed": "float",
    "wind_speed_std": "float",
}
meteo = pd.read_csv(
    "./data/insitu_measurements/HAC_meteo_20240121.csv", parse_dates=["time"], dtype=dtypes, na_values="NAN"
)
meteo_times = set(meteo.time)

ammonia_hourly = pd.read_csv("./data/insitu_measurements/Ammonia_20241211_hourly.csv", parse_dates=["Date"])


# periods
rain_bounds = (
    pd.Timestamp(year=2024, month=10, day=7, hour=0),
    pd.Timestamp(year=2024, month=10, day=9, hour=0),
)
clear_bounds = (
    pd.Timestamp(year=2024, month=10, day=28, hour=18),
    pd.Timestamp(year=2024, month=10, day=30, hour=18),
)
envelop_bounds = (
    pd.Timestamp(year=2024, month=10, day=18, hour=18),
    pd.Timestamp(year=2024, month=10, day=20, hour=18),
)

rain_runs = {
    "YSU": "Yonsei Univeristy",
    "MYJ": "Mello-Yamada-Janjic",
    "KEPS": "K-epsilon-theta^2",
    "SH": "Shin-Hong",
    "MYJ25": "Mellor-Yamada Nakanishi\nand Niino level 2.5",
}

clear_runs = {
    "YSU": "Yonsei Univeristy",
    "MYNN": "Mellor-Yamada Nakanishi\nand Niino level 2.5",
    "KEPS": "K-epsilon-theta^2",
}

envelop_runs = {
    "YSU": "Yonsei Univeristy",
    "MYNN": "Mellor-Yamada Nakanishi\nand Niino level 2.5",
    "KEPS": "K-epsilon-theta^2",
}

colors = {
    "YSU": "cornflowerblue",
    "MYJ": "olivedrab",
    "KEPS": "darkorchid",
    "SH": "firebrick",
    "MYJ25": "forestgreen",
    "MYNN": "forestgreen",
    "THMP": "dodgerblue",
    "WDM6": "deepskyblue",
}

# %%


def plot_bl_comparisons(run_dict, bounds, general_path, ammonia, HAC_met):
    tick_locs = mdates.drange(bounds[0], bounds[1] + pd.Timedelta(6, "h"), pd.Timedelta(6, "h"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]
    fig, ax = plt.subplots(5, 1, figsize=(10, 12))
    mask = (ammonia.time >= bounds[0]) & (ammonia.time <= bounds[1])
    ax[0].plot(ammonia.loc[mask, "time"], ammonia.loc[mask, "NH3"])
    ax[0].set_ylabel("Ammonia [ppb]")
    ax[0].set_xticks(tick_locs)
    ax[0].set_xticklabels([])

    ncfile_NPRK = Dataset(f"{general_path}_NPRK_YSU.nc")
    ncfile_HAC = Dataset(f"{general_path}_HAC_YSU.nc")
    HGT_NPRK = np.squeeze(ncfile_NPRK.variables["HGT"][0])
    HGT_HAC = np.squeeze(ncfile_HAC.variables["HGT"][0])
    HGT_diff = HGT_HAC - HGT_NPRK

    HAC_mask = (HAC_met.time >= bounds[0]) & (HAC_met.time <= bounds[1])
    ax[2].plot(HAC_met.loc[HAC_mask, "time"], HAC_met.loc[HAC_mask, "temp"] + 273.15, color="k", label="HAC")
    ax[3].plot(HAC_met.loc[HAC_mask, "time"], HAC_met.loc[HAC_mask, "rh"], color="k", label="HAC")

    RMSE_T2 = {}
    RMSE_RH2 = {}

    for key, val in run_dict.items():
        ncfile = Dataset(f"{general_path}_NPRK_{key}.nc")
        wrf_times = pd.Series(getvar(ncfile, "Times", meta=False, timeidx=None))
        wrf_mask = (wrf_times >= bounds[0]) & (wrf_times <= bounds[1])
        pbl = np.squeeze(ncfile.variables["PBLH"][wrf_mask])
        ax[1].plot(wrf_times[wrf_mask], pbl, color=colors[key], label=val, alpha=0.7)
        ncfile_HAC = Dataset(f"{general_path}_HAC_{key}.nc")
        T2 = np.squeeze(ncfile_HAC.variables["T2"])
        rh2 = getvar(ncfile_HAC, "rh2", meta=False, timeidx=None)
        ax[2].plot(wrf_times[wrf_mask], T2[wrf_mask], color=colors[key], label=val)
        ax[3].plot(wrf_times[wrf_mask], rh2[wrf_mask], color=colors[key], label=val)

        shared_times = set(HAC_met.time).intersection(set(wrf_times[wrf_mask]))

        RMSE_mask_HAC = [x in shared_times for x in HAC_met.time]
        RMSE_mask_wrf = [x in shared_times for x in wrf_times]
        # Metrics
        RMSE_T2[key] = np.sqrt(
            np.mean((HAC_met.loc[RMSE_mask_HAC, "temp"] + 273.15 - T2[RMSE_mask_wrf]) ** 2)
        )
        RMSE_RH2[key] = np.sqrt(np.mean((HAC_met.loc[RMSE_mask_HAC, "rh"] - rh2[RMSE_mask_wrf]) ** 2))

    for a in ax:
        a.set_xticks(tick_locs)
        a.set_xticklabels([])

    ax[-1].set_xticklabels(tick_labels)
    ax[-1].set_xlabel("Date [UTC]")

    ax[1].axhline(HGT_diff, 0, 1, linestyle="-.", color="grey", label="HAC grid cell")
    ax[1].set_ylabel("Height above ground [m]")

    ax[1].legend(bbox_to_anchor=[1.01, 0.5], loc="center left", ncols=1)

    ax[4].pcolormesh(
        wrf_times[wrf_mask],
        get_middled_geopotential_height(ncfile_NPRK) / 1000,
        np.squeeze(ncfile_NPRK.variables["Zhh_BASTA"])[wrf_mask, :].T,
        vmin=-35,
        vmax=20,
        cmap="jet",
    )
    ax[4].set_ylabel("Height [km]")
    ax[4].set_ylim(0, 12)

    ax[0].set_title("Ammonia Measurements HAC")
    ax[1].set_title("PBLH from WRF NPRK")
    ax[2].set_title("2m Temperature HAC")
    ax[3].set_title("2m RH HAC")
    ax[4].set_title("Reflectivity from CR-SIM [dBZ] YSU")

    # put in metrics
    string_T2 = "RMSE 2m Temperature"
    string_RH2 = "RMSE 2m RH"
    for key in run_dict.keys():
        string_T2 += f"\n - {run_dict[key]}: {RMSE_T2[key]:.2f}"
        string_RH2 += f"\n - {run_dict[key]}: {RMSE_RH2[key]:.2f}"

    fig.text(0.775, 0.5, string_T2)
    fig.text(0.775, 0.3025, string_RH2)

    plt.tight_layout()


# # rain
# plot_bl_comparisons(
#     run_dict=rain_runs,
#     bounds=rain_bounds,
#     general_path="./data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06",
#     ammonia=ammonia_resample,
#     HAC_met=meteo,
# )

# # clear
# plot_bl_comparisons(
#     run_dict=clear_runs,
#     bounds=clear_bounds,
#     general_path="./data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27",
#     ammonia=ammonia_resample,
#     HAC_met=meteo,
# )

# # envelopment
# plot_bl_comparisons(
#     run_dict=envelop_runs,
#     bounds=envelop_bounds,
#     general_path="./data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17",
#     ammonia=ammonia_resample,
#     HAC_met=meteo,
# )

# %%

rain_runs = {
    "YSU": "YSU",
    "KEPS": "KEPS",
    "MYNN": "MYNN",
}

clear_runs = {
    "YSU": "YSU",
    "MYNN": "MYNN",
    "KEPS": "KEPS",
}

envelop_runs = {
    "YSU": "YSU",
    "MYNN": "MYNN",
    "KEPS": "KEPS",
}


data = []

legend_size = 13
small_size = 12
medium_size = 13
large_size = 14
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=legend_size)  # legend fontsize
plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
plt.rc("axes", titlesize= medium_size)
for run_dict, general_path, label, bounds in zip(
    [rain_runs, clear_runs, envelop_runs],
    [
        "./data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06",
        "./data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27",
        "./data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17",
    ],
    [1, 3, 2],
    [rain_bounds, clear_bounds, envelop_bounds],
):

    ncfile_NPRK = WRFDataset(Path(f"{general_path}_NPRK_YSU.nc"))
    ncfile_HAC = WRFDataset(Path(f"{general_path}_HAC_YSU.nc"))
    HGT_NPRK = np.squeeze(ncfile_NPRK._ncfile.variables["HGT"][0])
    HGT_HAC = np.squeeze(ncfile_HAC._ncfile.variables["HGT"][0])
    HGT_diff = HGT_HAC - HGT_NPRK
    meteo_times = set(meteo.time)

    wrf_times = set(ncfile_NPRK.times)
    intersect_times = wrf_times.intersection(meteo_times)
    wrf_time_mask = [x in intersect_times for x in ncfile_NPRK.times]
    # wrf_time_mask_wind_speed = wrf_time_mask
    meteo_time_mask = [x in intersect_times for x in meteo.time]

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    tick_locs = mdates.drange(bounds[0], bounds[1] + pd.Timedelta(6, "h"), pd.Timedelta(6, "h"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]
    for key, val in run_dict.items():
        ncfile = WRFDataset(Path(f"{general_path}_NPRK_{key}.nc"))
        pbl = np.squeeze(ncfile.variables("PBLH"))
        axs[1].plot(ncfile.times, pbl, color=colors[key], label=val, alpha=0.8, linewidth=2.5)
        ncfile_HAC = WRFDataset(Path(f"{general_path}_HAC_{key}.nc"))
        # T2 = np.squeeze(ncfile_HAC.variables("T2"))
        # rh2 = getvar(ncfile_HAC, "rh2", meta=False, timeidx=None)
        # ax[2].plot(ncfile.times, T2, color=colors[key], label=val)
        # ax[3].plot(ncfile.times, rh2, color=colors[key], label=val)

        shared_times = meteo_times.intersection(set(ncfile.times))

        RMSE_mask_HAC = [x in shared_times for x in meteo.time]
        RMSE_mask_wrf = [x in shared_times for x in wrf_times]
        # Metrics
        T_diff = ncfile_HAC.variables("T2")[wrf_time_mask] - (meteo.temp[meteo_time_mask].to_numpy() + 273.15)
        RMSE_T = np.sqrt(np.mean(T_diff**2))
        # relative humidity
        RH_diff = ncfile_HAC.getvar("rh2")[wrf_time_mask] - meteo.rh[meteo_time_mask].to_numpy()
        RMSE_RH = np.sqrt(np.mean((RH_diff) ** 2))
        # wind speed

        u = ncfile_HAC.variables("U10")
        v = ncfile_HAC.variables("V10")
        uv = np.sqrt(u**2 + v**2)
        uv_diff = uv[wrf_time_mask] - meteo.wind_speed[meteo_time_mask].to_numpy()
        RMSE_WS = np.sqrt(np.nanmean(uv_diff**2))

        # wind dir
        wind_direction_cart = np.arctan2(-v, -u) / np.pi * 180
        wind_direction_polar = (90 - wind_direction_cart + 360) % 360

        WD_diff_raw = uv[wrf_time_mask] - meteo.wind_dir[meteo_time_mask].to_numpy()
        WD_diff = (WD_diff_raw + 180) % 360 - 180
        # we add 180 and take the remainder of 360 so that values over 180 shrink
        # then we substract 180, some values will become negative, but since we
        # are going to square them it's okay

        RMSE_WD = np.sqrt(np.mean((WD_diff) ** 2))

        data.append(
            {
                "period": label,
                "PBL_Scheme": key,
                "RMSE_T": RMSE_T,
                "RMSE_RH": RMSE_RH,
                "RMSE_WS": RMSE_WS,
                "RMSE_WD": RMSE_WD,
            }
        )

    axs[1].set_xticks(tick_locs, tick_labels)
    axs[1].set_xlabel("Date [UTC]")

    axs[1].axhline(HGT_diff, 0, 1, linestyle="-.", color="grey", label="HAC grid cell", linewidth=2.5)
    axs[1].set_ylabel("Height above ground [m]")

    axs[1].legend(bbox_to_anchor=[0.5, -0.35], loc="upper center", ncols=4)

    mask = (ammonia_hourly.Date >= bounds[0]) & (ammonia_hourly.Date <= bounds[1])
    axs[0].plot(
        ammonia_hourly.loc[mask, "Date"], ammonia_hourly.loc[mask, "NH3 ppb (- replaced fo LOD/sqr2)"], linewidth=2.5
    )
    axs[0].set_ylabel("Ammonia [ppb]")
    axs[0].set_xticks([])

    axs[0].set_title("Ammonia Measurements HAC")
    axs[1].set_title("PBLH from WRF NPRK")
    axs[0].set_xlim(bounds[0], bounds[1])
    axs[1].set_xlim(bounds[0], bounds[1])

    file_name = f"{general_path.split('/')[2]}_bl_evolution.png"
    plt.savefig(f"./figures/final_plots/{file_name}", dpi=250, bbox_inches="tight")
    # plt.close()

df = pd.DataFrame(data)
var_name_map = {
    "RMSE_T": "Temperature [K]",
    "RMSE_RH": "Relative Humidity [%]",
    "RMSE_WS": "Wind Speed [m/s]",
    "RMSE_WD": "Wind Direction [deg]",
    "PBL_Scheme": "PBL Scheme",
}
df.set_index(["period", "PBL_Scheme"], inplace=True)
df.rename(columns=var_name_map, inplace=True)
df.to_csv("./figures/final_plots/met_variables_RMSE_BL_comparison.csv", float_format="%.2f")
