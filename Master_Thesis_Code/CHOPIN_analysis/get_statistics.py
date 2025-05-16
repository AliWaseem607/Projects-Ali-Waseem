# %%
import sys
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

sys.path.append("./")
from utils import set_up_dated_x_axis
from WRFMultiDataset import (
    MIRAMultiDatasetFactory,
    WRFDataset,
    WRFMultiDataset,
    WRFMultiDatasetFactory,
)

# %%
mapping_21 = {5: 0, 2: 1, 4: 2, 0: 3, 1: 4, 3: 5}

clusters = pd.read_csv("./figures/cluster_analysis_21/data.csv", parse_dates=["start_time", "end_time"])
label_mapping = mapping_21
clusters["label_standardized"] = -1
for key, val in label_mapping.items():
    clusters.loc[clusters.label == key, "label_standardized"] = val

clusters.sort_values("start_time", inplace=True)
clusters.reset_index(inplace=True, drop=True)
clusters.drop("Unnamed: 0", axis=1, inplace=True)

wrf_dataset_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))
MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))

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
# meteo_dropna_times = set(meteo.dropna().time)

setup_1 = {"bl": 5, "mp": 10, "sip": True, "name": "WRF-SIP-MYNN"}
setup_2 = {"bl": 5, "mp": 10, "sip": False, "name": "WRF-CONTROL-MYNN"}
setup_3 = {"bl": 1, "mp": 10, "sip": True, "name": "WRF-SIP-YSU"}
setups = [setup_2, setup_1, setup_3]


def get_diff_values_HAC(meteo: pd.DataFrame, wrf_dataset: WRFDataset | WRFMultiDataset) -> dict[str, float]:
    # get the overlapped points
    results = {}
    meteo_times = set(meteo.time)
    wrf_times = set(wrf_dataset.times)
    intersect_times = wrf_times.intersection(meteo_times)
    wrf_time_mask = [x in intersect_times for x in wrf_dataset.times]
    meteo_time_mask = [x in intersect_times for x in meteo.time]
    T_diff = wrf_dataset.variables("T2")[wrf_time_mask] - (meteo.temp[meteo_time_mask].to_numpy() + 273.15)
    results["T"] = T_diff
    # relative humidity
    RH_diff = wrf_dataset.getvar("rh2")[wrf_time_mask] - meteo.rh[meteo_time_mask].to_numpy()
    results["RH"] = RH_diff
    # wind speed

    u = wrf_dataset.variables("U10")
    v = wrf_dataset.variables("V10")
    uv = np.sqrt(u**2 + v**2)
    uv_diff = uv[wrf_time_mask] - meteo.wind_speed[meteo_time_mask].to_numpy()
    results["WS"] = uv_diff

    # wind dir
    wind_direction_cart = np.arctan2(-v, -u) / np.pi * 180
    wind_direction_polar = (90 - wind_direction_cart + 360) % 360

    WD_diff_raw = wind_direction_polar[wrf_time_mask] - meteo.wind_dir[meteo_time_mask].to_numpy()
    WD_diff = (WD_diff_raw + 180) % 360 - 180
    # we add 180 and take the remainder of 360 so that values over 180 shrink
    # then we substract 180, some values will become negative, but since we
    # are going to square them it's okay

    results["WD"] = WD_diff

    return results


legend_size = 10
small_size = 11
medium_size = 12
large_size = 13
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=legend_size)  # legend fontsize
plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
plt.rc("axes", titlesize=medium_size)
# %%

last_date = meteo.time.max()
# get clustered statistics
cluster_stats = clusters[["start_time", "end_time"]].copy()
cluster_stats["label"] = clusters.label_standardized
cluster_stats["RMSE_T"] = np.nan
cluster_stats["RMSE_RH"] = np.nan
cluster_stats["RMSE_WD"] = np.nan
cluster_stats["RMSE_WS"] = np.nan

cluster_stats_1 = cluster_stats.copy()
cluster_stats_2 = cluster_stats.copy()
cluster_stats_3 = cluster_stats.copy()

cluster_errors_1 = {c: {"T": [], "RH": [], "WD": [], "WS": []} for c in range(6)}
cluster_errors_2 = {c: {"T": [], "RH": [], "WD": [], "WS": []} for c in range(6)}
cluster_errors_3 = {c: {"T": [], "RH": [], "WD": [], "WS": []} for c in range(6)}

del cluster_stats

all_cluster_stats = [cluster_stats_1, cluster_stats_2, cluster_stats_3]
cluster_errors = [cluster_errors_1, cluster_errors_2, cluster_errors_3]

# for each set up style
for setup, cluster_stats, errors in zip(setups, all_cluster_stats, cluster_errors):
    # get the periods that we have this set up
    periods = wrf_dataset_factory.get_periods(
        station="NPRK", mp_phys=setup["mp"], bl_phys=setup["bl"], sip=setup["sip"]
    )
    clusters_filled = 0
    # for each period
    for period in periods:
        period_start = period[0]
        period_end = period[1]
        # if the period is past the last date skip it
        if period_end > last_date:
            continue
        # filter the cluster values now
        mask = (cluster_stats.start_time < period_end) & (cluster_stats.end_time > period_start)
        cluster_slice = cluster_stats.loc[mask, :]
        # go through each row that is within the period and get statistics
        for i, row in cluster_slice.iterrows():
            # check is there is at least 75% of cluster is covered by period
            # each cluster is 12 hours so if the period starts after the first 3 hours
            # it cannot cover 75% of the time
            if period[0] > (row.start_time + pd.Timedelta(3, "h")):
                continue

            # same thing in reverses
            if period[1] < (row.end_time - pd.Timedelta(3, "h")):
                continue

            # check that there is enough weather data
            # the weather data is also in 5 minute increments
            mask_meteo = (meteo.time >= period_start) & (meteo.time <= period_end)
            # 10 hours as 5 minute increments
            required_points = 12 * 60 / 5
            if len(meteo[mask_meteo].dropna()) < required_points * 0.75:
                continue
            # now we have enough data to calculate a statistic
            idx = int(i)
            # get wrf data
            wrf_dataset = wrf_dataset_factory.get_dataset(
                start_time=period_start,
                end_time=period_end,
                station="HAC",
                mp_phys=setup["mp"],
                bl_phys=setup["bl"],
                sip=setup["sip"],
            )

            diff_results = get_diff_values_HAC(meteo, wrf_dataset)
            errors[row.label]["T"].append(diff_results["T"])
            errors[row.label]["RH"].append(diff_results["RH"])
            errors[row.label]["WS"].append(diff_results["WS"])
            errors[row.label]["WD"].append(diff_results["WD"])

            cluster_stats.loc[idx, "RMSE_T"] = np.sqrt(np.mean(diff_results["T"] ** 2))
            cluster_stats.loc[idx, "RMSE_RH"] = np.sqrt(np.mean((diff_results["RH"]) ** 2))
            cluster_stats.loc[idx, "RMSE_WS"] = np.sqrt(np.nanmean(diff_results["WS"] ** 2))
            cluster_stats.loc[idx, "RMSE_WD"] = np.sqrt(np.mean((diff_results["WD"]) ** 2))

# %%


def get_all_results_arr(errors: dict[int, dict[str, list]]) -> np.ndarray:
    T_grouped = []
    RH_grouped = []
    WS_grouped = []
    WD_grouped = []
    for i in range(6):
        T_grouped += errors[i]["T"]
        RH_grouped += errors[i]["RH"]
        WS_grouped += errors[i]["WS"]
        WD_grouped += errors[i]["WD"]

    T_arr = np.concatenate(T_grouped)
    RH_arr = np.concatenate(RH_grouped)
    WS_arr = np.concatenate(WS_grouped)
    WD_arr = np.concatenate(WD_grouped)

    results = np.stack([T_arr, RH_arr, WD_arr, WS_arr])
    return results


cluster_results_1 = get_all_results_arr(cluster_errors_1)
cluster_results_2 = get_all_results_arr(cluster_errors_2)
cluster_results_3 = get_all_results_arr(cluster_errors_3)

# Save intermediate results
np.save("./data/intermediate_results/cluster_errors_1.npy", cluster_results_1)
np.save("./data/intermediate_results/cluster_errors_2.npy", cluster_results_2)
np.save("./data/intermediate_results/cluster_errors_3.npy", cluster_results_3)

cluster_stats_1.to_csv(Path("./data/intermediate_results/cluster_stats_1.csv"), index=False)
cluster_stats_2.to_csv(Path("./data/intermediate_results/cluster_stats_2.csv"), index=False)
cluster_stats_3.to_csv(Path("./data/intermediate_results/cluster_stats_3.csv"), index=False)


# %%
# Make box plots for RMSE of each cluster and setup
def plot_color_boxplot_RMSE(
    ax,
    cluster_stats: pd.DataFrame,
    var: str,
    color: str,
    offset: float,
    pos_multiplier: float = 3,
    width: float = 0.25,
):
    clusters = cluster_stats.dropna(subset="RMSE_T").label.unique()
    positions = [x * pos_multiplier + offset for x in clusters]
    data = []
    for i in clusters:
        mask_label = cluster_stats.label == i
        mask_na = ~np.isnan(cluster_stats[var])
        mask = mask_label & mask_na
        data.append(cluster_stats.loc[mask, var].to_numpy())

    for d, pos in zip(data, positions):
        if d.shape[0] < 5:
            ax.scatter([pos for _ in range(d.shape[0])], d, color=color)
            continue
        bp = ax.boxplot(d, positions=[pos], widths=[width], whis=(0, 100))
        # bp = ax.violinplot(d, positions=[pos], widths=[width - 0.1])

        for item in bp.keys():
            plt.setp(bp[item], color=color, linewidth=1.75)


fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))

for ax, var in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], ["RMSE_T", "RMSE_RH", "RMSE_WS", "RMSE_WD"]):
    plot_color_boxplot_RMSE(
        ax=ax,
        cluster_stats=cluster_stats_2,
        var=var,
        color="tab:blue",
        offset=-0.9,
        pos_multiplier=3,
        width=0.65,
    )
    plot_color_boxplot_RMSE(
        ax=ax,
        cluster_stats=cluster_stats_1,
        var=var,
        color="tab:red",
        offset=0,
        pos_multiplier=3,
        width=0.65,
    )
    plot_color_boxplot_RMSE(
        ax=ax,
        cluster_stats=cluster_stats_3,
        var=var,
        color="tab:orange",
        offset=0.9,
        pos_multiplier=3,
        width=0.65,
    )

axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Relative Humidity\nRMSE [%]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")


for ax in axs.flatten():
    ax.set_xticks([0, 3, 6, 9, 12, 15], [1, 2, 3, 4, 5, 6])
    ax.set_xlabel("Cluster")
    ax.set_xlim(-2, 17)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-ymax * 0.05, ymax)


setup_1_handle = Patch(color="tab:red", label="MYNN SIP")
setup_2_handle = Patch(color="tab:blue", label="MYNN CONTROL")
setup_3_handle = Patch(color="tab:orange", label="YSU SIP")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle, setup_3_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1, 1.25),
)
plt.savefig("./figures/final_plots/met_RMSE_box_plot_by_cluster.png", bbox_inches="tight", dpi=250)


# %%
# Make violin plots for RMSE of each cluster and setup
def plot_color_violinplot_RMSE(
    ax,
    cluster_stats: pd.DataFrame,
    var: str,
    color: str,
    offset: float,
    pos_multiplier: float = 3,
    width: float = 0.25,
):
    clusters = cluster_stats.dropna(subset="RMSE_T").label.unique()
    positions = [x * pos_multiplier + offset for x in clusters]
    data = []
    for i in clusters:
        mask_label = cluster_stats.label == i
        mask_na = ~np.isnan(cluster_stats[var])
        mask = mask_label & mask_na
        data.append(cluster_stats.loc[mask, var].to_numpy())

    for d, pos in zip(data, positions):
        if d.shape[0] < 5:
            ax.scatter([pos for _ in range(d.shape[0])], d, color=color)
            continue
        # bp = ax.violinplot(d, positions=[pos], widths=[width], whis=999)
        bp = ax.violinplot(d, positions=[pos], widths=[width - 0.1])

        for item in bp.keys():
            plt.setp(bp[item], color=color)


fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))

for ax, var in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], ["RMSE_T", "RMSE_RH", "RMSE_WS", "RMSE_WD"]):
    plot_color_violinplot_RMSE(
        ax=ax,
        cluster_stats=cluster_stats_2,
        var=var,
        color="tab:blue",
        offset=-0.8,
        pos_multiplier=3,
        width=0.65,
    )
    plot_color_violinplot_RMSE(
        ax=ax,
        cluster_stats=cluster_stats_1,
        var=var,
        color="tab:red",
        offset=0,
        pos_multiplier=3,
        width=0.65,
    )
    plot_color_violinplot_RMSE(
        ax=ax,
        cluster_stats=cluster_stats_3,
        var=var,
        color="tab:orange",
        offset=0.8,
        pos_multiplier=3,
        width=0.65,
    )

axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Relative Humidity\nRMSE [%]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")


for ax in axs.flatten():
    ax.set_xticks([0, 3, 6, 9, 12, 15], [1, 2, 3, 4, 5, 6])
    ax.set_xlabel("Cluster")
    ax.set_xlim(-2, 17)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-ymax * 0.05, ymax)


setup_1_handle = Patch(color="tab:red", label="MYNN SIP")
setup_2_handle = Patch(color="tab:blue", label="MYNN CONTROL")
setup_3_handle = Patch(color="tab:orange", label="YSU SIP")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle, setup_3_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1, 1.2),
)
plt.savefig("./figures/final_plots/met_RMSE_violin_plot_by_cluster.png", bbox_inches="tight", dpi=250)


# %%
# Make cluster plots without WRF-SIP-YSU and cluster 3
def plot_color_boxplot_RMSE_2(
    ax,
    cluster_stats: pd.DataFrame,
    var: str,
    color: str,
    offset: float,
    pos_multiplier: float = 3,
    width: float = 0.25,
):
    clusters = cluster_stats.dropna(subset="RMSE_T").label.unique()
    # positions = [x * pos_multiplier + offset for x in clusters]
    positions = []
    for x in clusters:
        if x < 3:
            positions.append(x * pos_multiplier + offset)
        else:
            positions.append((x - 1) * pos_multiplier + offset)
    data = []
    for i in clusters:
        mask_label = cluster_stats.label == i
        mask_na = ~np.isnan(cluster_stats[var])
        mask = mask_label & mask_na
        data.append(cluster_stats.loc[mask, var].to_numpy())

    for d, pos in zip(data, positions):
        if d.shape[0] < 5:
            ax.scatter([pos for _ in range(d.shape[0])], d, color=color)
            continue
        bp = ax.boxplot(d, positions=[pos], widths=[width], whis=(0, 100))

        for item in bp.keys():
            plt.setp(bp[item], color=color, linewidth=1.75)


fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))

for ax, var in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], ["RMSE_T", "RMSE_RH", "RMSE_WS", "RMSE_WD"]):
    plot_color_boxplot_RMSE_2(
        ax=ax,
        cluster_stats=cluster_stats_2,
        var=var,
        color="tab:blue",
        offset=-0.6,
        pos_multiplier=3,
        width=0.75,
    )
    plot_color_boxplot_RMSE_2(
        ax=ax,
        cluster_stats=cluster_stats_1,
        var=var,
        color="tab:red",
        offset=0.6,
        pos_multiplier=3,
        width=0.75,
    )

axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Relative Humidity\nRMSE [%]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")


for ax in axs.flatten():
    ax.set_xticks([0, 3, 6, 9, 12], [1, 2, 4, 5, 6])
    ax.set_xlabel("Cluster")
    ax.set_xlim(-2, 14)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-ymax * 0.05, ymax * 1.05)


setup_1_handle = Patch(color="tab:red", label="MYNN SIP")
setup_2_handle = Patch(color="tab:blue", label="MYNN CONTROL")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1.06, 1.25),
)
plt.savefig("./figures/final_plots/met_RMSE_box_plot_by_cluster_tight.png", bbox_inches="tight", dpi=250)


# %%
def get_clustered_results_arr(cluster_errors: dict[int, dict[str, list]]) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    for i in range(6):
        if len(cluster_errors[i]["T"]) == 0:
            continue
        result[i] = {}
        result[i]["T"] = np.concatenate(cluster_errors[i]["T"])
        result[i]["RH"] = np.concatenate(cluster_errors[i]["RH"])
        result[i]["WD"] = np.concatenate(cluster_errors[i]["WD"])
        result[i]["WS"] = np.concatenate(cluster_errors[i]["WS"])

    return result


cluster_errors_arr_1 = get_clustered_results_arr(cluster_errors_1)
cluster_errors_arr_2 = get_clustered_results_arr(cluster_errors_2)
cluster_errors_arr_3 = get_clustered_results_arr(cluster_errors_3)
# %%
# get the RMSE for all of the variables (overall and by clusters)
data_by_cluster = []
data_overall = []
setup = 1
for arr_dict in [cluster_errors_arr_1, cluster_errors_arr_2, cluster_errors_arr_3]:
    for clus in range(6):
        if not clus in arr_dict.keys():
            data_by_cluster.append(
                {
                    "RMSE_T": np.nan,
                    "RMSE_RH": np.nan,
                    "RMSE_WS": np.nan,
                    "RMSE_WD": np.nan,
                    "setup": setup,
                    "cluster": clus,
                }
            )
            continue
        RMSE_T = np.sqrt(np.nanmean(arr_dict[clus]["T"] ** 2))
        RMSE_RH = np.sqrt(np.nanmean(arr_dict[clus]["RH"] ** 2))
        RMSE_WS = np.sqrt(np.nanmean(arr_dict[clus]["WS"] ** 2))
        RMSE_WD = np.sqrt(np.nanmean(arr_dict[clus]["WD"] ** 2))
        data_by_cluster.append(
            {
                "RMSE_T": RMSE_T,
                "RMSE_RH": RMSE_RH,
                "RMSE_WS": RMSE_WS,
                "RMSE_WD": RMSE_WD,
                "setup": setup,
                "cluster": clus,
            }
        )

    RMSE_T = np.sqrt(np.nanmean(np.concatenate([arr_dict[i]["T"] for i in arr_dict.keys()]) ** 2))
    RMSE_RH = np.sqrt(np.nanmean(np.concatenate([arr_dict[i]["RH"] for i in arr_dict.keys()]) ** 2))
    RMSE_WS = np.sqrt(np.nanmean(np.concatenate([arr_dict[i]["WS"] for i in arr_dict.keys()]) ** 2))
    RMSE_WD = np.sqrt(np.nanmean(np.concatenate([arr_dict[i]["WD"] for i in arr_dict.keys()]) ** 2))
    data_overall.append(
        {
            "RMSE_T": RMSE_T,
            "RMSE_RH": RMSE_RH,
            "RMSE_WS": RMSE_WS,
            "RMSE_WD": RMSE_WD,
            "setup": setup,
        }
    )

    setup += 1

df_by_cluster = pd.DataFrame(data_by_cluster)
df_overall = pd.DataFrame(data_overall)
df_overall["cluster"] = "overall"
df_by_cluster["cluster"] = df_by_cluster.cluster.apply(lambda x: str(x + 1))
df_all = pd.concat([df_by_cluster, df_overall])
melted = pd.melt(df_all, id_vars=["setup", "cluster"])
var_name_map = {
    "RMSE_T": "Temperature [K]",
    "RMSE_RH": "Relative Humidity [%]",
    "RMSE_WS": "Wind Speed [m/s]",
    "RMSE_WD": "Wind Dir. [deg]",
}
setup_name_map = {1: "WRF-SIP-MYNN", 2: "WRF-CONTROL-MYNN", 3: "WRF-SIP-YSU"}
melted["setup"] = melted.setup.apply(lambda x: setup_name_map[x])
melted["variable"] = melted.variable.apply(lambda x: var_name_map[x])
melted["value"] = melted.value.round(2)


formed = melted.set_index(["variable", "setup"]).pivot(columns="cluster", values="value")
formed.to_csv("./figures/final_plots/met_variables_RMSE_cluster_and_overal.csv", float_format="%.2f")

# Get the number of periods for each cluster
number_data = []
for df, setup_name in zip(
    [cluster_stats_2, cluster_stats_1, cluster_stats_3], ["WRF-CONTROL-MYNN", "WRF-SIP-MYNN", "WRF-SIP-YSU"]
):
    new_row = {"setup": setup_name}
    for i in range(6):
        mask = df.label == i
        new_row[f"{i + 1}"] = len(df.loc[mask].dropna(subset=["RMSE_T", "RMSE_RH", "RMSE_WD", "RMSE_WS"]))

    number_data.append(new_row)
last_row = {"setup": "Overall"}
for i in range(6):
    mask = df.label == i
    last_row[f"{i + 1}"] = len(cluster_stats_1.loc[mask])

number_data.append(last_row)

amounts = pd.DataFrame(number_data)
amounts.set_index("setup", inplace=True)
amounts["Total"] = amounts.apply(lambda x: np.sum(x), axis=1)
amounts.to_csv("./figures/final_plots/amount_of_clusters_modelled.csv")


# %%


# Make box plots for each cluster and setup
def plot_color_boxplot(
    ax,
    cluster_errors_arr: dict[int, dict[str, np.ndarray]],
    var: str,
    color: str,
    offset: float,
    pos_multiplier: float = 3,
    width: float = 0.25,
    absol=False,
):
    keys = cluster_errors_arr.keys()
    positions = [x * pos_multiplier + offset for x in keys]
    if absol:
        data = [np.abs(cluster_errors_arr[i][var][~np.isnan(cluster_errors_arr[i][var])]) for i in keys]
    else:
        data = [cluster_errors_arr[i][var][~np.isnan(cluster_errors_arr[i][var])] for i in keys]
    bp = ax.boxplot(data, positions=positions, widths=[width for x in range(len(keys))], whis=999)
    # bp = ax.violinplot(data, positions = positions, widths=[width for x in range(len(keys))])
    for item in bp.keys():
        plt.setp(bp[item], color=color, linewidth=1.75)


fig, axs = plt.subplots(2, 2, figsize=(10, 6))

for ax, var in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], ["T", "RH", "WS", "WD"]):
    absol = False
    if var == "WD":
        absol = True
    plot_color_boxplot(
        ax=ax,
        cluster_errors_arr=cluster_errors_arr_2,
        var=var,
        color="tab:blue",
        offset=-0.9,
        pos_multiplier=3,
        width=0.65,
        absol=absol,
    )
    plot_color_boxplot(
        ax=ax,
        cluster_errors_arr=cluster_errors_arr_1,
        var=var,
        color="tab:red",
        offset=0,
        pos_multiplier=3,
        width=0.65,
        absol=absol,
    )
    plot_color_boxplot(
        ax=ax,
        cluster_errors_arr=cluster_errors_arr_3,
        var=var,
        color="tab:orange",
        offset=0.9,
        pos_multiplier=3,
        width=0.65,
        absol=absol,
    )
    if not absol:
        ymin, ymax = ax.get_ylim()
        new_ylim = np.max([np.abs(ymin), np.abs(ymax)])
        ax.set_ylim(-new_ylim, new_ylim)
        ax.axhline(0, -1, 7, alpha=0.4, linestyle="--", color="k")

axs[0, 0].set_ylabel("Temperature Error [K]")
axs[0, 1].set_ylabel("Relative Humidity\nError [%]")
axs[1, 0].set_ylabel("Wind Speed Error [m/s]")
axs[1, 1].set_ylabel("Abs. Wind Direction\nError [deg]")


for ax in axs.flatten():
    ax.set_xticks([0, 3, 6, 9, 12, 15], [1, 2, 3, 4, 5, 6])
    ax.set_xlabel("Cluster")
    ax.set_xlim(-2, 17)


setup_1_handle = Patch(color="tab:red", label="WRF-SIP-MYNN")
setup_2_handle = Patch(color="tab:blue", label="WRF-CONTROL-MYNN")
setup_3_handle = Patch(color="tab:orange", label="WRF-SIP-YSU")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle, setup_3_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1, 1.2),
)

plt.savefig("./figures/final_plots/met_errors_box_plot_by_cluster.png", bbox_inches="tight", dpi=250)

# %%
# Overall Box plot
overall_data = []
wrf_metadata = pd.read_csv("./data/metadata.csv", parse_dates=["start_time", "end_time", "true_start_time"])
for setup in setups:
    mask = (
        (wrf_metadata.mp == setup["mp"])
        & (wrf_metadata.sip == setup["sip"])
        & (wrf_metadata.bl == setup["bl"])
        & (wrf_metadata.station == "HAC")
    )
    for _, row in wrf_metadata.loc[mask].iterrows():
        ncfile = WRFDataset(Path(row.file_path))
        mask_meteo = (meteo.time >= ncfile.times.iloc[0]) & (meteo.time <= ncfile.times.iloc[-1])
        # 10 hours as 5 minute increments
        required_points = 12 * 60 / 5
        if len(meteo[mask_meteo].dropna()) < required_points * 0.75:
            continue
        diff_results = get_diff_values_HAC(meteo, ncfile)
        overall_data.append(
            {
                "setup": setup["name"],
                "RMSE_T": np.sqrt(np.mean(diff_results["T"] ** 2)),
                "RMSE_RH": np.sqrt(np.mean(diff_results["RH"] ** 2)),
                "RMSE_WS": np.sqrt(np.mean(diff_results["WS"] ** 2)),
                "RMSE_WD": np.sqrt(np.mean(diff_results["WD"] ** 2)),
            }
        )
overall_RMSE = pd.DataFrame(overall_data)

# %%
fig, axs = plt.subplots(2, 2, figsize=(7.5, 5))
pos = 0
for setup_name, color in zip(
    ["WRF-CONTROL-MYNN", "WRF-SIP-MYNN", "WRF-SIP-YSU"],
    ["tab:blue", "tab:red", "tab:orange"],
):
    mask = overall_RMSE.setup == setup_name
    for var, ax in zip(
        ["RMSE_T", "RMSE_RH", "RMSE_WS", "RMSE_WD"], [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    ):

        if var == "RMSE_WD":
            data = np.abs(overall_RMSE.loc[mask, var])
        else:
            data = overall_RMSE.loc[mask, var]

        bp = ax.boxplot(data, positions=[pos], widths=[0.5], whis=999)
        # bp = ax.violinplot(data, positions = positions, widths=[width for x in range(len(keys))])
        for item in bp.keys():
            plt.setp(bp[item], color=color, linewidth=1.75)

    pos += 1
axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Relative Humidity\nRMSE [%]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")


for ax in axs.flatten():
    ax.set_xticklabels([])
    ax.set_xlabel("WRF Setup")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-ymax * 0.05, ymax)

setup_1_handle = Patch(color="tab:red", label="WRF-SIP-MYNN")
setup_2_handle = Patch(color="tab:blue", label="WRF-CONTROL-MYNN")
setup_3_handle = Patch(color="tab:orange", label="WRF-SIP-YSU")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle, setup_3_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1.1, 1.2),
)
plt.savefig("./figures/final_plots/met_RMSE_boxplot_overall.png", bbox_inches="tight", dpi=250)

# %%
# get overall run hours
data = []
overall_total_run_hours = 0
for setup in setups:
    simulation_periods = wrf_dataset_factory.get_periods(
        "NPRK", mp_phys=setup["mp"], bl_phys=setup["bl"], sip=setup["sip"]
    )
    unique_run_hours = 0
    for start, end in simulation_periods:
        unique_run_hours += (end - start).total_seconds() / 3600
    mask = (
        (wrf_metadata.mp == setup["mp"])
        & (wrf_metadata.sip == setup["sip"])
        & (wrf_metadata.bl == setup["bl"])
        & (wrf_metadata.station == "HAC")
    )
    df_slice = wrf_metadata.loc[mask]
    total_run_hours = 0
    for _, row in df_slice.iterrows():
        total_run_hours += (row.end_time - row.start_time).total_seconds() / 3600 - 24  # subtract spinup
        overall_total_run_hours += (
            row.end_time - row.start_time
        ).total_seconds() / 3600 - 24  # subtract spinup
    data.append(
        {
            "Setup": setup["name"],
            "Unique Run Hours": int(np.ceil(unique_run_hours)),
            "Total Run Hours": int(np.ceil(total_run_hours)),
        }
    )

mask1 = (
    (wrf_metadata.mp == setups[0]["mp"])
    & (wrf_metadata.sip == setups[0]["sip"])
    & (wrf_metadata.bl == setups[0]["bl"])
    & (wrf_metadata.station == "HAC")
)
mask2 = (
    (wrf_metadata.mp == setups[1]["mp"])
    & (wrf_metadata.sip == setups[1]["sip"])
    & (wrf_metadata.bl == setups[1]["bl"])
    & (wrf_metadata.station == "HAC")
)
mask3 = (
    (wrf_metadata.mp == setups[2]["mp"])
    & (wrf_metadata.sip == setups[2]["sip"])
    & (wrf_metadata.bl == setups[2]["bl"])
    & (wrf_metadata.station == "HAC")
)
overall_runs_mask = mask1 | mask2 | mask3
df_slice = wrf_metadata.loc[overall_runs_mask].copy()
df_slice.reset_index(drop=True, inplace=True)
starts = []
ends = []
last_end = pd.Timestamp(0)
for i in range(len(df_slice)):
    row_start_time = df_slice.loc[i, "true_start_time"]
    row_end_time = df_slice.loc[i, "end_time"]

    if row_start_time > last_end + pd.Timedelta(5, "minutes"):  # type: ignore
        starts.append(row_start_time)
        if last_end != pd.Timestamp(0):
            ends.append(last_end)
        last_end = row_end_time

    else:
        if row_end_time > last_end:  # type: ignore
            last_end = row_end_time

ends.append(last_end)

overall_unique_run_hours = 0
for start, end in zip(starts, ends):
    overall_unique_run_hours += (end - start).total_seconds() / 3600

data.append(
    {
        "Setup": "Overall",
        "Unique Run Hours": int(np.ceil(overall_unique_run_hours)),
        "Total Run Hours": int(np.ceil(overall_total_run_hours)),
    }
)
overall_amounts = pd.DataFrame(data)
overall_amounts.set_index("Setup", inplace=True)
overall_amounts["Unique Run Hours"] = overall_amounts["Unique Run Hours"].apply(
    lambda x: f"{x} ({x/24:.2f} days)"
)
overall_amounts["Total Run Hours"] = overall_amounts["Total Run Hours"].apply(
    lambda x: f"{x} ({x/24:.2f} days)"
)
overall_amounts.to_csv("./figures/final_plots/amount_of_hours_modelled.csv")
# %%
# get statistics for by radiosondes

radiosondes_metadata = pd.read_csv("./data/metadata_radiosondes.csv", parse_dates=["start_time", "end_time"])
radiosondes_metadata["sip"] = radiosondes_metadata.sip.apply(lambda x: True if x == 1 else False)

radiosonde_data = []
for setup in setups:
    mask = (radiosondes_metadata.bl == setup["bl"]) & (radiosondes_metadata.sip == setup["sip"])

    setup_radiosondes = radiosondes_metadata.loc[mask]
    for _, row in setup_radiosondes.iterrows():
        df = pd.read_csv(row.file_path, parse_dates=["time"])
        RMSE_T = np.sqrt(np.mean((df["T"] - df["wrf_T"]) ** 2))
        RMSE_WD = np.sqrt(np.mean((df["wind_dir"] - df["wrf_wind_dir"]) ** 2))
        RMSE_WS = np.sqrt(np.mean((df["wind_speed"] - df["wrf_wind_speed"]) ** 2))

        dZ = np.diff(df.geopot_height) / 2
        water_vapour = df.SH * df.rho * 10**3  # g m-3
        wrf_water_vapour = df.wrf_SH * df.wrf_rho * 10**3  # g m-3

        water_vapour_centered = np.diff(water_vapour) / 2 + water_vapour.iloc[:-1]
        wrf_water_vapour_centered = np.diff(wrf_water_vapour) / 2 + wrf_water_vapour.iloc[:-1]

        WVC = water_vapour_centered * dZ
        wrf_WVC = wrf_water_vapour_centered * dZ
        WVP = np.sum(WVC)
        wrf_WVP = np.sum(wrf_WVC)

        WVP_bias = wrf_WVP - WVP

        radiosonde_data.append(
            {
                "setup": setup["name"],
                "RMSE_T": RMSE_T,
                "RMSE_WS": RMSE_WS,
                "RMSE_WD": RMSE_WD,
                "WVP": WVP_bias,
                "start_time": row.start_time,
                "end_time": row.end_time,
            }
        )

# Add the missing radiosondes
radiosonde_RMSE = pd.DataFrame(radiosonde_data)
for file in Path("./data/radiosondes").iterdir():
    if not file.is_file():
        continue
    df = pd.read_csv(file, parse_dates=["time"])
    if df.time.min().round("h") in set(radiosonde_RMSE.start_time.apply(lambda x: x.round("h"))):
        continue
    radiosonde_data.append(
        {
            "setup": "None",
            "RMSE_T": np.nan,
            "RMSE_WS": np.nan,
            "RMSE_WD": np.nan,
            "WVP": np.nan,
            "start_time": df.time.min(),
            "end_time": df.time.max(),
        }
    )

radiosonde_RMSE = pd.DataFrame(radiosonde_data)
radiosonde_RMSE.sort_values("start_time", inplace=True)
radiosonde_RMSE.reset_index(drop=True, inplace=True)
radiosonde_RMSE.to_csv("./figures/final_plots/radiosone_RMSE.csv", float_format="%.2f")


# %%
# Plot radiosondes overall
fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.5))
pos = 0
for setup_name, color in zip(
    ["WRF-CONTROL-MYNN", "WRF-SIP-MYNN", "WRF-SIP-YSU"],
    ["tab:blue", "tab:red", "tab:orange"],
):
    mask = radiosonde_RMSE.setup == setup_name
    for var, ax in zip(["RMSE_T", "WVP", "RMSE_WS", "RMSE_WD"], [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]):

        data = radiosonde_RMSE.loc[mask, var]

        bp = ax.boxplot(data, positions=[pos], widths=[0.5], whis=999)
        # bp = ax.violinplot(data, positions = positions, widths=[width for x in range(len(keys))])
        for item in bp.keys():
            plt.setp(bp[item], color=color, linewidth=1.75)

    pos += 1

axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Column Water Vapour\nError [g/m$^{2}$]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")


for ax in [axs[0, 0], axs[1, 0], axs[1, 1]]:
    ax.set_xticklabels([])
    ax.set_xlabel("WRF Setup")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-ymax * 0.05, ymax)

axs[0, 1].set_xticklabels([])
axs[0, 1].set_xlabel("WRF Setup")
setup_1_handle = Patch(color="tab:red", label="WRF-SIP-MYNN")
setup_2_handle = Patch(color="tab:blue", label="WRF-CONTROL-MYNN")
setup_3_handle = Patch(color="tab:orange", label="WRF-SIP-YSU")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle, setup_3_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1.075, 1.2),
)
plt.savefig("./figures/final_plots/met_RMSE_boxplot_radiosondes.png", bbox_inches="tight", dpi=250)

# %%
# Radiosonde clusters
radiosonde_RMSE["label"] = -1

for i, row in radiosonde_RMSE.iterrows():
    mask = (clusters.start_time <= row.start_time) & (clusters.end_time >= row.end_time)
    if len(clusters.loc[mask]) < 1:
        continue
    radiosonde_RMSE.loc[i, "label"] = clusters.loc[mask].iloc[0].label_standardized


pos_multiplier = 3
width = 0.65
fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))

for var, ax in zip(["RMSE_T", "WVP", "RMSE_WS", "RMSE_WD"], [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]):
    for setup_name, color, offset in zip(
        ["WRF-CONTROL-MYNN", "WRF-SIP-MYNN", "WRF-SIP-YSU"],
        ["tab:blue", "tab:red", "tab:orange"],
        [-0.9, 0, 0.9],
    ):
        mask = radiosonde_RMSE.setup == setup_name
        df_slice = radiosonde_RMSE.loc[mask]
        unique_clusters = df_slice.label.unique()
        positions = [x * pos_multiplier + offset for x in unique_clusters]
        data = [df_slice.loc[df_slice.label == i, var] for i in unique_clusters]
        for d, pos in zip(data, positions):
            if len(d) < 5:
                ax.scatter([pos for _ in range(len(d))], d, color=color)
                continue
            bp = ax.boxplot(d, positions=[pos], widths=[width], whis=(0, 100))
            for item in bp.keys():
                plt.setp(bp[item], color=color, linewidth=1.75)

        # bp = ax.boxplot(
        #     data, positions=positions, widths=[width for _ in range(len(unique_clusters))], whis=999
        # )
        # bp = ax.violinplot(data, positions = positions, widths=[width for x in range(len(keys))])

axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Column Water Vapour\nError [g/m$^{2}$]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")

for ax in axs.flatten():
    ax.set_xticks([0, 3, 6, 9, 12, 15], [1, 2, 3, 4, 5, 6])
    ax.set_xlabel("Cluster")
    ax.set_xlim(-2, 17)


setup_1_handle = Patch(color="tab:red", label="WRF-SIP-MYNN")
setup_2_handle = Patch(color="tab:blue", label="WRF-CONTROL-MYNN")
setup_3_handle = Patch(color="tab:orange", label="WRF-SIP-YSU")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle, setup_3_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1.1, 1.2),
)

plt.savefig("./figures/final_plots/radiosonde_RMSE_box_plot_by_cluster.png", bbox_inches="tight", dpi=250)
# %%
# Radiosonde clusters tight
radiosonde_RMSE["label"] = -1

for i, row in radiosonde_RMSE.iterrows():
    mask = (clusters.start_time <= row.start_time) & (clusters.end_time >= row.end_time)
    if len(clusters.loc[mask]) < 1:
        continue
    radiosonde_RMSE.loc[i, "label"] = clusters.loc[mask].iloc[0].label_standardized


pos_multiplier = 3
width = 0.75
fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))

for var, ax in zip(["RMSE_T", "WVP", "RMSE_WS", "RMSE_WD"], [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]):
    for setup_name, color, offset in zip(
        ["WRF-CONTROL-MYNN", "WRF-SIP-MYNN"],
        ["tab:blue", "tab:red"],
        [-0.6, 0.6],
    ):
        mask = radiosonde_RMSE.setup == setup_name
        df_slice = radiosonde_RMSE.loc[mask]
        unique_clusters = df_slice.label.unique()
        positions = []
        for x in unique_clusters:
            if x < 3:
                positions.append(x * pos_multiplier + offset)
            else:
                positions.append((x - 1) * pos_multiplier + offset)
        data = [df_slice.loc[df_slice.label == i, var] for i in unique_clusters]
        for d, pos in zip(data, positions):
            if len(d) < 5:
                ax.scatter([pos for _ in range(len(d))], d, color=color)
                continue
            bp = ax.boxplot(d, positions=[pos], widths=[width], whis=(0, 100))
            for item in bp.keys():
                plt.setp(bp[item], color=color, linewidth=1.75)

        # bp = ax.boxplot(
        #     data, positions=positions, widths=[width for _ in range(len(unique_clusters))], whis=999
        # )
        # bp = ax.violinplot(data, positions = positions, widths=[width for x in range(len(keys))])

axs[0, 0].set_ylabel("Temperature RMSE [K]")
axs[0, 1].set_ylabel("Column Water Vapour\nError [g/m$^{2}$]")
axs[1, 0].set_ylabel("Wind Speed RMSE [m/s]")
axs[1, 1].set_ylabel("Wind Direction\nRMSE [deg]")

for ax in axs.flatten():
    ax.set_xticks([0, 3, 6, 9, 12], [1, 2, 4, 5, 6])
    ax.set_xlabel("Cluster")
    ax.set_xlim(-2, 14)


setup_1_handle = Patch(color="tab:red", label="WRF-SIP-MYNN")
setup_2_handle = Patch(color="tab:blue", label="WRF-CONTROL-MYNN")


plt.tight_layout()
axs[0, 0].legend(
    handles=[setup_2_handle, setup_1_handle],
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(1.1, 1.2),
)

plt.savefig(
    "./figures/final_plots/radiosonde_RMSE_box_plot_by_cluster_tight.png", bbox_inches="tight", dpi=250
)

# %%
# Get numbers of radiosondes applicable

number_data_radiosondes = []
for setup_name in ["WRF-CONTROL-MYNN", "WRF-SIP-MYNN", "WRF-SIP-YSU"]:
    new_row = {"setup": setup_name}
    for i in range(6):
        mask = (radiosonde_RMSE.label == i) & (radiosonde_RMSE.setup == setup_name)
        new_row[f"{i + 1}"] = len(
            radiosonde_RMSE.loc[mask].dropna(subset=["RMSE_T", "WVP", "RMSE_WD", "RMSE_WS"])
        )

    number_data_radiosondes.append(new_row)
last_row = {"setup": "Overall"}
for i in range(6):
    mask = radiosonde_RMSE.label == i
    last_row[f"{i + 1}"] = len(radiosonde_RMSE.loc[mask])

number_data_radiosondes.append(last_row)

amounts_radiosondes = pd.DataFrame(number_data_radiosondes)
amounts_radiosondes.set_index("setup", inplace=True)
amounts_radiosondes["Total"] = amounts_radiosondes.apply(lambda x: np.sum(x), axis=1)
amounts_radiosondes.to_csv("./figures/final_plots/amount_of_clusters_with_radiosondes.csv")
