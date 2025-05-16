# %%
import sys
from calendar import monthrange
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.append("./")
from WRFMultiDataset import MIRAMultiDatasetFactory, WRFMultiDatasetFactory

# %%


def label_weekday(ax, i, j):
    x_offset_rate = 1
    for weekday in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]:
        cl = "k"
        if weekday == "Sa" or weekday == "Su":
            cl = "blue"
        ax.text(i + 0.5, j + 0.5, weekday, ha="center", va="center", color=cl, fontsize=12)
        i += x_offset_rate


def get_time_col(anchor: pd.Timestamp, day: pd.Timestamp):
    seconds = (day - anchor).total_seconds()
    col = seconds / (60 * 60 * 24)
    return col


def get_time_row_col(time: pd.Timestamp):
    if time < pd.Timestamp(year=2024, month=10, day=14):
        row = 6
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=9, day=30), day=time)
    elif time < pd.Timestamp(year=2024, month=10, day=28):
        row = 5
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=10, day=14), day=time)
    elif time < pd.Timestamp(year=2024, month=11, day=11):
        row = 4
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=10, day=28), day=time)
    elif time < pd.Timestamp(year=2024, month=11, day=25):
        row = 3
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=11, day=11), day=time)
    elif time < pd.Timestamp(year=2024, month=12, day=9):
        row = 2
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=11, day=25), day=time)
    elif time < pd.Timestamp(year=2024, month=12, day=23):
        row = 1
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=12, day=9), day=time)
    else:
        row = 0
        col = get_time_col(anchor=pd.Timestamp(year=2024, month=12, day=23), day=time)

    return row, col


def make_base_calander(title: str = "title here"):
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    top = 7
    left = 0
    ax.axis([0, left + 14, 0, top])
    ax.set_xticks(list(range(left, left + 15)))

    label_weekday(ax, left, top - 0.25)
    label_weekday(ax, left + 7, top - 0.25)

    start_day = pd.Timestamp(year=2024, month=10, day=12)
    end_day = pd.Timestamp(year=2024, month=12, day=26)

    start_month = start_day.month
    end_month = end_day.month

    second_week = 0
    month_alt = 0
    row = top - 1
    # grey out previous days
    weekday, _ = monthrange(2024, start_month)
    for i in range(weekday):
        ax.add_patch(Rectangle((left + i, row), 1, 1, color="grey", alpha=0.25))
    for i in range(start_month, end_month + 1):
        weekday, num_days = monthrange(2024, i)
        col = weekday + left
        for day_number in range(1, num_days + 1):
            if col >= left + 14:
                row -= 1
                col = left
            if (
                pd.Timestamp(year=2024, month=i, day=day_number) < start_day
                or pd.Timestamp(year=2024, month=i, day=day_number) > end_day
            ):
                ax.text(col + 0.1, row + 0.9, f"{day_number}", ha="left", va="top", color="dimgrey")
                ax.add_patch(Rectangle((col, row), 1, 1, color="grey", alpha=0.4))
            else:
                ax.text(col + 0.1, row + 0.9, f"{day_number}", ha="left", va="top", color="k")
            if month_alt == 1:
                ax.add_patch(Rectangle((col, row), 1, 1, color="cornflowerblue", alpha=0.25))
            col += 1
        if month_alt == 0:
            month_alt = 1
        else:
            month_alt = 0
    # grey out last days
    weekday, _ = monthrange(2025, 1)
    for i in range(weekday, 7):
        ax.add_patch(Rectangle((left + i + 7, row), 1, 1, color="grey", alpha=0.25))

    ax.axis("on")
    ax.grid(True, color="k")
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    ax.text(-0.1, 5, "October", ha="right", va="bottom", fontsize=14, rotation=90)
    ax.text(-0.1, 3.5, "November", ha="right", va="center", fontsize=14, rotation=90, color="royalblue")
    ax.text(-0.1, 0, "December", ha="right", va="bottom", fontsize=14, rotation=90)

    ax.text(7, top + 0.75, title, ha="center", va="bottom", fontsize=16)
    return ax


# %%
start_time = pd.Timestamp(year=2024, month=10, day=12)
end_time = pd.Timestamp(year=2024, month=12, day=26)
MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
MIRA = MIRA_factory.get_dataset(start_time, end_time)
time_deltas = MIRA.times.diff().apply(lambda x: x.total_seconds()).to_numpy(dtype=np.float32)
times = MIRA.times.reset_index(drop=True)
# plt.boxplot(time_deltas[np.where((time_deltas>60)&(time_deltas<30*60))])

max_jump = 5 * 60

starts = [times[0]]
stops = []
res = []
in_low_res = False
time_past = 0
for i in range(1, len(time_deltas)):
    delta_seconds = time_deltas[i]
    if delta_seconds > 3600:
        stops.append(times[i - 1])
        starts.append(times[i])
        if in_low_res:
            res.append("low")
        else:
            res.append("high")
        time_past = 0
        continue

    if delta_seconds > max_jump and not in_low_res:
        in_low_res = True
        stops.append(times[i - 1])
        res.append("high")
        starts.append(times[i - 1])
        low_res_start = i
    if in_low_res:
        time_past += delta_seconds
        if delta_seconds > max_jump:
            time_past = 0
        if time_past > 3 * 60 * 60:
            in_low_res = False
            stops.append(times[i - 1])
            starts.append(times[i - 1])
            res.append("low")


if in_low_res:
    res.append("low")
else:
    res.append("high")
stops.append(MIRA.times.iloc[-1])


# manual cleaning
MIRA_plotting_df = pd.DataFrame({"start": starts, "end": stops, "res": res})
MIRA_plotting_df.loc[0, "end"] = MIRA_plotting_df.loc[6, "end"]
MIRA_plotting_df.drop([1, 2, 3, 4, 5, 6], inplace=True)
MIRA_plotting_df.loc[7, "end"] = pd.Timestamp(year=2024, month=10, day=16, hour=9)
MIRA_plotting_df.drop([8, 9], inplace=True)
MIRA_plotting_df.loc[10, "end"] = MIRA_plotting_df.loc[12, "end"]
MIRA_plotting_df.loc[10, "res"] = "low"
MIRA_plotting_df.drop([11, 12], inplace=True)
MIRA_plotting_df.drop([13, 14], inplace=True)
MIRA_plotting_df.loc[15, "end"] = pd.Timestamp(year=2024, month=11, day=12, hour=10, minute=15)
MIRA_plotting_df.loc[15, "res"] = "low"
MIRA_plotting_df.loc[16, "start"] = pd.Timestamp(year=2024, month=11, day=12, hour=10, minute=15)
MIRA_plotting_df.loc[16, "end"] = pd.Timestamp(year=2024, month=12, day=26)
MIRA_plotting_df.loc[16, "res"] = "high"
MIRA_plotting_df.drop([17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)


# # # %%
# # start_month = 12
# # start_day = 3
# # start_hour = 12
# # end_month=12
# # end_day = 4
# # end_hour = 3
# # plt.pcolormesh(MIRA.times[(MIRA.times>pd.Timestamp(year=2024, month=start_month, day=start_day, hour=start_hour)) & (MIRA.times<pd.Timestamp(year=2024, month=end_month, day=end_day, hour=end_hour))], MIRA.range, MIRA.refl[(MIRA.times>pd.Timestamp(year=2024, month=start_month, day=start_day, hour=start_hour)) & (MIRA.times<pd.Timestamp(year=2024, month=end_month, day=end_day, hour=end_hour))].T)
# # time_deltas[(MIRA.times>pd.Timestamp(year=2024, month=start_month, day=start_day, hour=start_hour)) & (MIRA.times<pd.Timestamp(year=2024, month=end_month, day=end_day, hour=end_hour))]
# %%
ax = make_base_calander("Radar Data Availability")


for _, row in MIRA_plotting_df.iterrows():
    start_row, start_col = get_time_row_col(row.start)
    end_row, end_col = get_time_row_col(row.end)
    if row.res == "low":
        cl = "tab:red"
    else:
        cl = "tab:blue"
    if start_row == end_row:
        ax.add_patch(Rectangle((start_col, start_row + 0.2), end_col - start_col, 0.35, color=cl))
        continue

    for j in np.linspace(start_row, end_row, (int(start_row) - int(end_row) + 1)):
        if start_row == j:
            ax.add_patch(Rectangle((start_col, start_row + 0.2), 15 - start_col, 0.35, color=cl))
            continue

        if j == end_row and j != 0:
            ax.add_patch(Rectangle((0, end_row + 0.2), end_col, 0.35, color=cl))
            continue

        if j == 0:
            ax.add_patch(Rectangle((0, j + 0.2), 4, 0.35, color=cl))
            continue

        ax.add_patch(Rectangle((0, j + 0.2), 14, 0.35, color=cl))

patch1 = Rectangle((0, 0), 1, 1, color="tab:red")
patch2 = Rectangle((0, 0), 1, 1, color="tab:blue")

ax.legend(
    [patch2, patch1],
    ["Regular Data", "Coarse Data"],
    ncols=2,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.13),
    fontsize=13,
)

plt.savefig("./figures/calanders/radar_calander.png", bbox_inches="tight", dpi=300)

# %%
# Make cluster calander
plt.close()

ax = make_base_calander("Clusters")

clusters = pd.read_csv("./figures/cluster_analysis_21/data.csv", parse_dates=["start_time", "end_time"])
mapping_21 = {5: 0, 2: 1, 4: 2, 0: 3, 1: 4, 3: 5}
label_mapping = mapping_21
clusters["label_standardized"] = -1
for key, val in label_mapping.items():
    clusters.loc[clusters.label == key, "label_standardized"] = val

clusters.sort_values("start_time", inplace=True)
clusters.reset_index(inplace=True, drop=True)
clusters.drop("Unnamed: 0", axis=1, inplace=True)

cmap = plt.get_cmap("turbo", 8)
colors = [cmap(i) for i in range(6)]

height = 0.35
offset = 0.2
for _, row in clusters.iterrows():
    start_row, start_col = get_time_row_col(row.start_time)
    end_row, end_col = get_time_row_col(row.end_time)
    if start_row == end_row:
        ax.add_patch(
            Rectangle(
                (start_col, start_row + 0.2), end_col - start_col, 0.35, color=colors[row.label_standardized]
            )
        )
        continue

    for j in np.linspace(start_row, end_row, (int(start_row) - int(end_row) + 1)):
        if start_row == j:
            ax.add_patch(
                Rectangle(
                    (start_col, start_row + 0.2), 15 - start_col, 0.35, color=colors[row.label_standardized]
                )
            )
            continue

        if j == end_row and j != 0:
            ax.add_patch(Rectangle((0, end_row + 0.2), end_col, 0.35, color=colors[row.label_standardized]))
            continue

        if j == 0:
            ax.add_patch(Rectangle((0, j + 0.2), 4, 0.35, color=colors[row.label_standardized]))
            continue

        ax.add_patch(Rectangle((0, j + 0.2), 14, 0.35, color=colors[row.label_standardized]))


ax.legend(
    [Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(6)],
    [f"{i}" for i in range(1, 7)],
    ncols=6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.2),
    fontsize=13,
    title="Cluster ID",
    title_fontsize=13,
)
plt.savefig("./figures/calanders/cluster_calander.png", bbox_inches="tight", dpi=300)
# %%
# n = 8
# cmap = plt.get_cmap("turbo", n)
# for i in range(n):
#     if i <= 6:
#         plt.scatter(i, i, color=cmap(i))

# %%
# make Radiosonde Calander
plt.close()

# radiosondes = pd.read_csv("./data/metadata_radiosondes.csv", parse_dates=["start_time", "end_time"])
radiosondes = []
for file in Path("./data/radiosondes").iterdir():
    if not file.is_file():
        continue
    df = pd.read_csv(file, parse_dates=["time"])
    radiosondes.append(df.time.min())
ax = make_base_calander("Radiosonde Launches")

launch_rows = []
launch_cols = []
for time in radiosondes:

    launch_row, launch_col = get_time_row_col(time)
    if launch_col % 1 > 0.5:
        launch_col += 0.1
    if launch_col % 1 < 0.5:
        launch_col -= 0.1
    launch_rows.append(launch_row + 0.375)
    launch_cols.append(launch_col)

plt.scatter(launch_cols, launch_rows, s=40, color="tab:red", label="Radiosonde Launch")

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), fontsize=13)
plt.savefig("./figures/calanders/radiosonde_calander.png", bbox_inches="tight", dpi=300)

# %%
# WRF data calander
plt.close()

wrf_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))

SIP_YSU = wrf_factory.get_periods("NPRK", 10, 1, True)
CONTROL_MYNN = wrf_factory.get_periods("NPRK", 10, 5, False)
SIP_MYNN = wrf_factory.get_periods("NPRK", 10, 5, True)

ax = make_base_calander("WRF Simulations")
for dates, offset_y, cl in zip(
    [CONTROL_MYNN, SIP_YSU, SIP_MYNN], [0.275, 0.1, 0.45], ["tab:blue", "tab:orange", "tab:red"]
):
    for start, end in dates:
        start_row, start_col = get_time_row_col(start)
        end_row, end_col = get_time_row_col(end)
        height = 0.1
        print(start_col, start_row)
        print(end_col, end_row)
        print()
        if start_row == end_row:
            ax.add_patch(Rectangle((start_col, start_row + offset_y), end_col - start_col, height, color=cl))
            continue

        for j in np.linspace(start_row, end_row, (int(start_row) - int(end_row) + 1)):
            if start_row == j:
                ax.add_patch(Rectangle((start_col, start_row + offset_y), 15 - start_col, height, color=cl))
                continue

            # if j == end_row and j != 0:
            if j == end_row:
                ax.add_patch(Rectangle((0, end_row + offset_y), end_col, height, color=cl))
                continue

            # if j == 0:
            #     ax.add_patch(Rectangle((0, j + offset_y), 3, height, color=cl))
            #     continue

            ax.add_patch(Rectangle((0, j + offset_y), 14, height, color=cl))


patch1 = Line2D([0], [0], color="tab:red", lw=5)
patch2 = Line2D([0], [0], color="tab:blue", lw=5)
patch3 = Line2D([0], [0], color="tab:orange", lw=5)

ax.legend(
    [patch3, patch2, patch1],
    ["WRF-SIP-YSU", "WRF-CONTROL-MYNN", "WRF-SIP-MYNN"],
    ncols=3,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.13),
    fontsize=13,
)

plt.savefig("./figures/calanders/wrf_calander.png", bbox_inches="tight", dpi=300)

# %%
# make Complete Data Calander
plt.close()

ax = make_base_calander("Data Availability")

mira_height = 0.15
mira_offset = 0.1
for _, row in MIRA_plotting_df.iterrows():
    start_row, start_col = get_time_row_col(row.start)
    end_row, end_col = get_time_row_col(row.end)
    if row.res == "low":
        cl = "darkgreen"
    else:
        cl = "lawngreen"
    if start_row == end_row:
        ax.add_patch(
            Rectangle((start_col, start_row + mira_offset), end_col - start_col, mira_height, color=cl)
        )
        continue

    for j in np.linspace(start_row, end_row, (int(start_row) - int(end_row) + 1)):
        if start_row == j:
            ax.add_patch(
                Rectangle((start_col, start_row + mira_offset), 15 - start_col, mira_height, color=cl)
            )
            continue

        if j == end_row and j != 0:
            ax.add_patch(Rectangle((0, end_row + mira_offset), end_col, mira_height, color=cl))
            continue

        if j == 0:
            ax.add_patch(Rectangle((0, j + mira_offset), 4, mira_height, color=cl))
            continue

        ax.add_patch(Rectangle((0, j + mira_offset), 14, mira_height, color=cl))


radiosondes = []
for file in Path("./data/radiosondes").iterdir():
    if not file.is_file():
        continue
    df = pd.read_csv(file, parse_dates=["time"])
    radiosondes.append(df.time.min())

launch_rows = []
launch_cols = []
for time in radiosondes:

    launch_row, launch_col_real = get_time_row_col(time)
    if launch_col_real % 1 >= 0.5:
        launch_col = launch_col_real - launch_col_real % 1 + 0.8
    if launch_col_real % 1 < 0.5:
        launch_col = launch_col_real - launch_col_real % 1 + 0.5
    launch_rows.append(launch_row + 0.8)
    launch_cols.append(launch_col)

plt.scatter(launch_cols, launch_rows, s=50, color="saddlebrown", label="Radiosonde Launch")

meteo = pd.read_csv("./data/insitu_measurements/HAC_meteo_20240121_all.csv", parse_dates=["time"])
meteo.reset_index(drop=True)
# meteo["parsed_time"] = pd.Timestamp(year=2024, month=10, day=1)
# for i in range(len(meteo)):
#     time_str = meteo.loc[i, "time"]
#     year = int(time_str[:4])
#     month = int(time_str[5:7])
#     day = int(time_str[8:10])
#     hour = time_str.split(":")[0].split(" ")[-1]
#     minute = time_str.split(":")[-1]
#     meteo.loc[i, "parsed_time"] = pd.Timestamp(
#         year=year, month=month, day=day, hour=int(hour), minute=int(minute)
#     )
# meteo.drop("time", axis=1, inplace=True)
# meteo.rename(columns={"parsed_time": "time"}, inplace=True)
meteo_time_delta = meteo.time.diff().apply(lambda x: x.total_seconds())
time_jumps = np.argwhere(meteo_time_delta > 600)
meteo_starts = [meteo.time.iloc[0]]
meteo_ends = []
for idx in time_jumps:
    meteo_ends.append(meteo.time.loc[idx[0] - 1])
    meteo_starts.append(meteo.time.loc[idx[0]])
meteo_ends.append(meteo.time.iloc[-1])
meteo_height = 0.15
meteo_offset = 0.35
for start, end in zip(meteo_starts, meteo_ends):
    start_row, start_col = get_time_row_col(start)
    end_row, end_col = get_time_row_col(end)
    cl = "tab:orange"
    if start_row == end_row:
        ax.add_patch(
            Rectangle((start_col, start_row + meteo_offset), end_col - start_col, meteo_height, color=cl)
        )
        continue

    for j in np.linspace(start_row, end_row, (int(start_row) - int(end_row) + 1)):
        if start_row == j:
            ax.add_patch(
                Rectangle((start_col, start_row + meteo_offset), 15 - start_col, meteo_height, color=cl)
            )
            continue

        if j == end_row and j != 0:
            ax.add_patch(Rectangle((0, end_row + meteo_offset), end_col, meteo_height, color=cl))
            continue

        # if j == 0:
        #     ax.add_patch(Rectangle((0, j + meteo_offset), 4, meteo_height, color=cl))
        #     continue

        ax.add_patch(Rectangle((0, j + meteo_offset), 14, meteo_height, color=cl))


ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), fontsize=13)
marker = Line2D(
    [],
    [],
    color="saddlebrown",
    marker="o",
    linestyle="None",
    markersize=7,
)
patch1 = Line2D([0], [0], color="forestgreen", linewidth=8)
patch2 = Line2D([0], [0], color="limegreen", linewidth=8)
patch3 = Line2D([0], [0], color="tab:orange", linewidth=8)

ax.legend(
    [patch3, patch2, patch1, marker],
    ["Meteo. Data", "Regular Radar Data", "Coarse Radar Data", "Radiosonde Launch"],
    ncols=4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.13),
    fontsize=13,
)
plt.savefig("./figures/calanders/full_data_calander.png", bbox_inches="tight", dpi=300)


# %%
