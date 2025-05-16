# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("./")
from scipy.signal import convolve
from WRFMultiDataset import MIRAMultiDataset, MIRAMultiDatasetFactory

# %%
MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
start_time = pd.Timestamp(year=2024, month=10, day=12)
end_time = pd.Timestamp(year=2024, month=12, day=31)
# MIRA = MIRA_factory.get_dataset(start_time=start_time, end_time=end_time)


# MIRA_test = MIRA_factory.get_dataset(start_time=pd.Timestamp(year=2024, month=11, day=10, hour=11), end_time=pd.Timestamp(year=2024, month=12, day=22, hour=0))
# %%
def get_cloud_statistics(
    MIRA_refl: np.ndarray, MIRA_range: np.ndarray, MIRA_times: pd.Series, MIRA_clean_mask: np.ndarray
) -> np.ndarray | None:
    """
    This function computes the statistics for a given time period of reflectivity
    values. It returns an nd array of n x 5 where n corresponds to the amount of times
    and the 5 columns are the seconds since 2024-10-01 of the reading, cloud base
    height (bottom most signal), cloud top height (top most signal), first cloud depth
    (height of uniterrupted cloud above the bottom most signal), and average cloud
    base reflectivity.
    """
    if MIRA_refl.shape[0] == 0:
        print("no Data")
        return None
    refl = MIRA_refl.copy()

    refl[MIRA_clean_mask == 1] = np.nan

    # Check to see at least 5 values
    number_of_values = np.sum(~np.isnan(refl), axis=1)
    refl[number_of_values < 5, :] = np.nan

    # get base and top heights
    cloud_check = np.argwhere(~np.isnan(refl))
    _, first_points = np.unique(cloud_check[:, 0], return_index=True)
    _, last_points_reversed = np.unique(cloud_check[::-1, 0], return_index=True)

    last_points = np.abs(last_points_reversed - cloud_check.shape[0] + 1)
    base_time_idx = cloud_check[first_points, 0]
    base_range_idx = cloud_check[first_points, 1]
    base_height_values = MIRA_range[base_range_idx]
    # top_time_idx = cloud_check[last_points,0]
    top_range_idx = cloud_check[last_points, 1]
    top_height_values = MIRA_range[top_range_idx]

    # base_height_mean = np.mean(base_height_values)
    # base_height_std = np.std(base_height_values)

    # top_height_mean = np.mean(top_height_values)
    # top_height_std = np.std(top_height_values)

    cloud_check_nth = np.cumsum(~np.isnan(refl), axis=1, dtype="float")
    cloud_check_nth[np.isnan(refl)] = np.nan
    mask_remove_data = cloud_check_nth > 5
    cloud_check_nth[mask_remove_data] = np.nan
    refl_base_dBZ = refl.copy()
    refl_base_dBZ[np.isnan(cloud_check_nth)] = np.nan
    time_mask = np.any(~np.isnan(cloud_check_nth), axis=1)
    base_dBZ = np.nanmean(refl_base_dBZ[time_mask, :], axis=1)

    # clean
    del refl_base_dBZ
    del cloud_check_nth
    del time_mask
    # base_dBZ_mean = np.mean(base_dBZ)
    # base_dBZ_std = np.std(base_dBZ)

    # get cloud depth
    cloud_patches = refl[base_time_idx, :].copy()
    cloud_depth_mask = np.isnan(cloud_patches) & (
        np.tile(MIRA_range, (cloud_patches.shape[0], 1))
        > base_height_values.reshape(base_height_values.shape[0], 1)
    )
    del cloud_patches
    above_first_cloud = np.argwhere(cloud_depth_mask)
    _, cloud_depth_points = np.unique(above_first_cloud[:, 0], return_index=True)
    # cloud_depth_time_idx = above_first_cloud[cloud_depth_points, 0]
    cloud_depth_range_idx = above_first_cloud[cloud_depth_points, 1]
    cloud_depth_values = MIRA_range[cloud_depth_range_idx] - base_height_values

    clouds_times = MIRA_times.iloc[base_time_idx]
    base_time = pd.Timestamp(year=2024, month=10, day=1)
    cloud_times_epoch = [(x - base_time).total_seconds() for x in clouds_times]
    return np.stack(
        [cloud_times_epoch, base_height_values, top_height_values, cloud_depth_values, base_dBZ], axis=1
    )


# get_cloud_statistics(MIRA_test.refl, MIRA_range=MIRA_test.range, MIRA_times=MIRA_test.times)
# %%
# mask_range = (MIRA.range>8000) & (MIRA.range<9000)
# plt.pcolormesh(
#         MIRA_test.times,
#         MIRA_test.range,
#         MIRA_test.refl.T,
#         vmin=-60,
#         vmax=35,
#         cmap="jet",
#     )
# plt.plot(MIRA_test.times[base_time_idx], base_height_values)
# plt.plot(MIRA_test.times[top_time_idx], top_height_values)
# plt.plot(MIRA_test.times[base_time_idx], cloud_depth_values)
# #%%
# plt.pcolormesh(
#         MIRA_test.times[base_time_idx],
#         MIRA_test.range,
#         cloud_patches.T,
#         vmin=-60,
#         vmax=35,
#         cmap="jet",
#     )
# plt.plot(MIRA_test.times[base_time_idx], base_height_values)
# plt.plot(MIRA_test.times[top_time_idx], top_height_values)

# %%
splits = pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta(1, "d"))
data = []
for i in range(1, len(splits)):
    try:
        MIRA = MIRA_factory.get_dataset(start_time=splits[i - 1], end_time=splits[i])
    except:
        continue

    MIRA_refl = MIRA.refl
    MIRA_times = MIRA.times
    assert MIRA.clean_mask is not None
    MIRA_clean_mask = MIRA.clean_mask
    print(splits[i - 1], splits[i])
    split_data = get_cloud_statistics(
        MIRA_refl=MIRA_refl, MIRA_range=MIRA.range, MIRA_times=MIRA_times, MIRA_clean_mask=MIRA_clean_mask
    )
    if split_data is not None:
        data.append(split_data)

    ### Code to calculate cloud layers if need be
    ### we had to down sample though so save on space, so at some point we have to up
    ### sample again, should be relatively easy
    # refl = MIRA_refl.copy()
    # refl[MIRA_clean_mask] = np.nan
    # refl = refl[::10, :].copy()

    # filter_size = 25
    # avg_filter = np.ones((filter_size, filter_size)) / filter_size**2
    # padding = int((filter_size - 1) / 2)
    # pad_axis_0 = np.zeros((padding, refl.shape[1]))
    # pad_axis_0.fill(np.nan)
    # pad_axis_1 = np.zeros((int(refl.shape[0] + padding * 2), padding))
    # pad_axis_1.fill(np.nan)
    # refl_padded_0 = np.concatenate([pad_axis_0, refl, pad_axis_0], axis=0)
    # refl_padded = np.concatenate([pad_axis_1, refl_padded_0, pad_axis_1], axis=1)

    # refl_avg_matrix = np.zeros((refl.shape[0], refl.shape[1], filter_size**2), dtype=np.int8)
    # mid = int((filter_size**2 - 1) / 2)
    # refl_avg_matrix[:, :, mid] = refl
    # counter = 0
    # for i in range(filter_size):
    #     for j in range(filter_size):
    #         right_idx = refl.shape[0] + i
    #         left_idx = i
    #         top_idx = refl.shape[1] + j
    #         bottom_idx = j
    #         refl_avg_matrix[:, :, counter] = ~np.isnan(refl_padded[left_idx:right_idx, bottom_idx:top_idx])
    #         counter += 1
    # refl_blur = np.nansum(refl_avg_matrix, axis=2)
    # below_array = np.concatenate([np.zeros((refl_blur.shape[0], 1)), refl_blur[:, :-1]], axis=1)
    # is_cloud = refl_blur > 0
    # is_cloud_below = below_array > 0
    # is_bottom = is_cloud & (is_cloud != is_cloud_below)
    # cloud_layers = np.sum(is_bottom, axis=1)

data_arr = np.concatenate(data, axis=0)
filename = f"{start_time.strftime(r'%Y%m%d')}-{end_time.strftime(r'%Y%m%d')}_4_features.npy"


data_string = f"file: {filename}, written on {pd.Timestamp.today().strftime(r'%Y%m%dT%H:%M:%S')}, columns: time from 2024-10-01; cloud base height [m]; cloud top height [m]; lowest cloud depth [m]; average bottom reflectivity \n"

with open("/home/waseem/CHOPIN_analysis/data/MIRA_extracted_features/logs.txt", "a") as f:
    f.write(data_string)


np.save(f"/home/waseem/CHOPIN_analysis/data/MIRA_extracted_features/{filename}", data_arr)
