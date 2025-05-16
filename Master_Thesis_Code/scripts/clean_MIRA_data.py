#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from netCDF4 import Dataset  # type: ignore


#%%
def get_clean_mask(Z, mask_elv, min_points_range:int=2, min_points_time:int=3) -> np.ndarray:

    if np.all(~mask_elv):
        return np.zeros(Z.shape, dtype="int8")
    # assume that if we can't see up we still might have signal for the purpose of cleaning
    Z[~mask_elv] = 1
    dBZ = 10 * np.log10(Z)
    isval = ~np.isnan(dBZ)
    numb_times = dBZ.shape[0]
    numb_ranges = dBZ.shape[1]

    # Check that there are at least 2 consecutive range values
    number_of_checks = min_points_range-1
    check_range = np.zeros((numb_times, numb_ranges, 2 * number_of_checks + 1), dtype="bool")
    check_range[:, :, number_of_checks] = isval
    for n in range(1, number_of_checks +1):
        check_range[:, :, number_of_checks + n] = np.concatenate(
            [
                isval[:, n:],
                np.zeros((numb_times, n), dtype="bool"),
            ],
            axis=1,
        )
        check_range[:, :, number_of_checks - n] = np.concatenate(
            [np.zeros((numb_times, n), dtype="bool"), isval[:, :-n]], axis=1
        )

    consecutive_range_check = np.zeros((numb_times, numb_ranges), dtype="bool")
    for i in range(min_points_range):
        start_idx = 0 + i
        end_idx = min_points_range + i
        check_slice = check_range[:, :, start_idx:end_idx]
        consecutive_range_check = consecutive_range_check | (
            np.sum(check_slice, axis=2) >= min_points_range
        )

    # clean
    del check_range
    del number_of_checks

    # Check that there are at least 5 consecutive time values
    number_of_checks = min_points_time-1
    check_time = np.zeros((numb_times, numb_ranges, 2 * number_of_checks + 1))
    check_time[:, :, number_of_checks] = isval
    for n in range(1, number_of_checks + 1):
        check_time[:, :, number_of_checks + n] = np.concatenate(
            [isval[n:, :], np.zeros((n, numb_ranges), dtype="bool")], axis=0
        )
        check_time[:, :, number_of_checks - n] = np.concatenate(
            [np.zeros((n, numb_ranges), dtype="bool"), isval[:-n, :]], axis=0
        )

    consecutive_time_check = np.zeros((numb_times, numb_ranges), dtype="bool")
    for i in range(min_points_time):
        start_idx = 0 + i
        end_idx = min_points_time + i
        check_slice = check_time[:, :, start_idx:end_idx]
        consecutive_time_check = consecutive_time_check | (
            np.sum(check_slice, axis=2) >= min_points_time
        )

    # clean
    del check_time
    
    not_enough_points = isval & (~consecutive_range_check | ~consecutive_time_check)
    clean_mask_arr = np.zeros(Z.shape, dtype=np.int8)
    clean_mask_arr[not_enough_points] = 1
    # if np.all(mask_elv):
    #     clean_mask_arr[not_enough_points] = 1
    #     return clean_mask_arr
    
    # counter = 0
    # for i in range(clean_mask_arr.shape[0]):
    #     if not mask_elv[i]:
    #         continue
    #     clean_mask_arr[i, :][not_enough_points[counter,:]] = 1 
    #     counter+=1

    return clean_mask_arr


def main(MIRA_ncfile_path: Path | list[Path], reset_data:bool = False, clean_level:int = 0):
    if isinstance(MIRA_ncfile_path, Path):
        MIRA_ncfile_path = [MIRA_ncfile_path]

    for path in MIRA_ncfile_path:
        abs_path = path.absolute()
        print(f"Analyzing {abs_path}")
        ncfile = Dataset(str(abs_path), mode="r")
        
        if not reset_data and "clean_mask" in ncfile.variables.keys():
            print(f"clean_mask is already in {abs_path}")
            continue
            
        elv = ncfile.variables["elv"][:]
        mask_elv = elv > 85
        Z = ncfile.variables["Z"][:].copy()
        ncfile_times = ncfile.variables["time"][:].copy()
        ncfile.close()
        if clean_level == 0:
            clean_mask_arr = get_clean_mask(Z=Z, mask_elv=mask_elv, min_points_range=2, min_points_time=3)
        elif clean_level == 1:
            clean_mask_arr = get_clean_mask(Z=Z, mask_elv=mask_elv, min_points_range=4, min_points_time=30)
        else:
            raise NotImplementedError()

        ncfile = Dataset(str(abs_path), mode="a")

        if "clean_mask" in ncfile.variables.keys():
            clean_mask = ncfile.variables["clean_mask"]
        else:
            clean_mask = ncfile.createVariable("clean_mask", "i1", ("time", "range"), fill_value=0)
            clean_mask.units = "-"
            clean_mask.description = (
                "mask to filter radar data so that points without 2 adjacent "
                + "signals in range and 5 adjacent signals in time are removed to reduce "
                + "potential noise. 0 corresponds to points to keep and 1 coresponds to "
                + "points that should be replaced with np.nan. Note that if there is a "
                + "large gap in the time difference it is not removed (greater than "
                + "30 seconds)."
            )
    
        clean_mask[:, :] = clean_mask_arr
        ncfile.close()

if __name__ == "__main__":
    tyro.cli(main)
#%%
# fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# im0 = ax[0].pcolormesh(
#     ncfile.variables["time"][mask_elv],
#     ncfile.variables["range"][:] / 1000,
#     dBZ.T,
#     vmin=-60,
#     vmax=35,
#     cmap="jet",
# )
# cbar = plt.colorbar(im0)
# cbar.set_label("Reflectivity [dBZ]")
# ax[0].set_ylim(0, 12)
# ax[0].set_ylabel("Altitude [km]")
# ax[0].set_title("MIRA")

# clean_dBZ = dBZ.copy()
# clean_dBZ[not_enough_points] = np.nan

# im0 = ax[1].pcolormesh(
#     ncfile.variables["time"][mask_elv],
#     ncfile.variables["range"][:] / 1000,
#     clean_dBZ.T,
#     vmin=-60,
#     vmax=35,
#     cmap="jet",
# )
# cbar = plt.colorbar(im0)
# cbar.set_label("Reflectivity [dBZ]")
# ax[1].set_ylim(0, 12)
# ax[1].set_ylabel("Altitude [km]")
# ax[1].set_title(f"MIRA cleaned: {np.sum(not_enough_points)} points removed")
