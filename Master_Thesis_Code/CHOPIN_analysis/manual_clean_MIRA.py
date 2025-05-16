# %%
import glob

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

trimmed_files = np.sort(glob.glob("/scratch/waseem/MIRA_insitu/trimmed_data/*.nc"))

def check_for_val(arr, left, right, bottom, top):
    check = ~np.isnan(arr)
    ax_0 = np.arange(arr.shape[0])
    ax_1 = np.arange(arr.shape[1])
    mask_0 = (ax_0<= left) | (ax_0>=right)
    mask_1 = (ax_1<=bottom) | (ax_1>=top)
    check[mask_0, :] = False
    check[:, mask_1] = False
    return np.argwhere(check)
#%%

file_idx = 61

ncfile = Dataset(trimmed_files[file_idx], mode="r")

elv = ncfile.variables["elv"][:]
mask_elv = elv < 85
refl = ncfile.variables["Z"][:].filled().copy()
refl[mask_elv] = np.nan
refl_dBZ = 10 * np.log10(refl)

clean_mask = ncfile.variables["clean_mask"][:].copy()

ranges = np.arange(refl.shape[1])
times = np.arange(refl.shape[0])
ncfile.close()


fixes = []
mask = clean_mask == 1
old_clean = refl_dBZ.copy()
old_clean[mask] = np.nan
working_refl = refl_dBZ.copy()
working_refl[mask] = np.nan
removed = 0
# vals = check_for_val(working_refl, 0, 35000, 50, 75)
# fixes.append(vals)
# working_refl[vals[:,0], vals[:,1]] = np.nan
# removed += vals.shape[0]

# vals = check_for_val(working_refl, 4000, 35000, 55, 75)
# fixes.append(vals)
# working_refl[vals[:,0], vals[:,1]] = np.nan
# removed += vals.shape[0]

# vals = check_for_val(working_refl, 0, 3750, 0, 500)
# fixes.append(vals)
# working_refl[vals[:,0], vals[:,1]] = np.nan
# removed += vals.shape[0]

# vals = check_for_val(working_refl, 6300, 60200, 0, 500)
# fixes.append(vals)
# working_refl[vals[:,0], vals[:,1]] = np.nan
# removed += vals.shape[0]

# vals = check_for_val(working_refl, 12500, 17000, 0, 500)
# fixes.append(vals)
# working_refl[vals[:,0], vals[:,1]] = np.nan
# removed += vals.shape[0]


# ax[0].pcolormesh(times[0:], ranges, old_clean[0:, :].T, vmin=-60, vmax=35,  cmap="jet")
plt.figure(figsize=(10,5))
plt.pcolormesh(times[0:], ranges[:], working_refl[0:, :].T, vmin=-60, vmax=35,  cmap="jet")
plt.grid()
# ax[1].axhline(58, 0, 16000)

#%%
update_points = np.concatenate(fixes, axis=0)
# plt.pcolormesh(times[0:], ranges, working_refl[0:, :].T, vmin=-60, vmax=35,  cmap="jet")
# plt.scatter(update_points[:,0], update_points[:,1])

#%%
clean_mask[update_points[:, 0], update_points[:, 1]] = 1
ncfile = Dataset(trimmed_files[file_idx], mode="a")
ncfile.variables["clean_mask"][:, :] = clean_mask
ncfile.close()

