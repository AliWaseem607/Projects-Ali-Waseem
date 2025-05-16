#%%

import numpy as np
from netCDF4 import Dataset

test = Dataset("test.nc", mode="a")
radar = np.load("data/radar/REFL_CHPN_2024-09-30_BASTA.npy")
radar_times = np.load("data/radar/REFL_times_CHPN_2024-09-30_BASTA.npy")

test_radar = np.load("/scratch/waseem/radar_testing/REFL_test.npy")
test_times = np.load("/scratch/waseem/radar_testing/REFL_times_test.npy")
test_2_radar = np.load("/scratch/waseem/radar_testing/REFL_test_2.npy")


# # %%
# Zhh = test.createVariable("Zhh_BASTA", "f4", ("Time", "bottom_top",), fill_value=-999.0)
# Zhh.units = "dBZ"
# test.variables
# Zhh[280:, :] = test_radar

