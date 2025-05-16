# %%
import datetime
import glob
import re
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

# %%

netCDF_path = "/scratch/waseem/CHOPIN_envelop_oct17-20/wrfout_CHPN_d03_2024-10-17_MIRACUT.nc"

netCDF = Dataset(netCDF_path)
times_arr = getvar(netCDF, "Times", meta=False, timeidx=None)

output_path = "/scratch/waseem/testing/crsim/Output504.nc"
output = Dataset(output_path)

comparison_path = "/scratch/waseem/CHOPIN_envelop_oct17-20/crsim/out/Output504.nc"
comparison = Dataset(comparison_path)
# %%

Zhh = output.variables["Zhh"]
lons = output.variables["xlong"][:]
lats = output.variables["xlat"][:]

plt.figure()
plt.pcolormesh(lats, lons, Zhh[-45, :,:].T, vmin=-35, vmax=20, cmap="jet")