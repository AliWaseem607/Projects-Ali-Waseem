# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar

# ncks -d west_east,145,161 -d south_north,159,175 -d west_east_stag,145,162 -d south_north_stag,159,176 -v RDX,RDY,XLAT,XLONG,U,V,W,PB,P,T,PHB,PH,QVAPOR,QCLOUD,QRAIN,QICE,QSNOW,QGRAUP,QNRAIN,QNICE,QNSNOW,Times
# NPRK is 8,8 location 

# ncdump -v RDX,RDY,XLAT,XLONG,U,V,W,PB,P,T,PHB,PH,QVAPOR,QCLOUD,QRAIN,QICE,QSNOW,QGRAUP,QNRAIN,QNICE,QNSNOW /scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d03_2024-10-27_18\:00\:00.nc

#%%

MIRA_radar = Dataset("/home/waseem/radar_data/MIRA_combined.znc")
MIRA_times = pd.to_datetime(MIRA_radar.variables["time"], unit="s")
elevation = np.squeeze(MIRA_radar.variables["elv"])

# wrf = Dataset("/home/waseem/CHOPIN_analysis/data/wrfout_CHPN_d03_2024-10-17_MIRACUT.nc")
# spinup = 24*60/5
# wrf_times = getvar(wrf, "Times", timeidx=None, meta=False)

