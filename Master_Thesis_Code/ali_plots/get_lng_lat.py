import numpy as np
from netCDF4 import Dataset
from wrf import ALL_TIMES, ll_to_xy, xy_to_ll

simuCONTROL = '/scratch/waseem/CHOPIN_oct8-10_allsip/wrfout_Helmos_d03_2024-10-08_00:00:00.nc'
ncfile1 = Dataset(simuCONTROL)

# ###HAC coordinates
# latitude = 37.9843   #46.55
# longitude = 22.1963

##VL coordinates
latitude = 37.9995
longitude = 22.19329

indxy1=ll_to_xy(ncfile1, latitude, longitude, timeidx=ALL_TIMES)
indx1=np.array(indxy1[0])
indy1=np.array(indxy1[1])

print("west_east",indx1,"south_north",indy1)