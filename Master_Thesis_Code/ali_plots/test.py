#%% Imports
from netCDF4 import Dataset
import numpy as np
import datetime
import matplotlib.pyplot as plt
from wrf import to_np
import matplotlib.colors as colors

#%% Load data
data_path = "/scratch/waseem/Helmos_control/wrfout_Helmos_d03_CONTROL_VL.nc"

CONTROL = Dataset(data_path)


# Set up the times
wrf_time = np.arange(np.datetime64('2021-12-17T00:00'), np.datetime64('2021-12-19T12:05'),dtype='datetime64[s]')[::30*60]
spin_up = np.where(wrf_time == np.datetime64('2021-12-17T22:00'))[0][0]
end = np.where(wrf_time == np.datetime64('2021-12-19T12:00'))[0][0] + 1
wrf_time=wrf_time[spin_up:end]

start_time = datetime.datetime(2021, 12, 17, 18, 0)
end_time = datetime.datetime(2021, 12, 19, 14, 0)

### WRF constants
RA=287.15
RD=287.0
CP=1004.5
P1000MB=100000.0
EPS=0.622
        
presCONTROL = np.squeeze(CONTROL.variables['P'][:] + CONTROL.variables['PB'][:])
thetCONTROL = np.squeeze(CONTROL.variables['T'][:] + 300.0)
qvCONTROL = np.squeeze(CONTROL.variables['QVAPOR'][:])
tkCONTROL = ((presCONTROL / P1000MB)**(RD/CP) * thetCONTROL) # look up potential temperture to regular temperature
tCONTROL = tkCONTROL - 273.15 
tvCONTROL = tkCONTROL * (EPS + qvCONTROL) / (EPS * (1. + qvCONTROL)) # look up virtual temperature to regular temperature
rhoCONTROL = presCONTROL/RA/tvCONTROL
icncCONTROL = np.squeeze((CONTROL.variables['QNICE'][:] + CONTROL.variables['QNSNOW'][:] + CONTROL.variables['QNGRAUPEL'][:]))*rhoCONTROL*10**-3 #L-1
iwcCONTROL = np.squeeze((CONTROL.variables['QICE'][:] + CONTROL.variables['QSNOW'][:] + CONTROL.variables['QGRAUP'][:]))*rhoCONTROL*10**3 #gm-3
lwcCONTROL = np.squeeze((CONTROL.variables['QCLOUD'][:] + CONTROL.variables['QRAIN'][:]))*rhoCONTROL*10**3 #gm-3
pblhCONTROL = np.squeeze(CONTROL.variables['PBLH'][:])/1000 #PBLH in km

PHB = np.squeeze(CONTROL.variables["PHB"][0,:])
PH = np.squeeze(CONTROL.variables["PH"][0,:])
HGT = np.squeeze(CONTROL.variables["HGT"][0])
ZZASL = (PH+PHB)/9.81
ZZ = (PH+PHB)/9.81-HGT
ZZ_km = ZZ/1000
dz=np.zeros((len(ZZ)-1))
ZZmiddle=np.zeros((len(ZZ)-1))

kk = 0

for jj in range(len(ZZ)-1):
        dz[kk] = (ZZ[kk+1]-ZZ[kk])/2
        ZZmiddle[kk] = dz[kk] + ZZ[kk]
        kk=kk+1

icncCONTROL[icncCONTROL <= 10**(-5)] = np.nan
lwcCONTROL[lwcCONTROL <= 10**(-6)] = np.nan
iwcCONTROL[iwcCONTROL <= 10**(-6)] = np.nan

print(icncCONTROL.shape)
print(icncCONTROL[spin_up:end])


icncCONTROL = icncCONTROL[spin_up:end]
lwcCONTROL = lwcCONTROL[spin_up:end]
tCONTROL = tCONTROL[spin_up:end]
iwcCONTROL = iwcCONTROL[spin_up:end]
#%%
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=14)

fig, axs = plt.subplots(4,1,figsize=(14,14))

levs = np.logspace(-2, 3, 10)

plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.95, hspace=0.20)
icnc = axs[0].contourf(wrf_time, ZZmiddle/1000, to_np(icncCONTROL.T), levs, norm=colors.LogNorm())
cs = axs[0].contour(wrf_time, ZZmiddle/1000, tCONTROL.T, levels=np.arange(-50, 0, 5), colors='dimgray',linewidths=1)
axs[0].clabel(cs, inline=True, fontsize=12, fmt='%d$^\circ$C', colors='dimgrey')
cbar = fig.colorbar(icnc,ax=axs[0],aspect=15)
cbar.ax.set_yscale('log')
cbar.set_label('ICNC CONTROL [$\mathrm{L^{-1}}$]', fontsize=16)
axs[0].set_xlim(wrf_time[0], wrf_time[-1])
axs[0].set_ylabel("Altitude [km]")
axs[0].set_ylim(0,5)
axs[0].set_xticklabels([])
br_handle = plt.Line2D([], [], color='darkviolet', linewidth=2)
hm_handle = plt.Line2D([], [], color='darkcyan', linewidth=2)
br_handle2 = plt.Line2D([], [], color='darkviolet', linestyle='--', linewidth=2)
sb_handle = plt.Line2D([], [], color='magenta', linewidth=2)
ds_handle = plt.Line2D([], [], color='cyan', linewidth=2)
agg_handle = plt.Line2D([], [], color='red', linewidth=2)
axs[0].legend(handles=[br_handle, br_handle2, sb_handle, ds_handle, agg_handle], labels=[br_label, br_label2, sb_label, ds_label, agg_label], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.35),frameon=False) #labelcolor='white', bbox_to_anchor=(1.11, 0.92)
text_loc = np.datetime64('2021-12-19T10:30:00')
text =axs[0].text(text_loc, 4.5, '(a)', fontsize=14)




# %%
print(type(icncCONTROL))
npicnc = to_np(icncCONTROL.T)
print(type(npicnc))
print(icncCONTROL)
print(npicnc)
# %%

print(CONTROL.variables.keys())
print(CONTROL.dimensions.keys())
# %%
