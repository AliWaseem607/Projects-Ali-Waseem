# -*- coding: utf-8 -*-

from __future__ import print_function
from netCDF4 import Dataset
from wrf import getvar, interpline, CoordPair, xy_to_ll_proj, ll_to_xy_proj, to_np, xy_to_ll, ll_to_xy, ALL_TIMES
import numpy.testing as nt
import numpy.ma as ma
import sys
import subprocess
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import pandas as pd
import os
import warnings
import matplotlib.colors as colors
from matplotlib.cm import get_cmap
import csv
import typhon


pine_inps = []
demott_inps = []
pine_temp = []
dmt10_inps = []

with open(r"./Data/PINE/PINE_INP_vs_DeMott10.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        pine_inps.append(float(row[0]) if row[0] != '' else np.nan)
        demott_inps.append(float(row[1]) if row[1] != '' else np.nan)
        pine_temp.append(float(row[2]) if row[2] != '' else np.nan)

pine_temp = [float(x) for x in pine_temp]

###Meyers INPs
T0 = 273.15 ##in K
meyers_inps = [None] * len(pine_temp)

for i in range(len(pine_temp)):
    meyers_inps[i] = math.exp(-2.8 + 0.262*(T0 - (pine_temp[i] + 273.15)))

cooper_inps = [None] * len(pine_temp)

for i in range(len(pine_temp)):
    cooper_inps[i] = 0.005*math.exp(0.304 * (T0 - (pine_temp[i] + 273.15)))
    cooper_inps[i] = min(cooper_inps[i],500.e3)
   
dmt10_inps = [None] * len(pine_temp)
demott_a = 5.94e-5
demott_b = 3.33 
demott_c = 0.0264
demott_d = 0.0033
demott_num = 0.30

for i in range(len(pine_temp)):
    dmt10_inps[i] = demott_a*((273.15-(pine_temp[i] + 273.15))**(demott_b))*(demott_num**((demott_c*(273.15-(pine_temp[i] + 273.15)))+demott_d)) # NUM OF ICE CRYSTALS L-1
    dmt10_inps[i] = min(dmt10_inps[i],500.e3)

kfgao_inps = [None] * len(pine_temp)   
kfgao_inps2 = [None] * len(pine_temp)
a_kf = 0.989841; a_kf2 = 1.08057
b_kf = 61.7058; b_kf2 = 115.2183
c_kf = 0.180653; c_kf2 = 0.3294061
d_kf = 9.06809; d_kf2 = 13.41891
e_kf = 0.731812; e_kf2 = 0.7348044
APS_0_5 = 0.30781*1000 #stdL-1 0.30781
Ratio = 20.2560

for i in range(len(pine_temp)):
    kfgao_inps[i] = (APS_0_5**a_kf + b_kf) * math.exp( - c_kf*(pine_temp[i]) - d_kf) * (1/(Ratio**e_kf)) # NUM OF ICE CRYSTALS L-1
    kfgao_inps[i] = min(kfgao_inps[i],500)

for i in range(len(pine_temp)):
    kfgao_inps2[i] = (APS_0_5**a_kf2 + b_kf2) * math.exp( - c_kf2*(pine_temp[i]) - d_kf2) * (1/(Ratio**e_kf2)) # NUM OF ICE CRYSTALS L-1
    kfgao_inps2[i] = min(kfgao_inps2[i],500)
   

###################################
# WRF Outputs - for isotherms     #
###################################
out_dir ='./Data/WRF'

simuALLSIP02 = out_dir + '/wrfout_d03_ALLSIP_VL.nc'
ALLSIP02 = Dataset(simuALLSIP02)

wrf_time = np.arange(np.datetime64('2021-12-17T00:00'), np.datetime64('2021-12-19T12:05'),dtype='datetime64[s]')[::300]
spin_up = np.where(wrf_time == np.datetime64('2021-12-17T22:00'))[0][0]
end = np.where(wrf_time == np.datetime64('2021-12-19T12:00'))[0][0] + 1
wrf_time=wrf_time[spin_up:end]

### WRF constants
RA=287.15
RD=287.0
CP=1004.5
P1000MB=100000.0
EPS=0.622

presALLSIP = np.squeeze(ALLSIP02.variables['P'][:] + ALLSIP02.variables['PB'][:])
thetALLSIP = np.squeeze(ALLSIP02.variables['T'][:] + 300.0)
qvALLSIP = np.squeeze(ALLSIP02.variables['QVAPOR'][:])
tkALLSIP = ((presALLSIP / P1000MB)**(RD/CP) * thetALLSIP) #Kelvin
tALLSIP = tkALLSIP - 273.15 #Celsius

PHB = np.squeeze(ALLSIP02.variables["PHB"][0,:])
PH = np.squeeze(ALLSIP02.variables["PH"][0,:])
HGT = np.squeeze(ALLSIP02.variables["HGT"][0])
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
        
###Remove spin-up time
tALLSIP = tALLSIP[spin_up:end]
tkALLSIP = tkALLSIP[spin_up:end]

cooper_wrf = np.ones((len(tALLSIP), 96))*np.nan
dmt10_wrf = np.ones((len(tALLSIP), 96))*np.nan
kfgao_wrf = np.ones((len(tALLSIP), 96))*np.nan
kfgao2_wrf = np.ones((len(tALLSIP), 96))*np.nan

###Cooper
for i in range(len(tkALLSIP)):
    for j in range(len(tkALLSIP[0,:])):
        cooper_wrf[i,j] = 0.005*math.exp(0.304 * (T0 - tkALLSIP[i,j]))
        cooper_wrf[i,j] = min(cooper_wrf[i,j],500)

for i in range(len(tkALLSIP)):
    for j in range(len(tkALLSIP[0,:])):
        dmt10_wrf[i,j] = demott_a*((273.15-tkALLSIP[i,j])**(demott_b))*(demott_num**((demott_c*(273.15-tkALLSIP[i,j]))+demott_d)) # NUM OF ICE CRYSTALS L-1
        dmt10_wrf[i,j] = min(dmt10_wrf[i,j],500)
        
for i in range(len(tkALLSIP)):
    for j in range(len(tkALLSIP[0,:])):
        kfgao_wrf[i,j] = (APS_0_5**a_kf + b_kf) * math.exp( - c_kf*(tALLSIP[i,j]) - d_kf) * (1/(Ratio**e_kf)) # NUM OF ICE CRYSTALS L-1
        kfgao_wrf[i,j] = min(kfgao_wrf[i,j],500)

for i in range(len(tkALLSIP)):
    for j in range(len(tkALLSIP[0,:])):
        kfgao2_wrf[i,j] = (APS_0_5**a_kf2 + b_kf2) * math.exp( - c_kf2*(tALLSIP[i,j]) - d_kf2) * (1/(Ratio**e_kf2)) # NUM OF ICE CRYSTALS L-1
        kfgao2_wrf[i,j] = min(kfgao2_wrf[i,j],500)
        


################
### Plotting ###
################
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=14)

figures_directory='./Figures'

fig, axs = plt.subplots(1,3,figsize=(18,8))
plt.subplots_adjust(top=0.9, bottom=0.12, left=0.06, right=0.88, wspace=0.3)

namefigure0 = figures_directory + '/FigureS11.png'

scatter = axs[0].scatter(pine_inps, demott_inps, c=pine_temp, cmap='temperature', alpha=0.8, vmin=-28, vmax=-23)
ax0_cbar = fig.add_axes([0.91, 0.12, 0.02, 0.78])
cbar = fig.colorbar(scatter, cax=ax0_cbar, orientation='vertical')
cbar.set_label('Temperature [°C]', fontsize=16)
axs[0].plot([0.01, 1000], [0.01, 1000], 'k')
axs[0].plot([0.01, 1000], [3*0.01, 3*1000], '--', color='black')
axs[0].plot([0.01, 1000], [1/3*0.01, 1/3*1000], '--', color='black')
axs[0].text(30, 200, "3:1", fontsize=12, color='black', ha='center', va='center')
axs[0].text(150, 20, "1:3", fontsize=12, color='black', ha='center', va='center')
axs[0].set_title('DeMott 2010')
axs[0].set_xlabel('Observed INPs [L$^{-1}$]')
axs[0].set_ylabel('Predicted INPs [L$^{-1}$]')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlim(10**(-2), 10**(3))
axs[0].set_ylim(10**(-2), 10**(3))
text =axs[0].text(0.015, 500, '(a)', fontsize=14)

scatter = axs[1].scatter(pine_inps, meyers_inps, c=pine_temp, cmap='temperature', alpha=0.8, vmin=-28, vmax=-23)
axs[1].plot([0.01, 1000], [0.01, 1000], 'k')
axs[1].plot([0.01, 1000], [3*0.01, 3*1000], '--', color='black')
axs[1].plot([0.01, 1000], [1/3*0.01, 1/3*1000], '--', color='black')
axs[1].text(30, 200, "3:1", fontsize=12, color='black', ha='center', va='center')
axs[1].text(150, 20, "1:3", fontsize=12, color='black', ha='center', va='center')
axs[1].set_title('Meyers 1992')
axs[1].set_xlabel('Observed INPs [L$^{-1}$]')
axs[1].set_ylabel('Predicted INPs [L$^{-1}$]')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlim(10**(-2), 10**(3))
axs[1].set_ylim(10**(-2), 10**(3))
text =axs[1].text(0.015, 500, '(b)', fontsize=14)

scatter = axs[2].scatter(pine_inps, cooper_inps, c=pine_temp, cmap='temperature', alpha=0.8, vmin=-28, vmax=-23)
axs[2].plot([0.01, 1000], [0.01, 1000], 'k')
axs[2].plot([0.01, 1000], [3*0.01, 3*1000], '--', color='black')
axs[2].plot([0.01, 1000], [1/3*0.01, 1/3*1000], '--', color='black')
axs[2].text(30, 200, "3:1", fontsize=12, color='black', ha='center', va='center')
axs[2].text(150, 20, "1:3", fontsize=12, color='black', ha='center', va='center')
axs[2].set_title('Cooper 1986')
axs[2].set_xlabel('Observed INPs [L$^{-1}$]')
axs[2].set_ylabel('Predicted INPs [L$^{-1}$]')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlim(10**(-2), 10**(3))
axs[2].set_ylim(10**(-2), 10**(3))
text =axs[2].text(0.015, 500, '(c)', fontsize=14)

plt.show()

fig.savefig(namefigure0, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')