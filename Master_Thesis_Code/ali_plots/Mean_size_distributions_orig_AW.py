# -*- coding: utf-8 -*-

from __future__ import print_function

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.special import gamma
from wrf import ALL_TIMES, getvar, to_np

from WRF_Classes import (
    SizeDistributionParameters,
    WRFConstants,
    WRFSimulationMeanSizeDistribution,
)

#%%
#This is the new WRF outputs with a 5-min output frequency
wrf_time = np.arange(np.datetime64('2021-12-17T00:00'), np.datetime64('2021-12-19T12:05'),dtype='datetime64[s]')[::300]
spin_up = np.where(wrf_time == np.datetime64('2021-12-17T22:00'))[0][0]
end = np.where(wrf_time == np.datetime64('2021-12-19T12:00'))[0][0] + 1

tstmp1 = np.datetime64('2021-12-18T08:00')
i_t1 = np.where(wrf_time == tstmp1)[0][0]
tstmp2 = np.datetime64('2021-12-18T17:00')
i_t2 = np.where(wrf_time == tstmp2)[0][0]


### SIZE DISTRIBUTION ###

### ICE CRYSTALS ###
ice_SD = SizeDistributionParameters(
    Dp_min = 1, #um
    Dp_max = 800, #um
    npts= 301
)

### SNOW ###
snow_SD = SizeDistributionParameters(
    Dp_min = 10, #um
    Dp_max = 10000, #um
    npts= 301
)

### GRAUPEL ###
graupel_SD = SizeDistributionParameters(
    Dp_min = 5, #um
    Dp_max = 5000, #um
    npts= 301
)

### RAIN ###
rain_SD = SizeDistributionParameters(
    Dp_min = 10, #um
    Dp_max = 3000, #um
    npts= 301
)

### CLOUD DROPLET ###
cloud_SD = SizeDistributionParameters(
    Dp_min = 0.1, #um
    Dp_max = 20, #um
    npts= 301
)

#####################
### WRF constants ###
#####################
wrf_constants = WRFConstants(
    RA=287.15,
    RD=287.0,
    CP=1004.5,
    P1000MB=100000.0,
    EPS=0.622,
    NDCNST = 200, #cm-3
    pice = 500, #kg/m3
    psnow = 100, #kg/m3
    pgraupel = 900, #kg/m3
    prain = 997, #kg/m3
    pcloud = 997, #kg/m3
    LAMRMIN = 1/(2800),
    LAMRMAX = 1/(20),
    LAMIMIN = 1/(350),
    LAMIMAX = 1,
    LAMSMIN = 1/(2000),
    LAMSMAX = 1/(10),
    LAMGMIN = 1/(2000),
    LAMGMAX = 1/(20),
)

######################
# WRF model outputs  #
######################


out_dir ='/home/waseem/Helmos_data_60h'

simuALLSIP = out_dir + '/wrfout_Helmos_d03_ALLSIP_VL.nc'
ALLSIP = WRFSimulationMeanSizeDistribution(wrfout_path=Path(simuALLSIP), constants=wrf_constants)

simuCONTROL = out_dir + '/wrfout_Helmos_d03_CONTROL_VL.nc'
CONTROL = WRFSimulationMeanSizeDistribution(wrfout_path=Path(simuCONTROL), constants=wrf_constants)

simuDEMOTT = out_dir + '/wrfout_Helmos_d03_DEMOTT_VL.nc'
DEMOTT = WRFSimulationMeanSizeDistribution(wrfout_path=Path(simuDEMOTT), constants=wrf_constants)


# Find nearest grid points

PHB = np.squeeze(DEMOTT.netCDF.variables["PHB"][0,:])
PH = np.squeeze(DEMOTT.netCDF.variables["PH"][0,:])
HGT = np.squeeze(DEMOTT.netCDF.variables["HGT"][0])
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



presALLSIP = np.squeeze(ALLSIP.netCDF.variables['P'][:] + ALLSIP.netCDF.variables['PB'][:])
thetALLSIP = np.squeeze(ALLSIP.netCDF.variables['T'][:] + 300.0)
qvALLSIP = np.squeeze(ALLSIP.netCDF.variables['QVAPOR'][:])
tkALLSIP = ((presALLSIP / wrf_constants.P1000MB)**(wrf_constants.RD/wrf_constants.CP) * thetALLSIP)
tALLSIP = tkALLSIP - 273.15


### INSTANTANEOUS OUTPUTS ###
# Convert specific time to numpy.datetime64
tstmp1 = np.datetime64('2021-12-18T09:20') #03:50 for 1st period # 09:20 for the 2nd period
idt = np.searchsorted(wrf_time, tstmp1, side="right") - 1

lev = 23 #23 = 2.26 km  ZZmiddle[24]/1000 11 for the first period

### INSTANTANEOUS OUTPUTS ###
lev2 = 6 #6 = 0.54 km  #ZZmiddle[6]/1000  3 for the 1st period



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

fig, axs = plt.subplots(2,3, figsize=(18,12))

plt.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.96, hspace=0.3,
                    wspace=0.3)



fig, ax = plt.subplots(2,2,figsize=(20,15))

namefigure = figures_directory + '/FigureS6.png'

cmap = plt.cm.get_cmap('turbo')
levs = np.logspace(np.log10(50), np.log10(3000), 12) #snow characteristic size
cs = ax[0,0].contour(wrf_time[spin_up:end], ZZmiddle/1000, tALLSIP[spin_up:end,:].T, levels=np.arange(-50, 0, 5), colors='black',linewidths=2)
ax[0,0].clabel(cs, inline=True, fontsize=10, fmt='%d$^\circ$C')
lamda = ax[0,0].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/ALLSIP.LAMBDA_S[spin_up:end,:].T), levs, cmap = cmap, norm=colors.LogNorm())
cbar = fig.colorbar(lamda,ax=ax[0,0],pad=.01)
cbar.ax.set_yscale('log')
cbar.set_label('1/lambdas [μm]', fontsize=16)
ax[0,0].set_xlim(wrf_time[spin_up], wrf_time[-1])
ax[0,0].set_ylabel("Height [km]")
ax[0,0].set_ylim(0,5)
ax[0,0].set_yticks([0,1,2,3,4,5])
xfmt = mdates.DateFormatter('%d/%m' +'\n'+ '%H:%M')
xlocator = mdates.MinuteLocator(interval = 360)
ax[0,0].xaxis.set_major_locator(xlocator)
ax[0,0].xaxis.set_major_formatter(xfmt)
text_loc = np.datetime64('2021-12-19T10:30:00')
text =ax[0,0].text(text_loc, 4.7, '(a)', fontsize=14)
for label in ax[0,0].get_xticklabels():
    label.set_ha("right")
    label.set_rotation(0)
plt.grid()

cmap = plt.cm.get_cmap('turbo')
levs = np.logspace(np.log10(10), np.log10(200), 12) #rain characteristic size
cs = ax[0,1].contour(wrf_time[spin_up:end], ZZmiddle/1000, tALLSIP[spin_up:end,:].T, levels=np.arange(-50, 0, 5), colors='black',linewidths=2)
ax[0,1].clabel(cs, inline=True, fontsize=10, fmt='%d$^\circ$C')
lamda = ax[0,1].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/ALLSIP.LAMBDA_R[spin_up:end,:].T), levs, cmap = cmap)
cbar = fig.colorbar(lamda,ax=ax[0,1],pad=.01)
cbar.ax.set_yscale('log')
cbar.set_label('1/lambdar [μm]', fontsize=16)
ax[0,1].set_xlim(wrf_time[spin_up], wrf_time[-1])
ax[0,1].set_ylabel("Height [km]")
ax[0,1].set_ylim(0,5)
ax[0,1].set_yticks([0,1,2,3,4,5])
xfmt = mdates.DateFormatter('%d/%m' +'\n'+ '%H:%M')
xlocator = mdates.MinuteLocator(interval = 360)
text =ax[0,1].text(text_loc, 4.7, '(b)', fontsize=14)
ax[0,1].xaxis.set_major_locator(xlocator)
ax[0,1].xaxis.set_major_formatter(xfmt)
for label in ax[0,1].get_xticklabels():
    label.set_ha("right")
    label.set_rotation(0)
plt.grid()

cmap = plt.cm.get_cmap('turbo')
levs = np.logspace(np.log10(10), np.log10(500), 12) #ice characteristic size
cs = ax[1,0].contour(wrf_time[spin_up:end], ZZmiddle/1000, tALLSIP[spin_up:end,:].T, levels=np.arange(-50, 0, 5), colors='black',linewidths=2)
ax[1,0].clabel(cs, inline=True, fontsize=10, fmt='%d$^\circ$C')
lamda = ax[1,0].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/ALLSIP.LAMBDA_I[spin_up:end,:].T), levs, cmap = cmap, norm=colors.LogNorm())
cbar = fig.colorbar(lamda,ax=ax[1,0],pad=.01)
cbar.ax.set_yscale('log')
cbar.set_label('1/lambdai [μm]', fontsize=16)
ax[1,0].set_xlim(wrf_time[spin_up], wrf_time[-1])
ax[1,0].set_ylabel("Height [km]")
ax[1,0].set_xlabel("Time [UTC]")
ax[1,0].set_ylim(0,5)
ax[1,0].set_yticks([0,1,2,3,4,5])
xfmt = mdates.DateFormatter('%d/%m' +'\n'+ '%H:%M')
xlocator = mdates.MinuteLocator(interval = 360)
ax[1,0].xaxis.set_major_locator(xlocator)
ax[1,0].xaxis.set_major_formatter(xfmt)
text =ax[1,0].text(text_loc, 4.7, '(c)', fontsize=14)
for label in ax[1,0].get_xticklabels():
    label.set_ha("right")
    label.set_rotation(0)
plt.grid()


cmap = plt.cm.get_cmap('turbo')
levs = np.logspace(np.log10(10), np.log10(1000), 12) #graupel characteristic size
cs = ax[1,1].contour(wrf_time[spin_up:end], ZZmiddle/1000, tALLSIP[spin_up:end,:].T, levels=np.arange(-50, 0, 5), colors='black',linewidths=2)
ax[1,1].clabel(cs, inline=True, fontsize=10, fmt='%d$^\circ$C')
lamda = ax[1,1].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/ALLSIP.LAMBDA_G[spin_up:end,:].T), levs, cmap = cmap, norm=colors.LogNorm())
cbar = fig.colorbar(lamda,ax=ax[1,1],pad=.01)
cbar.ax.set_yscale('log')
#cbar.set_label('Lambda ($\mathrm{μm^{-1}}$)', fontsize=16)
cbar.set_label('1/lambdag [μm]', fontsize=16)
ax[1,1].set_xlim(wrf_time[spin_up], wrf_time[-1])
ax[1,1].set_ylabel("Height [km]")
ax[1,1].set_xlabel("Time [UTC]")
ax[1,1].set_ylim(0,5)
ax[1,1].set_yticks([0,1,2,3,4,5])
xfmt = mdates.DateFormatter('%d/%m' +'\n'+ '%H:%M')
xlocator = mdates.MinuteLocator(interval = 360)
ax[1,1].xaxis.set_major_locator(xlocator)
ax[1,1].xaxis.set_major_formatter(xfmt)
text =ax[1,1].text(text_loc, 4.7, '(d)', fontsize=14)
for label in ax[1,1].get_xticklabels():
    label.set_ha("right")
    label.set_rotation(0)
plt.grid()

plt.subplots_adjust(top=0.92, bottom=0.14, left=0.05, right=1.0, wspace=0.15)

plt.show()

fig.savefig(namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')






fig, ax = plt.subplots(figsize=(14,8))
namefigure5 = figures_directory + '/char_size_contour_ALLSIP_cloud.png'
cmap = plt.cm.get_cmap('turbo')
levs = np.logspace(np.log10(0.1), np.log10(4), 15) #rain characteristic size
cs = plt.contour(wrf_time[spin_up:end], ZZmiddle/1000, tALLSIP[spin_up:end,:].T, levels=np.arange(-50, 0, 10), colors='black',linewidths=2)
ax.clabel(cs, inline=True, fontsize=10, fmt='%d$^\circ$C')
lamda = ax.contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/ALLSIP.LAMBDA_C[spin_up:end,:].T), levs, cmap = cmap)
cbar = fig.colorbar(lamda,pad=.01)
cbar.ax.set_yscale('log')
#cbar.set_label('Lambda ($\mathrm{μm^{-1}}$)', fontsize=16)
cbar.set_label('1/lambdac [μm]', fontsize=16)
ax.set_xlim(wrf_time[spin_up], wrf_time[-1])
ax.set_ylabel("Height [km]")
ax.set_xlabel("Time [UTC]")
ax.set_ylim(0,5)
ax.set_yticks([0,1,2,3,4,5])
xfmt = mdates.DateFormatter('%d/%m' +'\n'+ '%H:%M')
xlocator = mdates.MinuteLocator(interval = 360)
ax.xaxis.set_major_locator(xlocator)
ax.xaxis.set_major_formatter(xfmt)
for label in ax.get_xticklabels():
    label.set_ha("right")
    label.set_rotation(0)
plt.grid()
plt.subplots_adjust(top=0.85, bottom=0.12, left=0.05, right=1.0, wspace=0.3)
plt.show()
fig.savefig(namefigure5, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')






namefigure= figures_directory + '/FigureS9.png'

def plot_distribution(
        sim:WRFSimulationMeanSizeDistribution, 
        ice_SD:SizeDistributionParameters,
        snow_SD:SizeDistributionParameters, 
        graupel_SD:SizeDistributionParameters, 
        # rain_SD:SizeDistributionParameters, 
        # cloud_SD:SizeDistributionParameters,  
        ax_ice:Axes,
        ax_snow:Axes,
        ax_graupel:Axes,
        idx_time:int,
        idx_level:int,
        plot_kwargs:dict,
        semilogy_kwargs:dict,
    ) -> None:

    lambdaI = sim.LAMBDA_I[idx_time,idx_level]
    lambdaG = sim.LAMBDA_G[idx_time,idx_level]
    lambdaS = sim.LAMBDA_S[idx_time,idx_level]
    # lambdaC = sim.LAMBDA_C[idx_time,idx_level]
    # lambdaR = sim.LAMBDA_R[idx_time,idx_level]

    N0 = sim.N0_I[idx_time,idx_level]
    N0S = sim.N0_S[idx_time,idx_level]
    N0G = sim.N0_G[idx_time,idx_level]
    # N0R = sim.N0_R[idx_time,idx_level]
    # N0C = sim.N0_C[idx_time,idx_level]


    dndDpI = N0 * np.exp(-lambdaI*ice_SD.Dp)
    dndlogDpI = dndDpI * ice_SD.Dp * np.log(10)
    dndDpS = N0S * np.exp(-lambdaS*snow_SD.Dp)
    dndlogDpS = dndDpS * snow_SD.Dp * np.log(10)
    dndDpG = N0G * np.exp(-lambdaG*graupel_SD.Dp)
    dndlogDpG = dndDpG * graupel_SD.Dp * np.log(10)
    # dndDpR = N0R * math.exp(-lambdaR*rain_SD.Dp)
    # dndlogDpR = dndDpR * rain_SD.Dp * np.log(10)
    # dndDpC = N0C * math.exp(-lambdaC*cloud_SD.Dp)
    # dndlogDpC = dndDpC * cloud_SD.Dp * np.log(10)


    iNi = np.where(dndlogDpI == np.nanmax(dndlogDpI))
    iNs = np.where(dndlogDpS == np.nanmax(dndlogDpS))
    iNg = np.where(dndlogDpG == np.nanmax(dndlogDpG))
    # iNr = np.where(dndlogDpR == np.nanmax(dndlogDpR))
    # iNc = np.where(dndlogDpC == np.nanmax(dndlogDpC))

    ax_ice.plot(ice_SD.Dp*0.001,dndlogDpI, **plot_kwargs)
    ax_snow.plot(snow_SD.Dp*0.001,dndlogDpS, **plot_kwargs)
    ax_graupel.plot(graupel_SD.Dp*0.001,dndlogDpG, **plot_kwargs)
    ax_ice.semilogy([ice_SD.Dp[iNi]*0.001, ice_SD.Dp[iNi]*0.001], [np.array([10**(-7)]), dndlogDpI[iNi]], **semilogy_kwargs)
    ax_snow.semilogy([snow_SD.Dp[iNs]*0.001, snow_SD.Dp[iNs]*0.001], [np.array([10**(-7)]), dndlogDpS[iNs]], **semilogy_kwargs)
    # ax_graupel.semilogy([graupel_SD.Dp[iNg]*0.001, graupel_SD.Dp[iNg]*0.001], [np.ones(iNg[0].shape)*10**(-7), dndlogDpG[iNg]], **semilogy_kwargs)


fig, axs = plt.subplots(2,3, figsize=(18,12))

plt.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.96, hspace=0.3,
                    wspace=0.3)

plot_distribution(
     sim=CONTROL, 
     ice_SD=ice_SD, 
     snow_SD=snow_SD, 
     graupel_SD=graupel_SD, 
     ax_ice=axs[0,0], 
     ax_snow=axs[0,1],
     ax_graupel=axs[0,2],
     idx_time=int(idt),
     idx_level=lev,
     plot_kwargs={"linestyle":"-", "linewidth":2.5, "label":"CONTROL", "color":"k"},
     semilogy_kwargs={"linestyle":"--", "color":"k"}
)
plot_distribution(
     sim=ALLSIP, 
     ice_SD=ice_SD, 
     snow_SD=snow_SD, 
     graupel_SD=graupel_SD, 
     ax_ice=axs[0,0], 
     ax_snow=axs[0,1],
     ax_graupel=axs[0,2],
     idx_time=int(idt),
     idx_level=lev,
     plot_kwargs={"linestyle":"-", "linewidth":2.5, "label":"ALLSIP", "color":"cyan"},
     semilogy_kwargs={"linestyle":"--", "color":"cyan"}
)
plot_distribution(
     sim=DEMOTT, 
     ice_SD=ice_SD, 
     snow_SD=snow_SD, 
     graupel_SD=graupel_SD, 
     ax_ice=axs[0,0], 
     ax_snow=axs[0,1],
     ax_graupel=axs[0,2],
     idx_time=int(idt),
     idx_level=lev,
     plot_kwargs={"linestyle":"-", "linewidth":2.5, "label":"DEMOTT", "color":"blue"},
     semilogy_kwargs={"linestyle":"--", "color":"blue"}
)

axs[0,0].set_ylabel("d($N_{ice}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[0,0].set_xlabel("$D_{p}$ (mm)")
axs[0,0].set_xlim(0.4*0.001,3000*0.001)
axs[0,0].set_ylim(1e-7,1e-1)
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,0].text(5.5*10**(-4),0.04,'(a)', fontsize=14)

axs[0,1].set_ylabel("d($N_{snow}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[0,1].set_xlabel("$D_{p}$ (mm)")
axs[0,1].set_xlim(12*0.0001,30000*0.001)
axs[0,1].set_ylim(10**(-7),1e-1)
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[0,1].text(0.0015,0.04,'(b)', fontsize=14)

axs[0,2].set_ylabel("d($N_{graupel}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[0,2].set_xlabel("$D_{p}$ (mm)")
axs[0,2].set_xlim(10*0.0001,10*1000*0.001)
axs[0,2].set_ylim(10**(-9),10**(-2))
axs[0,2].set_xscale('log')
axs[0,2].set_yscale('log')
axs[0,2].text(0.0012,0.003,'(c)', fontsize=14)
axs[0,2].legend(loc = 'upper right',ncol=1)

plot_distribution(
     sim=CONTROL, 
     ice_SD=ice_SD, 
     snow_SD=snow_SD, 
     graupel_SD=graupel_SD, 
     ax_ice=axs[1,0], 
     ax_snow=axs[1,1],
     ax_graupel=axs[1,2],
     idx_time=int(idt),
     idx_level=lev,
     plot_kwargs={"linestyle":"-", "linewidth":2.5, "label":"CONTROL", "color":"k"},
     semilogy_kwargs={"linestyle":"--", "color":"k"}
)
plot_distribution(
     sim=ALLSIP, 
     ice_SD=ice_SD, 
     snow_SD=snow_SD, 
     graupel_SD=graupel_SD, 
     ax_ice=axs[1,0], 
     ax_snow=axs[1,1],
     ax_graupel=axs[1,2],
     idx_time=int(idt),
     idx_level=lev,
     plot_kwargs={"linestyle":"-", "linewidth":2.5, "label":"CONTROL", "color":"cyan"},
     semilogy_kwargs={"linestyle":"--", "color":"cyan"}
)
plot_distribution(
     sim=DEMOTT, 
     ice_SD=ice_SD, 
     snow_SD=snow_SD, 
     graupel_SD=graupel_SD, 
     ax_ice=axs[1,0], 
     ax_snow=axs[1,1],
     ax_graupel=axs[1,2],
     idx_time=int(idt),
     idx_level=lev,
     plot_kwargs={"linestyle":"-", "linewidth":2.5, "label":"CONTROL", "color":"blue"},
     semilogy_kwargs={"linestyle":"--", "color":"blue"}
)

axs[1,0].set_ylabel("d($N_{ice}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[1,0].set_xlabel("$D_{p}$ (mm)")
axs[1,0].set_xlim(0.4*0.001,3000*0.001)
axs[1,0].set_ylim(1e-7,1e-1)
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,0].text(5.5*10**(-4),0.04,'(d)', fontsize=14)

axs[1,1].set_ylabel("d($N_{snow}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[1,1].set_xlabel("$D_{p}$ (mm)")
axs[1,1].set_xlim(12*0.0001,30000*0.001)
axs[1,1].set_ylim(10**(-7),1e-1)
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
axs[1,1].text(0.0015,0.04,'(e)', fontsize=14)

axs[1,2].set_ylabel("d($N_{graupel}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[1,2].set_xlabel("$D_{p}$ (mm)")
axs[1,2].set_xlim(10*0.0001,10*1000*0.001)
axs[1,2].set_ylim(10**(-9),10**(-2))
axs[1,2].set_xscale('log')
axs[1,2].set_yscale('log')
axs[1,2].text(0.0012,0.003,'(f)', fontsize=14)

plt.show()

fig.savefig(namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
