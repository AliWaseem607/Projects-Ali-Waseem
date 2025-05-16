# -*- coding: utf-8 -*-

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import interp1d
import glob

plt.rcParams['font.size']=15

#This is the time axis for WRF outputs every 5 min
wrf_time = np.arange(np.datetime64('2021-12-17T00:00'), np.datetime64('2021-12-19T12:05'),dtype='datetime64[s]')[::300]
spin_up = np.where(wrf_time == np.datetime64('2021-12-17T22:00'))[0][0]
end = np.where(wrf_time == np.datetime64('2021-12-19T12:00'))[0][0] + 1
wrf_time=wrf_time[spin_up:end]

#This is the time axis for CR-SIM outouts every 5 min
wrf_time2 = np.arange(np.datetime64('2021-12-17T21:00'), np.datetime64('2021-12-19T12:05'),dtype='datetime64[s]')[::300]
spin_up2 = np.where(wrf_time2 == np.datetime64('2021-12-17T22:00'))[0][0]
end2 = np.where(wrf_time2 == np.datetime64('2021-12-19T12:00'))[0][0] + 1
wrf_time2=wrf_time2[spin_up2:end2]


def load_crsim_data(data_dir, file_prefix, spin_up, end):
    REFL = np.load(f"{data_dir}{file_prefix}.npy")[spin_up:end]
    return REFL


###################
# CR-SIM data     #
###################
crsim_data_dir = './Data/CR-SIM/'
REFL_CONTROL = load_crsim_data(crsim_data_dir, 'REFL_CONTROL_VL', spin_up2,end2)
REFL_DEMOTT = load_crsim_data(crsim_data_dir, 'REFL_DEMOTT_VL', spin_up2,end2)
REFL_ALLSIP = load_crsim_data(crsim_data_dir, 'REFL_ALLSIP_VL', spin_up2,end2)


###################################
# W-prof reflectivity data        #
###################################
def load_wprof_data(file_path):
    nc = Dataset(file_path)
    dtime = nc.variables['Time'][:].astype(int)
    time = [datetime.datetime.utcfromtimestamp(tt) for tt in dtime]
    SNR = nc.variables['SnR'][:]
    Ze = nc.variables['Ze'][:]
    Ze[SNR<-14]=np.nan
    Ze_corr = nc.variables['Ze_corrected'][:]
    Ze_corr[SNR<-14]=np.nan
    Rgates = nc.variables['Rgate'][:]
    
    return time, Ze, Ze_corr, Rgates

wprof_time, carmel_Ze, carmel_Ze_corr, Rgates = load_wprof_data('./Data/WProf/Attenuation_correction/HELMOS_att_corr.nc')

carmel_time = np.array(wprof_time, dtype='datetime64[s]')


######################
# WRF model outputs  #
######################
out_dir ='./Data/WRF'

simuALLSIP = out_dir + '/wrfout_d03_ALLSIP_VL.nc'
ALLSIP = Dataset(simuALLSIP)

simuCONTROL = out_dir + '/wrfout_d03_CONTROL_VL.nc'
CONTROL = Dataset(simuCONTROL)

simuDEMOTT = out_dir + '/wrfout_d03_DEMOTT_VL.nc'
DEMOTT = Dataset(simuDEMOTT)

### WRF constants
RA=287.15
RD=287.0
CP=1004.5
P1000MB=100000.0
EPS=0.622

presALLSIP = np.squeeze(ALLSIP.variables['P'][:] + ALLSIP.variables['PB'][:])
thetALLSIP = np.squeeze(ALLSIP.variables['T'][:] + 300.0)
qvALLSIP = np.squeeze(ALLSIP.variables['QVAPOR'][:])
tkALLSIP = ((presALLSIP / P1000MB)**(RD/CP) * thetALLSIP)
tALLSIP = tkALLSIP - 273.15 
tvALLSIP = tkALLSIP * (EPS + qvALLSIP) / (EPS * (1. + qvALLSIP))
rhoALLSIP = presALLSIP/RA/tvALLSIP
icncALLSIP = np.squeeze((ALLSIP.variables['QNICE'][:] + ALLSIP.variables['QNSNOW'][:] + ALLSIP.variables['QNGRAUPEL'][:]))*rhoALLSIP*10**-3 #L-1
iwcALLSIP = np.squeeze((ALLSIP.variables['QICE'][:] + ALLSIP.variables['QSNOW'][:] + ALLSIP.variables['QGRAUP'][:]))*rhoALLSIP*10**3 #gm-3
lwcALLSIP = np.squeeze((ALLSIP.variables['QCLOUD'][:] + ALLSIP.variables['QRAIN'][:]))*rhoALLSIP*10**3 #gm-3
brALLSIP = np.squeeze(ALLSIP.variables['DNI_BR'][:])*rhoALLSIP*10**-3 #L-1s-1
hmALLSIP = np.squeeze(ALLSIP.variables['DNI_HM'][:])*rhoALLSIP*10**-3 #L-1s-1
dsALLSIP = np.squeeze((ALLSIP.variables['DNI_DS1'][:]) + (ALLSIP.variables['DNI_DS2'][:]) + (ALLSIP.variables['DNS_BF1'][:]) + (ALLSIP.variables['DNG_BF1'][:]))*rhoALLSIP*10**-3 #L-1s-1
sbALLSIP = np.squeeze((ALLSIP.variables['DNI_SBS'][:]) + (ALLSIP.variables['DNI_SBG'][:]))*rhoALLSIP*10**-3 #L-1s-1
pipALLSIP = np.squeeze((ALLSIP.variables['DNI_CON'][:]) + (ALLSIP.variables['DNI_IMM'][:]) + (ALLSIP.variables['DNI_NUC'][:]) + (ALLSIP.variables['DNS_CCR'][:]))*rhoALLSIP*10**-3 #L-1s-1
aggALLSIP = abs(np.squeeze(ALLSIP.variables['DNS_AGG'][:]))*rhoALLSIP*10**-3 #L-1s-1
rimALLSIP = np.squeeze(ALLSIP.variables['DQC_RIM'][:])*rhoALLSIP*10**3 #gm-3s-1
depALLSIP = np.squeeze(ALLSIP.variables['DQI_DEP'][:]+ALLSIP.variables['DQS_DEP'][:]+ALLSIP.variables['DQG_DEP'][:])*rhoALLSIP*10**3 #gm-3s-1
rhiALLSIP = np.squeeze(ALLSIP.variables['DSI_TEN'][:])*100
subALLSIP = abs(np.squeeze(ALLSIP.variables['DQI_SUB'][:]+ALLSIP.variables['DQS_SUB'][:]-ALLSIP.variables['DQS_DEP'][:]))*rhoALLSIP*10**3 #gm-3s-1 #this is because DQS_SUB includes also the deposition effects in the tendency

icncALLSIP[icncALLSIP <= 10**(-5)] = np.nan
lwcALLSIP[lwcALLSIP <= 10**(-6)] = np.nan
iwcALLSIP[iwcALLSIP <= 10**(-6)] = np.nan
pipALLSIP[pipALLSIP <= 0] = np.nan
aggALLSIP[aggALLSIP <= 0] = np.nan

icncALLSIP = icncALLSIP[spin_up:end]
lwcALLSIP = lwcALLSIP[spin_up:end]
tALLSIP = tALLSIP[spin_up:end]
iwcALLSIP = iwcALLSIP[spin_up:end]
brALLSIP = brALLSIP[spin_up:end]
hmALLSIP = hmALLSIP[spin_up:end]
dsALLSIP = dsALLSIP[spin_up:end]
sbALLSIP = sbALLSIP[spin_up:end]
pipALLSIP = pipALLSIP[spin_up:end]
aggALLSIP = aggALLSIP[spin_up:end]
rimALLSIP = rimALLSIP[spin_up:end]
depALLSIP = depALLSIP[spin_up:end]
rhiALLSIP = rhiALLSIP[spin_up:end]

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

ZZmiddle=ZZmiddle/1000  #in m


PHB2 = np.squeeze(DEMOTT.variables["PHB"][0,:])
PH2 = np.squeeze(DEMOTT.variables["PH"][0,:])
HGT2 = np.squeeze(DEMOTT.variables["HGT"][0])
ZZASL2 = (PH2+PHB2)/9.81
ZZ2 = (PH2+PHB2)/9.81-HGT2
ZZ_km2 = ZZ2/1000
dz2=np.zeros((len(ZZ2)-1))
ZZmiddle2=np.zeros((len(ZZ2)-1))

kk = 0

for jj in range(len(ZZ2)-1):
        dz2[kk] = (ZZ2[kk+1]-ZZ2[kk])/2
        ZZmiddle2[kk] = dz2[kk] + ZZ2[kk]
        kk=kk+1

ZZmiddle2=ZZmiddle2/1000  #in m


PHB3 = np.squeeze(ALLSIP.variables["PHB"][0,:])
PH3 = np.squeeze(ALLSIP.variables["PH"][0,:])
HGT3 = np.squeeze(ALLSIP.variables["HGT"][0])
ZZASL3 = (PH3+PHB3)/9.81
ZZ3 = (PH3+PHB3)/9.81-HGT3
ZZ_km3 = ZZ3/1000
dz3=np.zeros((len(ZZ3)-1))
ZZmiddle3=np.zeros((len(ZZ3)-1))

kk = 0

for jj in range(len(ZZ3)-1):
        dz3[kk] = (ZZ3[kk+1]-ZZ3[kk])/2
        ZZmiddle3[kk] = dz3[kk] + ZZ3[kk]
        kk=kk+1
    
ZZmiddle3=ZZmiddle3/1000  #in m


################
# W-prof data  #
################
out_dir ='./Data/WProf/LEVEL0'

lv0 = out_dir + '/211218_090002_P06_ZEN_LV0.nc'
nc0 = Dataset(lv0)
t = nc0.variables['Time']
dt = [datetime.datetime.utcfromtimestamp(tt) for tt in t]

radar_time = np.array(dt, dtype='datetime64[s]')

#2nd cloud Period
tstmp = np.datetime64('2021-12-18T09:19:54')
i_t =  np.where(radar_time == tstmp)[0][0] 

#WProf uses three chirps, whose ranges are as follows: chirp 0:
#104-998 m, chirp 1: 1008-3496 m, chirp 2: 3512-8683 m
spec0 = np.array(nc0.variables['Doppler-spectrum-vert-chirp0'][:])
spec1 = np.array(nc0.variables['Doppler-spectrum-vert-chirp1'][:])
spec2 = np.array(nc0.variables['Doppler-spectrum-vert-chirp2'][:])

rg0 = np.array(nc0.variables['Rgate_chirp0'][:])/1000
vel0 = np.array(nc0.variables['Vfft_chirp0'][:])
rg1 = np.array(nc0.variables['Rgate_chirp1'][:])/1000
vel1 = np.array(nc0.variables['Vfft_chirp1'][:])
rg2 = np.array(nc0.variables['Rgate_chirp2'][:])/1000
vel2 = np.array(nc0.variables['Vfft_chirp2'][:])

dv0 = vel0[1]-vel0[0]
dv1 = vel1[1]-vel1[0]
dv2 = vel2[1]-vel2[0]

linspec0 = 10*np.log10(spec0/dv0)
linspec1 = 10*np.log10(spec1/dv1)
linspec2 = 10*np.log10(spec2/dv2)


# Find the indices of the two closest timestamps
idx1 = np.searchsorted(wrf_time, tstmp, side="right") - 1
idx2 = idx1 + 1

# Interpolate the temperature value at time t using linear interpolation
x1, x2 = wrf_time[idx1], wrf_time[idx2]
y1, y2 = tALLSIP[idx1], tALLSIP[idx2]
dt1 = tstmp - x1
y_interp = y1 + (dt1) * (y2 - y1) / (x2 - x1)


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

namefigure = figures_directory + '/Figure5.png'

fig,axs=plt.subplots(1,3,figsize=(20,12))

plt.subplots_adjust(top=0.80, bottom=0.12, left=0.05, right=0.86, hspace=0.20)

plt.draw()
p0 = axs[0].get_position().get_points().flatten()

im1=axs[0].pcolormesh(vel0,rg0,linspec0[i_t],vmin=-40,vmax=25,cmap='turbo',shading='nearest')
axs[0].set_ylabel('Altitude above radar [km]') 
im2=axs[0].pcolormesh(vel1,rg1,linspec1[i_t],vmin=-40,vmax=25,cmap='turbo',shading='nearest')
im=axs[0].pcolormesh(vel2,rg2,linspec2[i_t],vmin=-40,vmax=25,cmap='turbo',shading='nearest')
ax0_cbar = fig.add_axes([p0[0], 0.87, p0[2]-p0[0], 0.02])
cbar = plt.colorbar(im1, cax = ax0_cbar, orientation='horizontal')
cbar.set_label('spectral Ze [dBsZ]',fontsize=16)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
axs[0].set_xlim(-7.5,7.5)
axs[0].set_ylim(0.13,3) 
axs[0].grid()
axs[0].set_xlabel('Doppler velocity [m s$^{-1}$]')
title = dt[i_t].strftime("%d/%m/%Y %H:%M:%S")
axs[0].set_title(title)
axs[0].text(-6.5, 2.85, '(a)', fontsize=14)


interp_func = interp1d(y_interp,ZZmiddle3)

pos = 4.0

t10 = -10.0
iso10 = interp_func(t10)
axs[0].axhline(iso10, color='black', linestyle='--', linewidth=2)
axs[0].text(pos, iso10+0.02, '-10°C', color='black', fontsize=12)

t13 = -13.0
iso13 = interp_func(t13)
axs[0].axhline(iso13, color='black', linestyle='--', linewidth=2)
axs[0].text(pos, iso13+0.02, '-13°C', color='black', fontsize=12)

t16 = -16.0
iso16 = interp_func(t16)
axs[0].axhline(iso16, color='black', linestyle='--', linewidth=2)
axs[0].text(pos, iso16+0.02, '-16°C', color='black', fontsize=12)

t20 = -20.0
iso20 = interp_func(t20)
axs[0].axhline(iso20, color='black', linestyle='--', linewidth=2)
axs[0].text(pos, iso20+0.02, '-20°C', color='black', fontsize=12)

t25 = -25.0
iso25 = interp_func(t25)
axs[0].axhline(iso25, color='black', linestyle='--', linewidth=2)
axs[0].text(pos, iso25+0.02, '-25°C', color='black', fontsize=12)


#WProf
t1a = np.datetime64('2021-12-18T07:30')
t2a = np.datetime64('2021-12-18T12:00')
it1 = np.searchsorted(carmel_time, t1a, side="right")-1
it2 = np.searchsorted(carmel_time, t2a, side="right")

#CONTROL
t3a = np.datetime64('2021-12-18T07:30')
t4a = np.datetime64('2021-12-18T16:20')
it3 = np.searchsorted(wrf_time2, t3a, side="right")-1
it4 = np.searchsorted(wrf_time2, t4a, side="right")-1

#DEMOTT
t5a = np.datetime64('2021-12-18T07:30')
t6a = np.datetime64('2021-12-18T17:20')
it5 = np.searchsorted(wrf_time2, t5a, side="right")-1
it6 = np.searchsorted(wrf_time2, t6a, side="right")-1

#ALLSIP
t7a = np.datetime64('2021-12-18T07:30')
t8a = np.datetime64('2021-12-18T17:45')
it7 = np.searchsorted(wrf_time2, t7a, side="right")-1
it8 = np.searchsorted(wrf_time2, t8a, side="right")-1

diff_t7 = np.abs(wrf_time2 - t7a)
diff_t8 = np.abs(wrf_time2 - t8a)

idx_t7 = np.argmin(diff_t7)
idx_t8 = np.argmin(diff_t8)

t_ALLSIP_med1 = np.nanmedian(tALLSIP[idx_t7:idx_t8], axis=0)
interp_func2 = interp1d(t_ALLSIP_med1,ZZmiddle3)


Ze_plot25 = np.nanpercentile(carmel_Ze_corr[it1:it2], 25, axis=0)
Ze_plot75 = np.nanpercentile(carmel_Ze_corr[it1:it2], 75, axis=0)
CONTROL25 = np.nanpercentile(REFL_CONTROL[it3:it4], 25, axis=0)
CONTROL75 = np.nanpercentile(REFL_CONTROL[it3:it4], 75, axis=0)
DEMOTT25 = np.nanpercentile(REFL_DEMOTT[it5:it6], 25, axis=0)
DEMOTT75 = np.nanpercentile(REFL_DEMOTT[it5:it6], 75, axis=0)
ALLSIP25 = np.nanpercentile(REFL_ALLSIP[it7:it8], 25, axis=0)
ALLSIP75 = np.nanpercentile(REFL_ALLSIP[it7:it8], 75, axis=0)

axs[1].plot(np.nanmedian(carmel_Ze_corr[it1:it2,:],axis=0), Rgates, color='dimgray',linewidth=2.5, label='WProf')
axs[1].fill_betweenx(Rgates, Ze_plot25, Ze_plot75, alpha=0.3, color='dimgray')
axs[1].plot(np.nanmedian(REFL_CONTROL[it3:it4,:],axis=0), ZZ_km, color='black',linewidth=2.5, label='CONTROL')
axs[1].plot(np.nanmedian(REFL_DEMOTT[it5:it6,:],axis=0), ZZ_km2, color='cyan',linewidth=2.5, label='DEMOTT')
axs[1].plot(np.nanmedian(REFL_ALLSIP[it7:it8,:],axis=0), ZZ_km3, color='blue',linewidth=2.5, label='ALLSIP')


axs[1].legend(loc='best')
text =axs[1].text(-34, 2.85, '(b)', fontsize=14)


pos = -30

axs[1].axhline(iso10, color='black', linestyle='--', linewidth=2)
axs[1].text(pos, iso10+0.02, '-10°C', color='black', fontsize=12)

axs[1].axhline(iso13, color='black', linestyle='--', linewidth=2)
axs[1].text(pos, iso13+0.02, '-13°C', color='black', fontsize=12)

axs[1].axhline(iso16, color='black', linestyle='--', linewidth=2)
axs[1].text(pos, iso16+0.02, '-16°C', color='black', fontsize=12)

axs[1].axhline(iso20, color='black', linestyle='--', linewidth=2)
axs[1].text(pos, iso20+0.02, '-20°C', color='black', fontsize=12)

axs[1].axhline(iso25, color='black', linestyle='--', linewidth=2)
axs[1].text(pos, iso25+0.02, '-25°C', color='black', fontsize=12)


axs[1].set_ylabel('Altitude [km]')
axs[1].set_xlabel(r'Ze$_{W}$ [dBZ]')
axs[1].set_ylim(0.13, 3.0)
axs[1].set_xlim(-35,25)


#Define a 10-min window
tstmp2 = tstmp - np.timedelta64(5,'m')
t1 = np.searchsorted(wrf_time2, tstmp2, side="right")+1

tstmp3 = tstmp + np.timedelta64(5,'m')
t2 = np.searchsorted(wrf_time2, tstmp3, side="right")+1

BR25 = np.nanpercentile(brALLSIP[t1:t2], 25, axis=0)
BR75 = np.nanpercentile(brALLSIP[t1:t2], 75, axis=0)
SBR25 = np.nanpercentile(sbALLSIP[t1:t2], 25, axis=0)
SBR75 = np.nanpercentile(sbALLSIP[t1:t2], 75, axis=0)
AGG25 = np.nanpercentile(aggALLSIP[t1:t2], 25, axis=0)
AGG75 = np.nanpercentile(aggALLSIP[t1:t2], 75, axis=0)
RIM25 = np.nanpercentile(rimALLSIP[t1:t2], 25, axis=0)
RIM75 = np.nanpercentile(rimALLSIP[t1:t2], 75, axis=0)
DEP25 = np.nanpercentile(depALLSIP[t1:t2], 25, axis=0)
DEP75 = np.nanpercentile(depALLSIP[t1:t2], 75, axis=0)

line1, = axs[2].plot(np.nanmedian(brALLSIP[t1:t2,:],axis=0), ZZmiddle3, '-', color='darkviolet', linewidth=2.5, label='BR')
axs[2].fill_betweenx(ZZmiddle3, BR25, BR75, alpha=0.3, color='darkviolet')
line4, = axs[2].plot(np.nanmedian(aggALLSIP[t1:t2,:],axis=0), ZZmiddle3, '-', color='red', linewidth=2.5, label='Aggregation')
axs[2].fill_betweenx(ZZmiddle3, AGG25, AGG75, alpha=0.3, color='red')
line3, = axs[2].plot(np.nanmedian(sbALLSIP[t1:t2,:],axis=0), ZZmiddle3, '-', color='magenta', linewidth=2.5, label='SUBBR')
axs[2].fill_betweenx(ZZmiddle3, SBR25, SBR75, alpha=0.3, color='magenta')

pos = 1

axs[2].axhline(iso10, color='black', linestyle='--', linewidth=2)
axs[2].text(pos, iso10+0.02, '-10°C', color='black', fontsize=12)

axs[2].axhline(iso13, color='black', linestyle='--', linewidth=2)
axs[2].text(pos, iso13+0.02, '-13°C', color='black', fontsize=12)

axs[2].axhline(iso16, color='black', linestyle='--', linewidth=2)
axs[2].text(pos, iso16+0.02, '-16°C', color='black', fontsize=12)

axs[2].axhline(iso20, color='black', linestyle='--', linewidth=2)
axs[2].text(pos, iso20+0.02, '-20°C', color='black', fontsize=12)

axs[2].axhline(iso25, color='black', linestyle='--', linewidth=2)
axs[2].text(pos, iso25+0.02, '-25°C', color='black', fontsize=12)

axs[2].set_xlabel("Number tendency [L$^{-1}$ s$^{-1}$]")
axs[2].set_xlim(1e-8, 1e1)
axs[2].set_xscale('log')
axs[2].set_ylabel("Altitude [km]")
axs[2].set_ylim(0.13, 3)
axs[2].text(2*10**(-8), 2.85, '(c)', fontsize=14)
axs[2].grid()

ax2 = axs[2].twiny()
line5, = ax2.plot(np.nanmedian(rimALLSIP[t1:t2,:],axis=0), ZZmiddle3, '-', color='#FFDB58', linewidth=2.5, label='Riming')
ax2.fill_betweenx(ZZmiddle3, RIM25, RIM75, alpha=0.3, color='#FFDB58')
line6, = ax2.plot(np.nanmedian(depALLSIP[t1:t2,:],axis=0), ZZmiddle3, '-', color='coral', linewidth=2.5, label='Vapor deposition')
ax2.fill_betweenx(ZZmiddle3, DEP25, DEP75, alpha=0.3, color='coral')
ax2.set_xlabel("Mass tendency [g m$^{-3}$ s$^{-1}$]", color='coral')
ax2.set_xlim(1e-8, 1e-2)
ax2.set_xscale('log')
ax2.spines['top'].set_color('coral')
ax2.tick_params(axis='x', colors='coral')

lines = [line1, line3, line4, line5, line6]
labels = [line.get_label() for line in lines]

legend = axs[2].legend(lines, labels, loc='upper left', ncol=1, bbox_to_anchor=(0.86, 0.6), bbox_transform=plt.gcf().transFigure)


plt.show()

fig.savefig(namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')