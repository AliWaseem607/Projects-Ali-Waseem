# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from netCDF4 import Dataset  # type: ignore
from scipy.interpolate import interp1d
from wrf import ALL_TIMES, getvar, to_np

from utils import get_wrf_times, load_radar_data, load_wprof_data

plt.rcParams['figure.facecolor']='white'

#suppress warnings
warnings.filterwarnings('ignore')


def plot_reflectivities(
        CONTROL: Dataset,
        DEMOTT: Dataset,
        ALLSIP: Dataset,
        REFL_CONTROL:np.ndarray,
        REFL_DEMOTT:np.ndarray,
        REFL_ALLSIP:np.ndarray,
        namelist_path: Path,
        spinup_time: np.timedelta64,
        save_path: Path,
        wprof_data_path: Path,
) -> None:
        
        # ###################################
        # # W-prof reflectivity data        #
        # ###################################
        
        wprof_time, carmel_Ze, carmel_Ze_corr, Rgates = load_wprof_data(
               Path(wprof_data_path,'Attenuation_correction/HELMOS_att_corr.nc')
        )

        radar_time = np.array(wprof_time, dtype='datetime64[s]')

        wrf_time = get_wrf_times(namelist_path, spinup_time=spinup_time)
        tick_start_time = pd.Timestamp(wrf_time[0]).to_pydatetime() - datetime.timedelta(hours=4)
        tick_end_time = pd.Timestamp(wrf_time[-1]).to_pydatetime() + datetime.timedelta(hours=2)

        tick_locs = mdates.drange(tick_start_time, tick_end_time, datetime.timedelta(hours=6))
        tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]
        
        assert ALL_TIMES is not None
        
        ### WRF constants
        RA=287.15 # gas constant for air
        RD=287.0 # gas constant for dry air most likely
        CP=1004.5 # Specific Heat Capacity
        P1000MB=100000.0 # Reference Pressure
        EPS=0.622 # Molar mass vapour/Molar mass dry air
                
        presCONTROL = np.squeeze(CONTROL.variables['P'][:] + CONTROL.variables['PB'][:])
        thetCONTROL = np.squeeze(CONTROL.variables['T'][:] + 300.0)
        print(thetCONTROL)
        exit(0)
        qvCONTROL = np.squeeze(CONTROL.variables['QVAPOR'][:])
        tkCONTROL = ((presCONTROL / P1000MB)**(RD/CP) * thetCONTROL) # actual temperature
        tvCONTROL = tkCONTROL * (EPS + qvCONTROL) / (EPS * (1. + qvCONTROL)) # virtual temperature
        rhoCONTROL = presCONTROL/RA/tvCONTROL 
        lwcCONTROL = np.squeeze(CONTROL.variables['QCLOUD'][:] + CONTROL.variables['QRAIN'][:])*rhoCONTROL*10**3 #gm-3
        lwcCONTROL[lwcCONTROL <= 10**(-5)] = np.nan
        zstagCONTROL = np.squeeze(getvar(CONTROL,"zstag",timeidx=ALL_TIMES)) # model height
        zstagCONTROL = zstagCONTROL[:]
        dzCONTROL = np.diff(zstagCONTROL,axis=1)
        lwpCONTROL = np.nansum(lwcCONTROL*dzCONTROL,axis=1) # units become M/L^2
        T2mCONTROL = np.array(getvar(CONTROL,"T2",timeidx=ALL_TIMES)-273.15) # 2m temperature
        tCONTROL = tkCONTROL - 273.15 # convert to C

        PHB = np.squeeze(CONTROL.variables["PHB"][0,:]) # perturbation geopotential
        PH = np.squeeze(CONTROL.variables["PH"][0,:]) # base state geopotential
        HGT = np.squeeze(CONTROL.variables["HGT"][0]) # terrian height
        ZZASL = (PH+PHB)/9.81 # This is the definition of total geopotential height from WRF
        ZZ = (PH+PHB)/9.81-HGT # This is the geopotential height above the surface
        ZZ_km = ZZ/1000
        ZZmiddle = np.mean(np.stack([ZZ[1:], ZZ[:-1]]), axis=0)
                
        presDEMOTT = np.squeeze(DEMOTT.variables['P'][:] + DEMOTT.variables['PB'][:])
        thetDEMOTT = np.squeeze(DEMOTT.variables['T'][:] + 300.0)
        qvDEMOTT = np.squeeze(DEMOTT.variables['QVAPOR'][:])
        tkDEMOTT = ((presDEMOTT / P1000MB)**(RD/CP) * thetDEMOTT)
        tvDEMOTT = tkDEMOTT * (EPS + qvDEMOTT) / (EPS * (1. + qvDEMOTT))
        rhoDEMOTT = presDEMOTT/RA/tvDEMOTT
        lwcDEMOTT = np.squeeze(DEMOTT.variables['QCLOUD'][:] + DEMOTT.variables['QRAIN'][:])*rhoDEMOTT*10**3 #gm-3
        lwcDEMOTT[lwcDEMOTT <= 10**(-5)] = np.nan
        zstagDEMOTT = np.squeeze(getvar(DEMOTT,"zstag",timeidx=ALL_TIMES))
        zstagDEMOTT = zstagDEMOTT[:]
        dzDEMOTT = np.diff(zstagDEMOTT,axis=1)
        lwpDEMOTT = np.nansum(lwcDEMOTT*dzDEMOTT,axis=1)
        T2mDEMOTT = np.array(getvar(DEMOTT,"T2",timeidx=ALL_TIMES)-273.15)
        tDEMOTT = tkDEMOTT - 273.15

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

        presALLSIP = np.squeeze(ALLSIP.variables['P'][:] + ALLSIP.variables['PB'][:])
        thetALLSIP = np.squeeze(ALLSIP.variables['T'][:] + 300.0)
        qvALLSIP = np.squeeze(ALLSIP.variables['QVAPOR'][:])
        tkALLSIP = ((presALLSIP / P1000MB)**(RD/CP) * thetALLSIP)
        tvALLSIP = tkALLSIP * (EPS + qvALLSIP) / (EPS * (1. + qvALLSIP))
        rhoALLSIP = presALLSIP/RA/tvALLSIP
        lwcALLSIP = np.squeeze(ALLSIP.variables['QCLOUD'][:] + ALLSIP.variables['QRAIN'][:])*rhoALLSIP*10**3 #gm-3
        lwcALLSIP[lwcALLSIP <= 10**(-5)] = np.nan
        zstagALLSIP = np.squeeze(getvar(ALLSIP,"zstag",timeidx=ALL_TIMES))
        zstagALLSIP = zstagALLSIP[:]
        dzALLSIP = np.diff(zstagALLSIP,axis=1)
        lwpALLSIP = np.nansum(lwcALLSIP*dzALLSIP,axis=1)
        T2mALLSIP = np.array(getvar(ALLSIP,"T2",timeidx=ALL_TIMES)-273.15)
        tALLSIP = tkALLSIP - 273.15

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


        lwpALLSIP = lwpALLSIP[-len(wrf_time):] ; T2mALLSIP = T2mALLSIP[-len(wrf_time):]
        lwpCONTROL = lwpCONTROL[-len(wrf_time):] ; T2mCONTROL = T2mCONTROL[-len(wrf_time):]
        lwpDEMOTT = lwpDEMOTT[-len(wrf_time):] ; T2mDEMOTT = T2mDEMOTT[-len(wrf_time):]

        tALLSIP = tALLSIP[-len(wrf_time):]
        tCONTROL = tCONTROL[-len(wrf_time):]
        tDEMOTT = tDEMOTT[-len(wrf_time):]


        ### 1st period ###
        #radar
        t1a = np.datetime64('2021-12-17T22:00')
        t2a = np.datetime64('2021-12-18T06:00')
        #control
        t3a = np.datetime64('2021-12-17T22:00')
        t4a = np.datetime64('2021-12-18T04:20') 
        #allsip
        t5a = np.datetime64('2021-12-17T22:00')
        t6a = np.datetime64('2021-12-18T04:45')
        #demott
        t7a = np.datetime64('2021-12-17T22:00')
        t8a = np.datetime64('2021-12-18T04:20')


        ### 2nd period ###
        #radar
        t1b = np.datetime64('2021-12-18T07:30')
        t2b = np.datetime64('2021-12-18T12:00')
        #control
        t3b = np.datetime64('2021-12-18T07:30')
        t4b = np.datetime64('2021-12-18T16:20')
        #allsip
        t5b = np.datetime64('2021-12-18T07:30')
        t6b = np.datetime64('2021-12-18T17:45')
        #demott
        t7b = np.datetime64('2021-12-18T07:30')
        t8b = np.datetime64('2021-12-18T17:20') 


        ### 3rd period ###
        #radar
        t1c = np.datetime64('2021-12-18T12:00')
        t2c = np.datetime64('2021-12-19T12:00')
        #control
        t3c = np.datetime64('2021-12-18T16:20')
        t4c = np.datetime64('2021-12-19T12:00')
        #allsip
        t5c = np.datetime64('2021-12-18T17:45')
        t6c = np.datetime64('2021-12-19T12:00')
        #demott
        t7c = np.datetime64('2021-12-18T17:20')
        t8c = np.datetime64('2021-12-19T12:00')



        #############################
        # Calculations for FigureS1 #
        #############################
        df = pd.DataFrame(carmel_Ze_corr, index=radar_time)
        df.index = pd.to_datetime(df.index, unit='s')
        resampled_df = df.resample('5T').mean()
        averaged_Ze = resampled_df.to_numpy()
        new_time_array = resampled_df.index.to_numpy()

        radar_time_float = radar_time.astype('float')
        wrf_time_float = wrf_time.astype('float')

        interp_func1d = interp1d(radar_time_float, carmel_Ze_corr, axis=0, kind='linear')
        Ze_plot_interp1d = interp_func1d(wrf_time_float)

        ### Median profiles between 1.8 and 2 km ###
        Wprof_median = np.nanmedian(averaged_Ze[:,183:190],axis=1)
        Wprof_p25 = np.nanpercentile(averaged_Ze[:,183:190],25,axis=1)
        Wprof_p75 = np.nanpercentile(averaged_Ze[:,183:190],75,axis=1)

        Wprof_median_int = np.nanmedian(Ze_plot_interp1d[:,183:190],axis=1)
        Wprof_p25_int = np.nanpercentile(Ze_plot_interp1d[:,183:190],25,axis=1)
        Wprof_p75_int = np.nanpercentile(Ze_plot_interp1d[:,183:190],75,axis=1)



        ######### Plotting ##########
        csfont = {'fontname':'Comic Sans MS'}

        plt.rc('font', size=14)
        plt.rc('axes', titlesize=14)
        plt.rc('axes', labelsize=14)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('legend', fontsize=14)
        plt.rc('figure', titlesize=14)

        z1=0
        z2=5
        z3=2

        # create the rectangle patch
        rect1 = patches.Rectangle((t1a, z1), t2a - t1a, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect2 = patches.Rectangle((t3a, z1), t4a - t3a, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect3 = patches.Rectangle((t7a, z1), t8a - t7a, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect4 = patches.Rectangle((t5a, z1), t6a - t5a, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore

        rect12 = patches.Rectangle((t1b, z1), t2b - t1b, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect22 = patches.Rectangle((t3b, z1), t4b - t3b, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect32 = patches.Rectangle((t7b, z1), t8b - t7b, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect42 = patches.Rectangle((t5b, z1), t6b - t5b, z2 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore

        rect13 = patches.Rectangle((t1c, z1), t2c - t1c, z3 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect23 = patches.Rectangle((t3c, z1), t4c - t3c, z3 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect33 = patches.Rectangle((t7c, z1), t8c - t7c, z3 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore
        rect43 = patches.Rectangle((t5c, z1), t6c - t5c, z3 - z1, linewidth=3, edgecolor='turquoise', facecolor='none', linestyle='solid') # type: ignore


        namefigure = str(Path(save_path, '/Figure1.png'))

        fig,axs = plt.subplots(4,1,figsize=(14,14))

        plt.rcParams['font.size']=15

        plt.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.90, hspace=0.20)

        plt.draw()
        p0 = axs[0].get_position().get_points().flatten()

        cmap = get_cmap("plasma",24)

        cs = axs[0].contour(wrf_time, ZZmiddle/1000, (tCONTROL.T), levels=np.arange(-50, 0, 5), colors='dimgray',linewidths=1)
        axs[0].clabel(cs, inline=True, fontsize=12, fmt='%d$^\circ$C', colors='dimgrey') # type: ignore
        axs[0].set_xlim(wrf_time[0], wrf_time[-1])
        im0=axs[0].pcolormesh(wprof_time,Rgates,carmel_Ze_corr.T,vmin=-35,vmax=20,cmap=cmap)
        cax = fig.add_axes([0.92, 0.10, 0.02, 0.85]) # type: ignore
        cbar = plt.colorbar(im0, cax=cax, orientation='vertical')
        cbar.set_label(r'Ze$_{W}$ [dBZ]', fontsize=16)
        cbar.ax.yaxis.set_ticks_position('right')
        axs[0].set_ylabel('Altitude above'+'\nradar [km]')
        axs[0].add_patch(rect1)
        axs[0].add_patch(rect12)
        axs[0].add_patch(rect13)
        axs[0].set_ylim(0, 5)
        axs[0].set_yticks([0,1,2,3,4,5])
        axs[0].set_xticks(tick_locs)
        axs[0].set_xticklabels(tick_labels)
        axs[0].set_xticklabels([])
        axs[0].grid()
        text_loc = np.datetime64('2021-12-19T11:00:00')
        text =axs[0].text(text_loc, 4.5, '(a)', fontsize=14)

        cs = axs[1].contour(wrf_time, ZZmiddle/1000, (tCONTROL.T), levels=np.arange(-50, 0, 5), colors='dimgray',linewidths=1)
        axs[1].clabel(cs, inline=True, fontsize=12, fmt='%d$^\circ$C', colors='dimgrey') # type: ignore
        axs[1].set_xlim(wrf_time[0], wrf_time[-1])
        axs[1].pcolormesh(to_np(wrf_time),to_np(ZZ)/1000,REFL_CONTROL.T,vmin=-35,vmax=20,cmap=cmap)
        axs[1].set_ylabel('Altitude'+'\n [km]')
        axs[1].set_ylim(0, 5)
        axs[1].set_yticks([0,1,2,3,4,5])
        axs[1].add_patch(rect2)
        axs[1].add_patch(rect22)
        axs[1].add_patch(rect23)
        axs[1].set_xticks(tick_locs)
        axs[1].set_xticklabels(tick_labels)
        axs[1].set_xticklabels([])
        axs[1].grid()
        text =axs[1].text(text_loc, 4.5, '(b)', fontsize=14)

        cs = axs[2].contour(wrf_time, ZZmiddle2/1000, (tDEMOTT.T), levels=np.arange(-50, 0, 5), colors='dimgray',linewidths=1)
        axs[2].clabel(cs, inline=True, fontsize=12, fmt='%d$^\circ$C', colors='dimgrey') # type: ignore
        axs[2].set_xlim(wrf_time[0], wrf_time[-1])
        axs[2].pcolormesh(to_np(wrf_time),to_np(ZZ2)/1000,REFL_DEMOTT.T,vmin=-35,vmax=20,cmap=cmap)
        axs[2].set_ylabel('Altitude'+'\n [km]')
        axs[2].set_ylim(0, 5)
        axs[2].set_yticks([0,1,2,3,4,5])
        axs[2].add_patch(rect3)
        axs[2].add_patch(rect32)
        axs[2].add_patch(rect33)
        axs[2].set_xticks(tick_locs)
        axs[2].set_xticklabels(tick_labels)
        axs[2].set_xticklabels([])
        axs[2].grid()
        text =axs[2].text(text_loc, 4.5, '(c)', fontsize=14)

        cs = axs[3].contour(wrf_time, ZZmiddle3/1000, (tALLSIP.T), levels=np.arange(-50, 0, 5), colors='dimgray',linewidths=1)
        axs[3].clabel(cs, inline=True, fontsize=12, fmt='%d$^\circ$C', colors='dimgrey') # type: ignore
        axs[3].set_xlim(wrf_time[0], wrf_time[-1])
        axs[3].pcolormesh(to_np(wrf_time),to_np(ZZ3)/1000,REFL_ALLSIP.T,vmin=-35,vmax=20,cmap=cmap)
        axs[3].set_ylabel('Altitude'+'\n [km]')
        axs[3].set_ylim(0, 5)
        axs[3].set_yticks([0,1,2,3,4,5])
        axs[3].add_patch(rect4)
        axs[3].add_patch(rect42)
        axs[3].add_patch(rect43)
        axs[3].grid()
        axs[3].set_xlabel('Time [UTC]')
        axs[3].set_xticks(tick_locs)
        axs[3].set_xticklabels(tick_labels)
        text =axs[3].text(text_loc, 4.5, '(d)', fontsize=14)

        # plt.show()

        fig.savefig(namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')




        namefigure2 = str(Path(save_path, '/FigureS1.png'))

        fig,axs = plt.subplots(figsize=(14,7))

        plt.subplots_adjust(top=0.87, bottom=0.12, left=0.08, right=0.95, hspace=0.20)

        axs.plot(new_time_array, Wprof_median, color='dimgray',linewidth=2,label='WProf')
        axs.fill_between(new_time_array, Wprof_p25, Wprof_p75, alpha=0.4, color='dimgray')

        axs.plot(wrf_time,np.nanmedian(REFL_CONTROL[:,20:22],axis=1), color='black',linewidth=2.5,label='CONTROL')
        axs.plot(wrf_time,np.nanmedian(REFL_DEMOTT[:,20:22],axis=1), color='cyan',linewidth=2.5,label='DEMOTT')
        axs.plot(wrf_time,np.nanmedian(REFL_ALLSIP[:,20:22],axis=1), color='blue',linewidth=2.5,label='ALLSIP')

        axs.set_xlim(wrf_time[0], wrf_time[-1])
        axs.set_ylabel(r'Median Ze$_{W}$ [dBZ]')
        axs.set_xlabel('Time [UTC]')
        axs.set_xticks(tick_locs)
        axs.set_xticklabels(tick_labels)
        axs.set_ylim(-20,15)
        axs.legend(loc='best',ncol=1)
        axs.grid()

        # plt.show()

        fig.savefig(namefigure2, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')





        namefigure3 = str(Path(save_path, '/FigureS4.png'))
        fig, axs = plt.subplots(figsize=(8,10))

        #WProf
        it1 = np.searchsorted(radar_time, t1c, side="right")-1
        it2 = np.searchsorted(radar_time, t2c, side="right")-1

        #CONTROL
        it3 = np.searchsorted(wrf_time, t3c, side="right")-1
        it4 = np.searchsorted(wrf_time, t4c, side="right")-1

        #ALLSIP
        it5 = np.searchsorted(wrf_time, t5c, side="right")-1
        it6 = np.searchsorted(wrf_time, t6c, side="right")-1

        #DEMOTT
        it7 = np.searchsorted(wrf_time, t7c, side="right")-1
        it8 = np.searchsorted(wrf_time, t8c, side="right")-1


        Ze_rad_med3 = np.nanmedian(carmel_Ze_corr[it1:it2], axis=0)
        Ze_rad_25pct3 = np.nanpercentile(carmel_Ze_corr[it1:it2], 25, axis=0)
        Ze_rad_75pct3 = np.nanpercentile(carmel_Ze_corr[it1:it2], 75, axis=0)
        Ze_CONTROL_med3 = np.nanmedian(REFL_CONTROL[it3:it4], axis=0)
        Ze_ALLSIP_med3 = np.nanmedian(REFL_ALLSIP[it5:it6], axis=0)
        Ze_DEMOTT_med3 = np.nanmedian(REFL_DEMOTT[it7:it8], axis=0)

        # create the interpolation function
        t_ALLSIP_med3 = np.nanmedian(tALLSIP[it5:it6], axis=0)
        interp_func = interp1d(t_ALLSIP_med3,ZZmiddle3/1000)

        thr = 2

        axs.plot(Ze_rad_med3,Rgates,'-',color='dimgray',linewidth=3,label='WProf')
        axs.fill_betweenx(Rgates, Ze_rad_25pct3, Ze_rad_75pct3, color='dimgray', alpha=0.3)
        axs.plot(Ze_CONTROL_med3[thr:],ZZ_km[thr:],'-',color='black',linewidth=2.5,label='CONTROL')
        axs.plot(Ze_DEMOTT_med3[thr:],ZZ_km2[thr:],'-',color='cyan',linewidth=2.5,label='DEMOTT')
        axs.plot(Ze_ALLSIP_med3[thr:],ZZ_km3[thr:],'-',color='blue',linewidth=2.5,label='ALLSIP')
        axs.set_ylim(0,2)
        axs.set_xlim(-35,25)
        axs.set_xlabel(r'Median Ze$_{W}$ [dBZ]')
        axs.set_ylabel("Altitude [km]")
        axs.grid()

        pos = 13.0

        t12 = -12.0
        iso12c = interp_func(t12)
        axs.axhline(iso12c, color='black', linestyle='--', linewidth=2)
        axs.text(pos, iso12c+0.02, '-12°C', color='black', fontsize=12)

        t17 = -17.0
        iso17c = interp_func(t17)
        axs.axhline(iso17c, color='black', linestyle='--', linewidth=2)
        axs.text(pos, iso17c+0.02, '-17°C', color='black', fontsize=12)

        t20 = -20.0
        iso20c = interp_func(t20)
        axs.axhline(iso20c, color='black', linestyle='--', linewidth=2)
        axs.text(pos, iso20c+0.02, '-20°C', color='black', fontsize=12)

        axs.legend(loc='upper center', bbox_to_anchor=(0.48, 1.11), ncol=4)

        # plt.show()

        fig.savefig(namefigure3, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')

if __name__ == "__main__":
        namelist_input = Path("/scratch/waseem/Helmos_control/namelist.input")
        spinup_time = np.timedelta64(22, "h")
        control_path = Path("/home/waseem/Helmos_data_60h/wrfout_Helmos_d03_CONTROL_VL.nc")
        demott_path = Path("/home/waseem/Helmos_data_60h/wrfout_Helmos_d03_DEMOTT_VL.nc")
        allsip_path = Path("/home/waseem/Helmos_data_60h/wrfout_Helmos_d03_ALLSIP_VL.nc")

        ALLSIP = Dataset(allsip_path)
        CONTROL = Dataset(control_path)
        DEMOTT = Dataset(demott_path)
        save_path = Path("/home/waseem/ali_plots/Figures")
        wprof_path = Path("/home/waseem/vivi_paper/Data/WProf")

        print(CONTROL.variables["T"])