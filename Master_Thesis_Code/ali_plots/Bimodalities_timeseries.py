# -*- coding: utf-8 -*-

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import matplotlib.dates as mdates
import netCDF4 as nc


plt.rcParams['font.size']=15


###################################
# W-prof data                     #
###################################
folder_path ='./Data/WProf/LEVEL0'

linspec0 = None
linspec1 = None
linspec2 = None

x_axis = np.array([])

file_list = sorted(glob.glob(f"{folder_path}\\*P06_ZEN_LV0.nc"))

for element in file_list:
    
    nc = Dataset(element)
    
    dt = [datetime.datetime.utcfromtimestamp(tt) for tt in nc.variables['Time']]

    if dt[-1] < datetime.datetime(2021, 12, 18, 3, 45):
        continue

    if dt[-1] > datetime.datetime(2021, 12, 18, 6, 5):
        continue 
    
    #WProf uses three chirps, whose ranges are as follows: chirp 0:
    #104-998 m, chirp 1: 1008-3496 m, chirp 2: 3512-8683 m
    spec0 = np.array(nc.variables['Doppler-spectrum-vert-chirp0'][:])
    spec1 = np.array(nc.variables['Doppler-spectrum-vert-chirp1'][:])
    spec2 = np.array(nc.variables['Doppler-spectrum-vert-chirp2'][:])
    
    rg0 = np.array(nc.variables['Rgate_chirp0'][:])/1000
    vel0 = np.array(nc.variables['Vfft_chirp0'][:])
    rg1 = np.array(nc.variables['Rgate_chirp1'][:])/1000
    vel1 = np.array(nc.variables['Vfft_chirp1'][:])
    rg2 = np.array(nc.variables['Rgate_chirp2'][:])/1000
    vel2 = np.array(nc.variables['Vfft_chirp2'][:])
    
    dv0 = vel0[1]-vel0[0]
    dv1 = vel1[1]-vel1[0]
    dv2 = vel2[1]-vel2[0]
    
    lin0 = 10*np.log10(spec0/dv0)
    lin1 = 10*np.log10(spec1/dv1)
    lin2 = 10*np.log10(spec2/dv2)
    
    if linspec0 is None:
        linspec0 = lin0 
    else:
        linspec0 = np.concatenate((linspec0, lin0), axis=0)  
    
    if linspec1 is None:
        linspec1 = lin1 
    else:
        linspec1 = np.concatenate((linspec1, lin1), axis=0)
        
    if linspec2 is None:
        linspec2 = lin2 
    else:
        linspec2 = np.concatenate((linspec2, lin2), axis=0)

    x_axis = np.append(x_axis, dt)

radar_time = np.array(x_axis, dtype='datetime64[s]')


######### Plotting ##########
figures_directory='./Figures'
csfont = {'fontname':'Comic Sans MS'}

plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=14)


figures_directory='./Figures'


target = 1.08 #km
alt = np.abs(rg0 - target).argmin()

tstmp1 = np.datetime64('2021-12-18T03:50')
t1 = np.searchsorted(radar_time, tstmp1, side="right") - 1

tstmp2 = np.datetime64('2021-12-18T06:00')
t2 = np.searchsorted(radar_time, tstmp2, side="right")

date_fmt = mdates.DateFormatter('%H:%M')
xlocator = mdates.MinuteLocator(interval = 15)

fig,axs = plt.subplots(figsize=(14,6))

title = f"Altitude: {rg0[alt]:.2f} km"
titlefig = "Nimbostratus cloud_2"

im1=axs.pcolormesh(radar_time[t1:t2+1],vel0,linspec0[t1:t2+1,alt,:].T,vmin=np.nanmin(linspec0[t1:t2+1,alt,:]),vmax=np.nanmax(linspec0[t1:t2+1,alt,:]),cmap='turbo',shading='gouraud')
axs.set_xlabel('Time [UTC]')
plt.colorbar(im1,label='Spectral Ze [dBsZ]',pad=.05,ax=axs)
axs.set_ylim(-4,4)
axs.set_xlim(np.datetime64('2021-12-18T03:50'), np.datetime64('2021-12-18T06:00')) 
axs.grid()
axs.set_ylabel('Doppler velocity [m s$^{-1}$]')
axs.xaxis.set_major_locator(xlocator)
axs.xaxis.set_major_formatter(date_fmt)

plt.show()

fig.savefig(figures_directory + "/" + f"{titlefig}_spectrogram.png",dpi=300,format="png",pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight',facecolor='w')

