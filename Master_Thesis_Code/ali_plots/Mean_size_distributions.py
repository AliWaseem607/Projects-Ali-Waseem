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
import matplotlib.colors as colors
from scipy.special import gamma

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
Dp_min1 = 1 #um
Dp_max1 = 800 #um
npts1 = 301
logDp1 = np.linspace(math.log10(Dp_min1), math.log10(Dp_max1), num=npts1)
u8_arr1 = np.array([logDp1[0], logDp1[-1]])
diff_u81 = np.diff(u8_arr1)
dlogDp1 = diff_u81/(npts1-1)
Dp1 = 10**logDp1
dDp1 = Dp1*(10**dlogDp1-1)

### SNOW ###
Dp_min2 = 10 #um
Dp_max2 = 10000 #um
npts2 = 301
logDp2 = np.linspace(math.log10(Dp_min2), math.log10(Dp_max2), num=npts2)
u8_arr2 = np.array([logDp2[0], logDp2[-1]])
diff_u82 = np.diff(u8_arr2)
dlogDp2 = diff_u82/(npts2-1)
Dp2 = 10**logDp2
dDp2 = Dp2*(10**dlogDp2-1)

### GRAUPEL ###
Dp_min3 = 5 #um
Dp_max3 = 5000 #um
npts3 = 301
logDp3 = np.linspace(math.log10(Dp_min3), math.log10(Dp_max3), num=npts3)
u8_arr3 = np.array([logDp3[0], logDp3[-1]])
diff_u83 = np.diff(u8_arr2)
dlogDp3 = diff_u83/(npts3-1)
Dp3 = 10**logDp3
dDp3 = Dp3*(10**dlogDp3-1)

### RAIN ###
Dp_min4 = 10 #um
Dp_max4 = 3000 #um
npts4 = 301
logDp4 = np.linspace(math.log10(Dp_min4), math.log10(Dp_max4), num=npts4)
u8_arr4 = np.array([logDp4[0], logDp4[-1]])
diff_u84 = np.diff(u8_arr4)
dlogDp4 = diff_u84/(npts4-1)
Dp4 = 10**logDp4
dDp4 = Dp4*(10**dlogDp4-1)

### CLOUD DROPLET ###
Dp_min5 = 0.1 #um
Dp_max5 = 20 #um
npts5 = 301
logDp5 = np.linspace(math.log10(Dp_min5), math.log10(Dp_max5), num=npts5)
u8_arr5 = np.array([logDp5[0], logDp5[-1]])
diff_u85 = np.diff(u8_arr5)
dlogDp5 = diff_u85/(npts5-1)
Dp5 = 10**logDp5
dDp5 = Dp5*(10**dlogDp5-1)


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


# Find nearest grid points
time = CONTROL.variables['Times']
indt = range(len(time))

znu = CONTROL.variables['ZNU']
nz = np.size(znu,1)

presCONTROL=np.zeros((len(indt),nz))
lwcCONTROL=np.zeros((len(indt),nz))
iwcCONTROL=np.zeros((len(indt),nz))
ICNCCONTROL=np.zeros((len(indt),nz))
NICE_CONTROL=np.zeros((len(indt),nz))
QICE_CONTROL=np.zeros((len(indt),nz))
NSNOW_CONTROL=np.zeros((len(indt),nz))
QSNOW_CONTROL=np.zeros((len(indt),nz))
NGRAUPEL_CONTROL=np.zeros((len(indt),nz))
QGRAUPEL_CONTROL=np.zeros((len(indt),nz))
lambda_CONTROL=np.zeros((len(indt),nz))
N0_CONTROL=np.zeros((len(indt),nz))  # 145, 99
lambdaS_CONTROL=np.zeros((len(indt),nz))
N0S_CONTROL=np.zeros((len(indt),nz))
lambdaG_CONTROL=np.zeros((len(indt),nz))
N0G_CONTROL=np.zeros((len(indt),nz))
NRAIN_CONTROL=np.zeros((len(indt),nz))
QRAIN_CONTROL=np.zeros((len(indt),nz))
lambdaR_CONTROL=np.zeros((len(indt),nz))
N0R_CONTROL=np.zeros((len(indt),nz))
QCLOUD_CONTROL=np.zeros((len(indt),nz))
lambdaC_CONTROL=np.zeros((len(indt),nz))
N0C_CONTROL=np.zeros((len(indt),nz))


presDEMOTT=np.zeros((len(indt),nz))
lwcDEMOTT=np.zeros((len(indt),nz))
iwcDEMOTT=np.zeros((len(indt),nz))
ICNCDEMOTT=np.zeros((len(indt),nz))
NICE_DEMOTT=np.zeros((len(indt),nz))
QICE_DEMOTT=np.zeros((len(indt),nz))
NSNOW_DEMOTT=np.zeros((len(indt),nz))
QSNOW_DEMOTT=np.zeros((len(indt),nz))
NGRAUPEL_DEMOTT=np.zeros((len(indt),nz))
QGRAUPEL_DEMOTT=np.zeros((len(indt),nz))
lambda_DEMOTT=np.zeros((len(indt),nz))
N0_DEMOTT=np.zeros((len(indt),nz))
lambdaG_DEMOTT=np.zeros((len(indt),nz))
N0G_DEMOTT=np.zeros((len(indt),nz))
lambdaS_DEMOTT=np.zeros((len(indt),nz))
N0S_DEMOTT=np.zeros((len(indt),nz))
NRAIN_DEMOTT=np.zeros((len(indt),nz))
QRAIN_DEMOTT=np.zeros((len(indt),nz))
lambdaR_DEMOTT=np.zeros((len(indt),nz))
N0R_DEMOTT=np.zeros((len(indt),nz))
QCLOUD_DEMOTT=np.zeros((len(indt),nz))
lambdaC_DEMOTT=np.zeros((len(indt),nz))
N0C_DEMOTT=np.zeros((len(indt),nz))


presALLSIP=np.zeros((len(indt),nz))
lwcALLSIP=np.zeros((len(indt),nz))
iwcALLSIP=np.zeros((len(indt),nz))
ICNCALLSIP=np.zeros((len(indt),nz))
NICE_ALLSIP=np.zeros((len(indt),nz))
QICE_ALLSIP=np.zeros((len(indt),nz))
NSNOW_ALLSIP=np.zeros((len(indt),nz))
QSNOW_ALLSIP=np.zeros((len(indt),nz))
NGRAUPEL_ALLSIP=np.zeros((len(indt),nz))
QGRAUPEL_ALLSIP=np.zeros((len(indt),nz))
lambda_ALLSIP=np.zeros((len(indt),nz))
N0_ALLSIP=np.zeros((len(indt),nz))
lambdaS_ALLSIP=np.zeros((len(indt),nz))
N0S_ALLSIP=np.zeros((len(indt),nz))
lambdaG_ALLSIP=np.zeros((len(indt),nz))
N0G_ALLSIP=np.zeros((len(indt),nz))
NRAIN_ALLSIP=np.zeros((len(indt),nz))
QRAIN_ALLSIP=np.zeros((len(indt),nz))
lambdaR_ALLSIP=np.zeros((len(indt),nz))
N0R_ALLSIP=np.zeros((len(indt),nz))
QCLOUD_ALLSIP=np.zeros((len(indt),nz))
lambdaC_ALLSIP=np.zeros((len(indt),nz))
N0C_ALLSIP=np.zeros((len(indt),nz))


PHB = np.squeeze(DEMOTT.variables["PHB"][0,:])
PH = np.squeeze(DEMOTT.variables["PH"][0,:])
HGT = np.squeeze(DEMOTT.variables["HGT"][0])
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

### WRF constants
RA=287.15
RD=287.0
CP=1004.5
P1000MB=100000.0
EPS=0.622

NDCNST = 200 #cm-3

pice = 500 #kg/m3
psnow = 100 #kg/m3
pgraupel = 900 #kg/m3
prain = 997 #kg/m3
pcloud = 997 #kg/m3

pi = 3.14159

heta = 0.0005714 * NDCNST + 0.2714
miu = 1/(heta)**2 - 1
miu = max(miu, 2)
miu = min(miu, 10)
alpha = pi*pcloud/6

LAMCMIN = (miu+1)/(60) ; LAMCMAX = (miu+1)  ##given in um-1
LAMRMIN = 1/(2800) ; LAMRMAX = 1/(20)
LAMIMIN = 1/(350) ; LAMIMAX = 1
LAMSMIN = 1/(2000) ; LAMSMAX = 1/(10)
LAMGMIN = 1/(2000) ; LAMGMAX = 1/(20)

presALLSIP = np.squeeze(ALLSIP.variables['P'][:] + ALLSIP.variables['PB'][:])
thetALLSIP = np.squeeze(ALLSIP.variables['T'][:] + 300.0)
qvALLSIP = np.squeeze(ALLSIP.variables['QVAPOR'][:])
tkALLSIP = ((presALLSIP / P1000MB)**(RD/CP) * thetALLSIP)
tALLSIP = tkALLSIP - 273.15

ii=0
 
for ind in indt:
  
   pres_provCONTROL = getvar(CONTROL,"pres",timeidx=ind)
   presCONTROL[ii,:] = np.squeeze(pres_provCONTROL[:])
   tv_provCONTROL = getvar(CONTROL, "tv",timeidx=ind)
   tvCONTROL =np.squeeze(tv_provCONTROL[:])

   pres_provDEMOTT = getvar(DEMOTT,"pres",timeidx=ind)
   presDEMOTT[ii,:] = np.squeeze(pres_provDEMOTT[:])
   tv_provDEMOTT = getvar(DEMOTT, "tv",timeidx=ind)
   tvDEMOTT = np.squeeze(tv_provDEMOTT[:])
   
   pres_provALLSIP = getvar(ALLSIP,"pres",timeidx=ind)
   presALLSIP[ii,:] = np.squeeze(pres_provALLSIP[:])
   tv_provALLSIP = getvar(ALLSIP, "tv",timeidx=ind)
   tvALLSIP = np.squeeze(tv_provALLSIP[:])


   # Compute vertically integrated variable
   rhoCONTROL=presCONTROL[ii,:]/RA/tvCONTROL
   lwc_provCONTROL=getvar(CONTROL,"QCLOUD",timeidx=ind)+getvar(CONTROL,"QRAIN",timeidx=ind) ##LWC = Qcloud + Qrain
   iwc_provCONTROL=getvar(CONTROL,"QICE",timeidx=ind)+getvar(CONTROL,"QSNOW",timeidx=ind)+getvar(CONTROL,"QGRAUP",timeidx=ind)
   lwcCONTROL[ii,:]=np.squeeze(lwc_provCONTROL[:])*rhoCONTROL*10**3
   iwcCONTROL[ii,:]=np.squeeze(iwc_provCONTROL[:])*rhoCONTROL*10**3
   ICNC_provCONTROL=getvar(CONTROL,"QNICE",timeidx=ind)+getvar(CONTROL,"QNSNOW",timeidx=ind)+getvar(CONTROL,"QNGRAUPEL",timeidx=ind)
   ICNCCONTROL[ii,:]=np.squeeze(ICNC_provCONTROL[:])*rhoCONTROL*10**-3  # L-1
   Ni_CONTROL=getvar(CONTROL,"QNICE",timeidx=ind) # kg-1
   NICE_CONTROL[ii,:]=np.squeeze(Ni_CONTROL[:])*rhoCONTROL  # m-3
   Qi_prov_CONTROL=getvar(CONTROL,"QICE",timeidx=ind) #kg/kg
   QICE_CONTROL[ii,:]=np.squeeze(Qi_prov_CONTROL[:])*rhoCONTROL  #kg/m3
   Ng_CONTROL=getvar(CONTROL,"QNGRAUPEL",timeidx=ind) # kg-1
   NGRAUPEL_CONTROL[ii,:]=np.squeeze(Ng_CONTROL[:])*rhoCONTROL  # m-3
   Qg_prov_CONTROL=getvar(CONTROL,"QGRAUP",timeidx=ind) #kg/kg
   QGRAUPEL_CONTROL[ii,:]=np.squeeze(Qg_prov_CONTROL[:])*rhoCONTROL  #kg/m3
   Ns_CONTROL=getvar(CONTROL,"QNSNOW",timeidx=ind) # kg-1
   NSNOW_CONTROL[ii,:]=np.squeeze(Ns_CONTROL[:])*rhoCONTROL  # m-3
   Qs_prov_CONTROL=getvar(CONTROL,"QSNOW",timeidx=ind) #kg/kg
   QSNOW_CONTROL[ii,:]=np.squeeze(Qs_prov_CONTROL[:])*rhoCONTROL  #kg/m3
   Nr_CONTROL=getvar(CONTROL,"QNRAIN",timeidx=ind) # kg-1
   NRAIN_CONTROL[ii,:]=np.squeeze(Nr_CONTROL[:])*rhoCONTROL  # m-3
   Qr_CONTROL=getvar(CONTROL,"QRAIN",timeidx=ind) #kg/kg
   QRAIN_CONTROL[ii,:]=np.squeeze(Qr_CONTROL[:])*rhoCONTROL  #kg/m3
   Qc_CONTROL=getvar(CONTROL,"QCLOUD",timeidx=ind) #kg/kg
   QCLOUD_CONTROL[ii,:]=np.squeeze(Qc_CONTROL[:])*rhoCONTROL  #kg/m3
         
   for i in range(nz) :
       if QICE_CONTROL[ii,i] > 0 :
           lambda_CONTROL[ii,i]=((pi*pice*NICE_CONTROL[ii,i] / QICE_CONTROL[ii,i])**(1/3)) *10**(-6) # um-1
           if lambda_CONTROL[ii,i] < LAMIMIN:
               lambda_CONTROL[ii,i] = LAMIMIN
           elif lambda_CONTROL[ii,i] > LAMIMAX:
               lambda_CONTROL[ii,i] = LAMIMAX     
           N0_CONTROL[ii,i]=NICE_CONTROL[ii,i]*lambda_CONTROL[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambda_CONTROL[ii,i]= np.nan
           N0_CONTROL[ii,i]= np.nan

   for i in range(nz) :
       if QSNOW_CONTROL[ii,i] > 0 :
           lambdaS_CONTROL[ii,i]=((pi*psnow*NSNOW_CONTROL[ii,i] / QSNOW_CONTROL[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaS_CONTROL[ii,i] < LAMSMIN:
               lambdaS_CONTROL[ii,i] = LAMSMIN
           elif lambdaS_CONTROL[ii,i] > LAMSMAX:
               lambdaS_CONTROL[ii,i] = LAMSMAX
           N0S_CONTROL[ii,i]=NSNOW_CONTROL[ii,i]*lambdaS_CONTROL[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambdaS_CONTROL[ii,i]= np.nan
           N0S_CONTROL[ii,i]= np.nan
         
   for i in range(nz) :
       if QGRAUPEL_CONTROL[ii,i] > 0 :
           lambdaG_CONTROL[ii,i]=((pi*pgraupel*NGRAUPEL_CONTROL[ii,i] / QGRAUPEL_CONTROL[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaG_CONTROL[ii,i] < LAMGMIN:
               lambdaG_CONTROL[ii,i] = LAMGMIN
           elif lambdaG_CONTROL[ii,i] > LAMGMAX:
               lambdaG_CONTROL[ii,i] = LAMGMAX
           N0G_CONTROL[ii,i]=NGRAUPEL_CONTROL[ii,i]*lambdaG_CONTROL[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambdaG_CONTROL[ii,i]= np.nan
           N0G_CONTROL[ii,i]= np.nan
         
   for i in range(nz) :
       if QRAIN_CONTROL[ii,i] > 0 :
           lambdaR_CONTROL[ii,i]=((pi*prain*NRAIN_CONTROL[ii,i] / QRAIN_CONTROL[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaR_CONTROL[ii,i] < LAMRMIN:
               lambdaR_CONTROL[ii,i] = LAMRMIN
           elif lambdaR_CONTROL[ii,i] > LAMRMAX:
               lambdaR_CONTROL[ii,i] = LAMRMAX
           N0R_CONTROL[ii,i]=NRAIN_CONTROL[ii,i]*lambdaR_CONTROL[ii,i]*10**(-6) #cm-3 um-1
       else :
            lambdaR_CONTROL[ii,i]= np.nan
            N0R_CONTROL[ii,i]= np.nan
         
   for i in range(nz) :
       if QCLOUD_CONTROL[ii,i] > 0 :
           lambdaC_CONTROL[ii,i]=( (alpha*NDCNST*10**(6)*gamma(miu+4) / (QCLOUD_CONTROL[ii,i]*gamma(miu+1)))**(1/3) ) *10**(-6) # um-1
           if lambdaC_CONTROL[ii,i] < LAMCMIN:
               lambdaC_CONTROL[ii,i] = LAMCMIN
           elif lambdaC_CONTROL[ii,i] > LAMCMAX:
               lambdaC_CONTROL[ii,i] = LAMCMAX
           N0C_CONTROL[ii,i] = (NDCNST*10**(6)*lambdaC_CONTROL[ii,i]**(miu+1) / gamma(miu+1)) *10**(-6) #cm-3 um-1   
       else :
           lambdaC_CONTROL[ii,i]= np.nan
           N0C_CONTROL[ii,i]= np.nanPRINT

   rhoDEMOTT=presDEMOTT[ii,:]/RA/tvDEMOTT
   lwc_provDEMOTT=getvar(DEMOTT,"QCLOUD",timeidx=ind)+getvar(DEMOTT,"QRAIN",timeidx=ind) ##LWC = Qcloud + Qrain
   iwc_provDEMOTT=getvar(DEMOTT,"QICE",timeidx=ind)+getvar(DEMOTT,"QSNOW",timeidx=ind)+getvar(DEMOTT,"QGRAUP",timeidx=ind)
   lwcDEMOTT[ii,:]=np.squeeze(lwc_provDEMOTT[:])*rhoDEMOTT*10**3
   iwcDEMOTT[ii,:]=np.squeeze(iwc_provDEMOTT[:])*rhoDEMOTT*10**3
   ICNC_provDEMOTT=getvar(DEMOTT,"QNICE",timeidx=ind)+getvar(DEMOTT,"QNSNOW",timeidx=ind)+getvar(DEMOTT,"QNGRAUPEL",timeidx=ind)
   ICNCDEMOTT[ii,:]=np.squeeze(ICNC_provDEMOTT[:])*rhoDEMOTT*10**-3  # L-1
   Ni_provDEMOTT=getvar(DEMOTT,"QNICE",timeidx=ind) # kg-1
   NICE_DEMOTT[ii,:]=np.squeeze(Ni_provDEMOTT[:])*rhoDEMOTT  # m-3
   Qi_provDEMOTT=getvar(DEMOTT,"QICE",timeidx=ind) #kg/kg
   QICE_DEMOTT[ii,:]=np.squeeze(Qi_provDEMOTT[:])*rhoDEMOTT  #kg/m3
   Ng_provDEMOTT=getvar(DEMOTT,"QNGRAUPEL",timeidx=ind) # kg-1
   NGRAUPEL_DEMOTT[ii,:]=np.squeeze(Ng_provDEMOTT[:])*rhoDEMOTT  # m-3
   Qg_provDEMOTT=getvar(DEMOTT,"QGRAUP",timeidx=ind) #kg/kg
   QGRAUPEL_DEMOTT[ii,:]=np.squeeze(Qg_provDEMOTT[:])*rhoDEMOTT  #kg/m3
   Ns_provDEMOTT=getvar(DEMOTT,"QNSNOW",timeidx=ind) # kg-1
   NSNOW_DEMOTT[ii,:]=np.squeeze(Ns_provDEMOTT[:])*rhoDEMOTT  # m-3
   Qs_provDEMOTT=getvar(DEMOTT,"QSNOW",timeidx=ind) #kg/kg
   QSNOW_DEMOTT[ii,:]=np.squeeze(Qs_provDEMOTT[:])*rhoDEMOTT  #kg/m3
   Nr_provDEMOTT=getvar(DEMOTT,"QNRAIN",timeidx=ind) # kg-1
   NRAIN_DEMOTT[ii,:]=np.squeeze(Nr_provDEMOTT[:])*rhoDEMOTT  # m-3
   Qr_DEMOTT=getvar(DEMOTT,"QRAIN",timeidx=ind) #kg/kg
   QRAIN_DEMOTT[ii,:]=np.squeeze(Qr_DEMOTT[:])*rhoDEMOTT  #kg/m3
   Qc_DEMOTT=getvar(DEMOTT,"QCLOUD",timeidx=ind) #kg/kg
   QCLOUD_DEMOTT[ii,:]=np.squeeze(Qc_DEMOTT[:])*rhoDEMOTT #kg/m3
   
   
   for i in range(nz) :
       if QICE_DEMOTT[ii,i] > 0 :
           lambda_DEMOTT[ii,i]=((pi*pice*NICE_DEMOTT[ii,i] / QICE_DEMOTT[ii,i])**(1/3)) *10**(-6) # um-1
           if lambda_DEMOTT[ii,i] < LAMIMIN:
               lambda_DEMOTT[ii,i] = LAMIMIN
           elif lambda_DEMOTT[ii,i] > LAMIMAX:
               lambda_DEMOTT[ii,i] = LAMIMAX     
           N0_DEMOTT[ii,i]=NICE_DEMOTT[ii,i]*lambda_DEMOTT[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambda_DEMOTT[ii,i]= np.nan
           N0_DEMOTT[ii,i]= np.nan
         
   for i in range(nz) :
       if QSNOW_DEMOTT[ii,i] > 0 :
           lambdaS_DEMOTT[ii,i]=((pi*psnow*NSNOW_DEMOTT[ii,i] / QSNOW_DEMOTT[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaS_DEMOTT[ii,i] < LAMSMIN:
               lambdaS_DEMOTT[ii,i] = LAMSMIN
           elif lambdaS_DEMOTT[ii,i] > LAMSMAX:
               lambdaS_DEMOTT[ii,i] = LAMSMAX
           N0S_DEMOTT[ii,i]=NSNOW_DEMOTT[ii,i]*lambdaS_DEMOTT[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambdaS_DEMOTT[ii,i]= np.nan
           N0S_DEMOTT[ii,i]= np.nan

   for i in range(nz) :
       if QGRAUPEL_DEMOTT[ii,i] > 0 :
           lambdaG_DEMOTT[ii,i]=((pi*pgraupel*NGRAUPEL_DEMOTT[ii,i] / QGRAUPEL_DEMOTT[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaG_DEMOTT[ii,i] < LAMGMIN:
               lambdaG_DEMOTT[ii,i] = LAMGMIN
           elif lambdaG_DEMOTT[ii,i] > LAMGMAX:
               lambdaG_DEMOTT[ii,i] = LAMGMAX
           N0G_DEMOTT[ii,i]=NGRAUPEL_DEMOTT[ii,i]*lambdaG_DEMOTT[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambdaG_DEMOTT[ii,i]= np.nan
           N0G_DEMOTT[ii,i]= np.nan

   for i in range(nz) :
       if QRAIN_DEMOTT[ii,i] > 0 :
           lambdaR_DEMOTT[ii,i]=((pi*prain*NRAIN_DEMOTT[ii,i] / QRAIN_DEMOTT[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaR_DEMOTT[ii,i] < LAMRMIN:
               lambdaR_DEMOTT[ii,i] = LAMRMIN
           elif lambdaR_DEMOTT[ii,i] > LAMRMAX:
               lambdaR_DEMOTT[ii,i] = LAMRMAX
           N0R_DEMOTT[ii,i]=NRAIN_DEMOTT[ii,i]*lambdaR_DEMOTT[ii,i]*10**(-6) #cm-3 um-1
       else :
            lambdaR_DEMOTT[ii,i]= np.nan
            N0R_DEMOTT[ii,i]= np.nan
         
   for i in range(nz) :
       if QCLOUD_DEMOTT[ii,i] > 0 :
           lambdaC_DEMOTT[ii,i]=( (alpha*NDCNST*10**(6)*gamma(miu+4) / (QCLOUD_DEMOTT[ii,i]*gamma(miu+1)))**(1/3) ) *10**(-6) # um-1
           if lambdaC_DEMOTT[ii,i] < LAMCMIN:
               lambdaC_DEMOTT[ii,i] = LAMCMIN
           elif lambdaC_DEMOTT[ii,i] > LAMCMAX:
               lambdaC_DEMOTT[ii,i] = LAMCMAX
           N0C_DEMOTT[ii,i] = (NDCNST*10**(6)*lambdaC_DEMOTT[ii,i]**(miu+1) / gamma(miu+1)) *10**(-6) #cm-3 um-1   
       else :
           lambdaC_DEMOTT[ii,i]= np.nan
           N0C_DEMOTT[ii,i]= np.nan
         

   rhoALLSIP=presALLSIP[ii,:]/RA/tvALLSIP
   lwc_provALLSIP=getvar(ALLSIP,"QCLOUD",timeidx=ind)+getvar(ALLSIP,"QRAIN",timeidx=ind) ##LWC = Qcloud + Qrain
   iwc_provALLSIP=getvar(ALLSIP,"QICE",timeidx=ind)+getvar(ALLSIP,"QSNOW",timeidx=ind)+getvar(ALLSIP,"QGRAUP",timeidx=ind)
   lwcALLSIP[ii,:]=np.squeeze(lwc_provALLSIP[:])*rhoALLSIP*10**3
   iwcALLSIP[ii,:]=np.squeeze(iwc_provALLSIP[:])*rhoALLSIP*10**3
   ICNC_provALLSIP=getvar(ALLSIP,"QNICE",timeidx=ind)+getvar(ALLSIP,"QNSNOW",timeidx=ind)+getvar(ALLSIP,"QNGRAUPEL",timeidx=ind)
   ICNCALLSIP[ii,:]=np.squeeze(ICNC_provALLSIP[:])*rhoALLSIP*10**-3  # L-1
   Ni_provALLSIP=getvar(ALLSIP,"QNICE",timeidx=ind) # kg-1
   NICE_ALLSIP[ii,:]=np.squeeze(Ni_provALLSIP[:])*rhoALLSIP  # m-3
   Qi_provALLSIP=getvar(ALLSIP,"QICE",timeidx=ind) #kg/kg
   QICE_ALLSIP[ii,:]=np.squeeze(Qi_provALLSIP[:])*rhoALLSIP  #kg/m3
   Ng_provALLSIP=getvar(ALLSIP,"QNGRAUPEL",timeidx=ind) # kg-1
   NGRAUPEL_ALLSIP[ii,:]=np.squeeze(Ng_provALLSIP[:])*rhoALLSIP  # m-3
   Qg_provALLSIP=getvar(ALLSIP,"QGRAUP",timeidx=ind) #kg/kg
   QGRAUPEL_ALLSIP[ii,:]=np.squeeze(Qg_provALLSIP[:])*rhoALLSIP  #kg/m3
   Ns_provALLSIP=getvar(ALLSIP,"QNSNOW",timeidx=ind) # kg-1
   NSNOW_ALLSIP[ii,:]=np.squeeze(Ns_provALLSIP[:])*rhoALLSIP  # m-3
   Qs_provALLSIP=getvar(ALLSIP,"QSNOW",timeidx=ind) #kg/kg
   QSNOW_ALLSIP[ii,:]=np.squeeze(Qs_provALLSIP[:])*rhoALLSIP  #kg/m3
   Nr_provALLSIP=getvar(ALLSIP,"QNRAIN",timeidx=ind) # kg-1
   NRAIN_ALLSIP[ii,:]=np.squeeze(Nr_provALLSIP[:])*rhoALLSIP  # m-3
   Qr_ALLSIP=getvar(ALLSIP,"QRAIN",timeidx=ind) #kg/kg
   QRAIN_ALLSIP[ii,:]=np.squeeze(Qr_ALLSIP[:])*rhoALLSIP  #kg/m3
   Qc_ALLSIP=getvar(ALLSIP,"QCLOUD",timeidx=ind) #kg/kg
   QCLOUD_ALLSIP[ii,:]=np.squeeze(Qc_ALLSIP[:])*rhoALLSIP  #kg/m3
   

   for i in range(nz) :
       if QICE_ALLSIP[ii,i] > 0 :
           lambda_ALLSIP[ii,i]=((pi*pice*NICE_ALLSIP[ii,i] / QICE_ALLSIP[ii,i])**(1/3)) *10**(-6) # um-1
           if lambda_ALLSIP[ii,i] < LAMIMIN:
               lambda_ALLSIP[ii,i] = LAMIMIN
           elif lambda_ALLSIP[ii,i] > LAMIMAX:
               lambda_ALLSIP[ii,i] = LAMIMAX     
           N0_ALLSIP[ii,i]=NICE_ALLSIP[ii,i]*lambda_ALLSIP[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambda_ALLSIP[ii,i]= np.nan
           N0_ALLSIP[ii,i]= np.nan

   for i in range(nz) :
       if QSNOW_ALLSIP[ii,i] > 0 :
           lambdaS_ALLSIP[ii,i]=((pi*psnow*NSNOW_ALLSIP[ii,i] / QSNOW_ALLSIP[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaS_ALLSIP[ii,i] < LAMSMIN:
               lambdaS_ALLSIP[ii,i] = LAMSMIN
           elif lambdaS_ALLSIP[ii,i] > LAMSMAX:
               lambdaS_ALLSIP[ii,i] = LAMSMAX
           N0S_ALLSIP[ii,i]=NSNOW_ALLSIP[ii,i]*lambdaS_ALLSIP[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambdaS_ALLSIP[ii,i]= np.nan
           N0S_ALLSIP[ii,i]= np.nan
         
   for i in range(nz) :
       if QGRAUPEL_ALLSIP[ii,i] > 0 :
           lambdaG_ALLSIP[ii,i]=((pi*pgraupel*NGRAUPEL_ALLSIP[ii,i] / QGRAUPEL_ALLSIP[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaG_ALLSIP[ii,i] < LAMGMIN:
               lambdaG_ALLSIP[ii,i] = LAMGMIN
           elif lambdaG_ALLSIP[ii,i] > LAMGMAX:
               lambdaG_ALLSIP[ii,i] = LAMGMAX
           N0G_ALLSIP[ii,i]=NGRAUPEL_ALLSIP[ii,i]*lambdaG_ALLSIP[ii,i]*10**(-6) #cm-3 um-1
       else :
           lambdaG_ALLSIP[ii,i]= np.nan
           N0G_ALLSIP[ii,i]= np.nan

   for i in range(nz) :
       if QRAIN_ALLSIP[ii,i] > 0 :
           lambdaR_ALLSIP[ii,i]=((pi*prain*NRAIN_ALLSIP[ii,i] / QRAIN_ALLSIP[ii,i])**(1/3)) *10**(-6) # um-1
           if lambdaR_ALLSIP[ii,i] < LAMRMIN:
               lambdaR_ALLSIP[ii,i] = LAMRMIN
           elif lambdaR_ALLSIP[ii,i] > LAMRMAX:
               lambdaR_ALLSIP[ii,i] = LAMRMAX
           N0R_ALLSIP[ii,i]=NRAIN_ALLSIP[ii,i]*lambdaR_ALLSIP[ii,i]*10**(-6) #cm-3 um-1
       else :
            lambdaR_ALLSIP[ii,i]= np.nan
            N0R_ALLSIP[ii,i]= np.nan
         
   for i in range(nz) :
       if QCLOUD_ALLSIP[ii,i] > 0 :
           lambdaC_ALLSIP[ii,i]=( (alpha*NDCNST*10**(6)*gamma(miu+4) / (QCLOUD_ALLSIP[ii,i]*gamma(miu+1)))**(1/3) ) *10**(-6) # um-1
           if lambdaC_ALLSIP[ii,i] < LAMCMIN:
               lambdaC_ALLSIP[ii,i] = LAMCMIN
           elif lambdaC_ALLSIP[ii,i] > LAMCMAX:
               lambdaC_ALLSIP[ii,i] = LAMCMAX
           N0C_ALLSIP[ii,i] = (NDCNST*10**(6)*lambdaC_ALLSIP[ii,i]**(miu+1) / gamma(miu+1)) *10**(-6) #cm-3 um-1   
       else :
           lambdaC_ALLSIP[ii,i]= np.nan
           N0C_ALLSIP[ii,i]= np.nan
           
   ii=ii+1


### INSTANTANEOUS OUTPUTS ###
# Convert specific time to numpy.datetime64
tstmp1 = np.datetime64('2021-12-18T09:20') #03:50 for 1st period # 09:20 for the 2nd period
idt = np.searchsorted(wrf_time, tstmp1, side="right") - 1

lev = 23 #23 = 2.26 km  ZZmiddle[24]/1000 11 for the first period

lambdaCONTROL = lambda_CONTROL[idt,lev]  #27/01/2014 00:00-06:00  ==  96:109
lambdaDEMOTT= (lambda_DEMOTT[idt,lev])
lambdaALLSIP= (lambda_ALLSIP[idt,lev])

lambdaGCONTROL = (lambdaG_CONTROL[idt,lev])
lambdaGDEMOTT= (lambdaG_DEMOTT[idt,lev])
lambdaGALLSIP= (lambdaG_ALLSIP[idt,lev])

lambdaSCONTROL = (lambdaS_CONTROL[idt,lev])
lambdaSDEMOTT= (lambdaS_DEMOTT[idt,lev])
lambdaSALLSIP= (lambdaS_ALLSIP[idt,lev])

lambdaCCONTROL = (lambdaC_CONTROL[idt,lev])
lambdaCDEMOTT= (lambdaC_DEMOTT[idt,lev])
lambdaCALLSIP= (lambdaC_ALLSIP[idt,lev])

lambdaRCONTROL = (lambdaR_CONTROL[idt,lev])
lambdaRDEMOTT= (lambdaR_DEMOTT[idt,lev])
lambdaRALLSIP= (lambdaR_ALLSIP[idt,lev])

N0CONTROL = (N0_CONTROL[idt,lev])
N0DEMOTT = (N0_DEMOTT[idt,lev])
N0ALLSIP = (N0_ALLSIP[idt,lev])

N0SCONTROL = (N0S_CONTROL[idt,lev])
N0SDEMOTT = (N0S_DEMOTT[idt,lev])
N0SALLSIP = (N0S_ALLSIP[idt,lev])

N0GCONTROL = (N0G_CONTROL[idt,lev])
N0GDEMOTT = (N0G_DEMOTT[idt,lev])
N0GALLSIP = (N0G_ALLSIP[idt,lev])

N0RCONTROL = (N0R_CONTROL[idt,lev])
N0RDEMOTT = (N0R_DEMOTT[idt,lev])
N0RALLSIP = (N0R_ALLSIP[idt,lev])

N0CCONTROL = (N0C_CONTROL[idt,lev])
N0CDEMOTT = (N0C_DEMOTT[idt,lev])
N0CALLSIP = (N0C_ALLSIP[idt,lev])

dndDpCONTROL = np.zeros(len(Dp1))
dndDpDEMOTT = np.zeros(len(Dp1))
dndDpALLSIP = np.zeros(len(Dp1))
dndDpALLSIP_HM2 = np.zeros(len(Dp1))

dndDpSCONTROL = np.zeros(len(Dp2))
dndDpSDEMOTT = np.zeros(len(Dp2))
dndDpSALLSIP = np.zeros(len(Dp2))
dndDpSALLSIP_HM2 = np.zeros(len(Dp2))

dndDpGCONTROL = np.zeros(len(Dp3))
dndDpGDEMOTT = np.zeros(len(Dp3))
dndDpGALLSIP = np.zeros(len(Dp3))
dndDpGALLSIP_HM2 = np.zeros(len(Dp3))

dndDpRCONTROL = np.zeros(len(Dp4))
dndDpRDEMOTT = np.zeros(len(Dp4))
dndDpRALLSIP = np.zeros(len(Dp4))
dndDpRALLSIP_HM2 = np.zeros(len(Dp4))

dndDpCCONTROL = np.zeros(len(Dp5))
dndDpCDEMOTT = np.zeros(len(Dp5))
dndDpCALLSIP = np.zeros(len(Dp5))
dndDpCALLSIP_HM2 = np.zeros(len(Dp5))

dndlogDpCONTROL = np.zeros(len(Dp1))
dndlogDpDEMOTT = np.zeros(len(Dp1))
dndlogDpALLSIP = np.zeros(len(Dp1))
dndlogDpALLSIP_HM2 = np.zeros(len(Dp1))

dndlogDpSCONTROL = np.zeros(len(Dp2))
dndlogDpSDEMOTT = np.zeros(len(Dp2))
dndlogDpSALLSIP = np.zeros(len(Dp2))
dndlogDpSALLSIP_HM2 = np.zeros(len(Dp2))

dndlogDpGCONTROL = np.zeros(len(Dp3))
dndlogDpGDEMOTT = np.zeros(len(Dp3))
dndlogDpGALLSIP = np.zeros(len(Dp3))
dndlogDpGALLSIP_HM2 = np.zeros(len(Dp3))

dndlogDpRCONTROL = np.zeros(len(Dp4))
dndlogDpRDEMOTT = np.zeros(len(Dp4))
dndlogDpRALLSIP = np.zeros(len(Dp4))
dndlogDpRALLSIP_HM2 = np.zeros(len(Dp4))

dndlogDpCCONTROL = np.zeros(len(Dp5))
dndlogDpCDEMOTT = np.zeros(len(Dp5))
dndlogDpCALLSIP = np.zeros(len(Dp5))
dndlogDpCALLSIP_HM2 = np.zeros(len(Dp5))


for k in range(len(Dp1)):
   dndDpCONTROL[k] = N0CONTROL * math.exp(-lambdaCONTROL*Dp1[k])
   dndlogDpCONTROL[k] = dndDpCONTROL[k] * Dp1[k] * np.log(10)
   dndDpDEMOTT[k] = N0DEMOTT * math.exp(-lambdaDEMOTT*Dp1[k])
   dndlogDpDEMOTT[k] = dndDpDEMOTT[k] * Dp1[k] * np.log(10)
   dndDpALLSIP[k] = N0ALLSIP * math.exp(-lambdaALLSIP*Dp1[k])
   dndlogDpALLSIP[k] = dndDpALLSIP[k] * Dp1[k] * np.log(10)


for k in range(len(Dp2)):
   dndDpSCONTROL[k] = N0SCONTROL * math.exp(-lambdaSCONTROL*Dp2[k])
   dndlogDpSCONTROL[k] = dndDpSCONTROL[k] * Dp2[k] * np.log(10)  
   dndDpSDEMOTT[k] = N0SDEMOTT * math.exp(-lambdaSDEMOTT*Dp2[k])
   dndlogDpSDEMOTT[k] = dndDpSDEMOTT[k] * Dp2[k] * np.log(10)
   dndDpSALLSIP[k] = N0SALLSIP * math.exp(-lambdaSALLSIP*Dp2[k])
   dndlogDpSALLSIP[k] = dndDpSALLSIP[k] * Dp2[k] * np.log(10)

for k in range(len(Dp3)):
   dndDpGCONTROL[k] = N0GCONTROL * math.exp(-lambdaGCONTROL*Dp3[k])
   dndlogDpGCONTROL[k] = dndDpGCONTROL[k] * Dp3[k] * np.log(10)  
   dndDpGDEMOTT[k] = N0GDEMOTT * math.exp(-lambdaGDEMOTT*Dp3[k])
   dndlogDpGDEMOTT[k] = dndDpGDEMOTT[k] * Dp3[k] * np.log(10)
   dndDpGALLSIP[k] = N0GALLSIP * math.exp(-lambdaGALLSIP*Dp3[k])
   dndlogDpGALLSIP[k] = dndDpGALLSIP[k] * Dp3[k] * np.log(10)
   
for k in range(len(Dp4)):
   dndDpRCONTROL[k] = N0RCONTROL * math.exp(-lambdaRCONTROL*Dp4[k])
   dndlogDpRCONTROL[k] = dndDpRCONTROL[k] * Dp4[k] * np.log(10)  
   dndDpRDEMOTT[k] = N0RDEMOTT * math.exp(-lambdaRDEMOTT*Dp4[k])
   dndlogDpRDEMOTT[k] = dndDpRDEMOTT[k] * Dp4[k] * np.log(10)
   dndDpRALLSIP[k] = N0RALLSIP * math.exp(-lambdaRALLSIP*Dp4[k])
   dndlogDpRALLSIP[k] = dndDpRALLSIP[k] * Dp4[k] * np.log(10)

   
for k in range(len(Dp5)):
   dndDpCCONTROL[k] = N0CCONTROL * math.exp(-lambdaCCONTROL*Dp5[k])
   dndlogDpCCONTROL[k] = dndDpCCONTROL[k] * Dp5[k] * np.log(10)  
   dndDpCDEMOTT[k] = N0GDEMOTT * math.exp(-lambdaCDEMOTT*Dp5[k])
   dndlogDpCDEMOTT[k] = dndDpCDEMOTT[k] * Dp5[k] * np.log(10)
   dndDpCALLSIP[k] = N0CALLSIP * math.exp(-lambdaCALLSIP*Dp5[k])
   dndlogDpCALLSIP[k] = dndDpCALLSIP[k] * Dp5[k] * np.log(10)


iNiCONTROL = np.where(dndlogDpCONTROL == np.nanmax(dndlogDpCONTROL))
iNiALLSIP = np.where(dndlogDpALLSIP == np.nanmax(dndlogDpALLSIP))
iNiALLSIP_HM2 = np.where(dndlogDpALLSIP_HM2 == np.nanmax(dndlogDpALLSIP_HM2))
iNiDEMOTT = np.where(dndlogDpDEMOTT == np.nanmax(dndlogDpDEMOTT))

iNsCONTROL = np.where(dndlogDpSCONTROL == np.nanmax(dndlogDpSCONTROL))
iNsALLSIP = np.where(dndlogDpSALLSIP == np.nanmax(dndlogDpSALLSIP))
iNsALLSIP_HM2 = np.where(dndlogDpSALLSIP_HM2 == np.nanmax(dndlogDpSALLSIP_HM2))
iNsDEMOTT = np.where(dndlogDpSDEMOTT == np.nanmax(dndlogDpSDEMOTT))

iNgCONTROL = np.where(dndlogDpGCONTROL == np.nanmax(dndlogDpGCONTROL))
iNgALLSIP = np.where(dndlogDpGALLSIP == np.nanmax(dndlogDpGALLSIP))
iNgALLSIP_HM2 = np.where(dndlogDpGALLSIP_HM2 == np.nanmax(dndlogDpGALLSIP_HM2))
iNgDEMOTT = np.where(dndlogDpGDEMOTT == np.nanmax(dndlogDpGDEMOTT))

iNrCONTROL = np.where(dndlogDpRCONTROL == np.nanmax(dndlogDpRCONTROL))
iNrALLSIP = np.where(dndlogDpRALLSIP == np.nanmax(dndlogDpRALLSIP))
iNrALLSIP_HM2 = np.where(dndlogDpRALLSIP_HM2 == np.nanmax(dndlogDpRALLSIP_HM2))
iNrDEMOTT = np.where(dndlogDpRDEMOTT == np.nanmax(dndlogDpRDEMOTT))

iNcCONTROL = np.where(dndlogDpCCONTROL == np.nanmax(dndlogDpCCONTROL))
iNcALLSIP = np.where(dndlogDpCALLSIP == np.nanmax(dndlogDpCALLSIP))
iNcALLSIP_HM2 = np.where(dndlogDpCALLSIP_HM2 == np.nanmax(dndlogDpCALLSIP_HM2))
iNcDEMOTT = np.where(dndlogDpCDEMOTT == np.nanmax(dndlogDpCDEMOTT))


### INSTANTANEOUS OUTPUTS ###
lev2 = 6 #6 = 0.54 km  #ZZmiddle[6]/1000  3 for the 1st period

lambdaCONTROL_2 = lambda_CONTROL[idt,lev2]  #27/01/2014 00:00-06:00  ==  96:109
lambdaDEMOTT_2 = (lambda_DEMOTT[idt,lev2])
lambdaALLSIP_2 = (lambda_ALLSIP[idt,lev2])

lambdaGCONTROL_2 = (lambdaG_CONTROL[idt,lev2])
lambdaGDEMOTT_2 = (lambdaG_DEMOTT[idt,lev2])
lambdaGALLSIP_2 = (lambdaG_ALLSIP[idt,lev2])

lambdaSCONTROL_2 = (lambdaS_CONTROL[idt,lev2])
lambdaSDEMOTT_2 = (lambdaS_DEMOTT[idt,lev2])
lambdaSALLSIP_2 = (lambdaS_ALLSIP[idt,lev2])

lambdaCCONTROL_2 = (lambdaC_CONTROL[idt,lev2])
lambdaCDEMOTT_2 = (lambdaC_DEMOTT[idt,lev2])
lambdaCALLSIP_2 = (lambdaC_ALLSIP[idt,lev2])

lambdaRCONTROL_2 = (lambdaR_CONTROL[idt,lev2])
lambdaRDEMOTT_2 = (lambdaR_DEMOTT[idt,lev2])
lambdaRALLSIP_2 = (lambdaR_ALLSIP[idt,lev2])

N0CONTROL_2 = (N0_CONTROL[idt,lev2])
N0DEMOTT_2 = (N0_DEMOTT[idt,lev2])
N0ALLSIP_2 = (N0_ALLSIP[idt,lev2])

N0SCONTROL_2 = (N0S_CONTROL[idt,lev2])
N0SDEMOTT_2 = (N0S_DEMOTT[idt,lev2])
N0SALLSIP_2 = (N0S_ALLSIP[idt,lev2])

N0GCONTROL_2 = (N0G_CONTROL[idt,lev2])
N0GDEMOTT_2 = (N0G_DEMOTT[idt,lev2])
N0GALLSIP_2 = (N0G_ALLSIP[idt,lev2])

N0RCONTROL_2 = (N0R_CONTROL[idt,lev2])
N0RDEMOTT_2 = (N0R_DEMOTT[idt,lev2])
N0RALLSIP_2 = (N0R_ALLSIP[idt,lev2])

N0CCONTROL_2 = (N0C_CONTROL[idt,lev2])
N0CDEMOTT_2 = (N0C_DEMOTT[idt,lev2])
N0CALLSIP_2 = (N0C_ALLSIP[idt,lev2])

dndDpCONTROL_2 = np.zeros(len(Dp1))
dndDpDEMOTT_2 = np.zeros(len(Dp1))
dndDpALLSIP_2 = np.zeros(len(Dp1))
dndDpALLSIP_HM2_2 = np.zeros(len(Dp1))

dndDpSCONTROL_2 = np.zeros(len(Dp2))
dndDpSDEMOTT_2 = np.zeros(len(Dp2))
dndDpSALLSIP_2 = np.zeros(len(Dp2))
dndDpSALLSIP_HM2_2 = np.zeros(len(Dp2))

dndDpGCONTROL_2 = np.zeros(len(Dp3))
dndDpGDEMOTT_2 = np.zeros(len(Dp3))
dndDpGALLSIP_2 = np.zeros(len(Dp3))
dndDpGALLSIP_HM2_2 = np.zeros(len(Dp3))

dndDpRCONTROL_2 = np.zeros(len(Dp4))
dndDpRDEMOTT_2 = np.zeros(len(Dp4))
dndDpRALLSIP_2 = np.zeros(len(Dp4))
dndDpRALLSIP_HM2_2 = np.zeros(len(Dp4))

dndDpCCONTROL_2 = np.zeros(len(Dp5))
dndDpCDEMOTT_2 = np.zeros(len(Dp5))
dndDpCALLSIP_2 = np.zeros(len(Dp5))
dndDpCALLSIP_HM2_2 = np.zeros(len(Dp5))

dndlogDpCONTROL_2 = np.zeros(len(Dp1))
dndlogDpDEMOTT_2 = np.zeros(len(Dp1))
dndlogDpALLSIP_2 = np.zeros(len(Dp1))
dndlogDpALLSIP_HM2_2 = np.zeros(len(Dp1))

dndlogDpSCONTROL_2 = np.zeros(len(Dp2))
dndlogDpSDEMOTT_2 = np.zeros(len(Dp2))
dndlogDpSALLSIP_2 = np.zeros(len(Dp2))
dndlogDpSALLSIP_HM2_2 = np.zeros(len(Dp2))

dndlogDpGCONTROL_2 = np.zeros(len(Dp3))
dndlogDpGDEMOTT_2 = np.zeros(len(Dp3))
dndlogDpGALLSIP_2 = np.zeros(len(Dp3))
dndlogDpGALLSIP_HM2_2 = np.zeros(len(Dp3))

dndlogDpRCONTROL_2 = np.zeros(len(Dp4))
dndlogDpRDEMOTT_2 = np.zeros(len(Dp4))
dndlogDpRALLSIP_2 = np.zeros(len(Dp4))
dndlogDpRALLSIP_HM2_2 = np.zeros(len(Dp4))

dndlogDpCCONTROL_2 = np.zeros(len(Dp5))
dndlogDpCDEMOTT_2 = np.zeros(len(Dp5))
dndlogDpCALLSIP_2 = np.zeros(len(Dp5))
dndlogDpCALLSIP_HM2_2 = np.zeros(len(Dp5))


for k in range(len(Dp1)):
   dndDpCONTROL_2[k] = N0CONTROL_2 * math.exp(-lambdaCONTROL_2*Dp1[k])
   dndlogDpCONTROL_2[k] = dndDpCONTROL_2[k] * Dp1[k] * np.log(10)  
   dndDpDEMOTT_2[k] = N0DEMOTT * math.exp(-lambdaDEMOTT_2*Dp1[k])
   dndlogDpDEMOTT_2[k] = dndDpDEMOTT_2[k] * Dp1[k] * np.log(10)
   dndDpALLSIP_2[k] = N0ALLSIP_2 * math.exp(-lambdaALLSIP_2*Dp1[k])
   dndlogDpALLSIP_2[k] = dndDpALLSIP_2[k] * Dp1[k] * np.log(10)


for k in range(len(Dp2)):
   dndDpSCONTROL_2[k] = N0SCONTROL_2 * math.exp(-lambdaSCONTROL_2*Dp2[k])
   dndlogDpSCONTROL_2[k] = dndDpSCONTROL_2[k] * Dp2[k] * np.log(10)  
   dndDpSDEMOTT_2[k] = N0SDEMOTT_2 * math.exp(-lambdaSDEMOTT_2*Dp2[k])
   dndlogDpSDEMOTT_2[k] = dndDpSDEMOTT_2[k] * Dp2[k] * np.log(10)
   dndDpSALLSIP_2[k] = N0SALLSIP_2 * math.exp(-lambdaSALLSIP_2*Dp2[k])
   dndlogDpSALLSIP_2[k] = dndDpSALLSIP_2[k] * Dp2[k] * np.log(10)

for k in range(len(Dp3)):
   dndDpGCONTROL_2[k] = N0GCONTROL_2 * math.exp(-lambdaGCONTROL_2*Dp3[k])
   dndlogDpGCONTROL_2[k] = dndDpGCONTROL_2[k] * Dp3[k] * np.log(10)  
   dndDpGDEMOTT_2[k] = N0GDEMOTT_2 * math.exp(-lambdaGDEMOTT_2*Dp3[k])
   dndlogDpGDEMOTT_2[k] = dndDpGDEMOTT_2[k] * Dp3[k] * np.log(10)
   dndDpGALLSIP_2[k] = N0GALLSIP_2 * math.exp(-lambdaGALLSIP_2*Dp3[k])
   dndlogDpGALLSIP_2[k] = dndDpGALLSIP_2[k] * Dp3[k] * np.log(10)
   
for k in range(len(Dp4)):
   dndDpRCONTROL_2[k] = N0RCONTROL_2 * math.exp(-lambdaRCONTROL_2*Dp4[k])
   dndlogDpRCONTROL_2[k] = dndDpRCONTROL_2[k] * Dp4[k] * np.log(10)  
   dndDpRDEMOTT_2[k] = N0RDEMOTT_2 * math.exp(-lambdaRDEMOTT_2*Dp4[k])
   dndlogDpRDEMOTT_2[k] = dndDpRDEMOTT_2[k] * Dp4[k] * np.log(10)
   dndDpRALLSIP_2[k] = N0RALLSIP_2 * math.exp(-lambdaRALLSIP_2*Dp4[k])
   dndlogDpRALLSIP_2[k] = dndDpRALLSIP_2[k] * Dp4[k] * np.log(10)

   
for k in range(len(Dp5)):
   dndDpCCONTROL_2[k] = N0CCONTROL_2 * math.exp(-lambdaCCONTROL_2*Dp5[k])
   dndlogDpCCONTROL_2[k] = dndDpCCONTROL_2[k] * Dp5[k] * np.log(10)  
   dndDpCDEMOTT_2[k] = N0GDEMOTT_2 * math.exp(-lambdaCDEMOTT_2*Dp5[k])
   dndlogDpCDEMOTT_2[k] = dndDpCDEMOTT_2[k] * Dp5[k] * np.log(10)
   dndDpCALLSIP_2[k] = N0CALLSIP_2 * math.exp(-lambdaCALLSIP_2*Dp5[k])
   dndlogDpCALLSIP_2[k] = dndDpCALLSIP_2[k] * Dp5[k] * np.log(10)


iNiCONTROL_2 = np.where(dndlogDpCONTROL_2 == np.nanmax(dndlogDpCONTROL_2))
iNiALLSIP_2 = np.where(dndlogDpALLSIP_2 == np.nanmax(dndlogDpALLSIP_2))
iNiALLSIP_HM2_2 = np.where(dndlogDpALLSIP_HM2_2 == np.nanmax(dndlogDpALLSIP_HM2_2))
iNiDEMOTT_2 = np.where(dndlogDpDEMOTT_2 == np.nanmax(dndlogDpDEMOTT_2))

iNsCONTROL_2 = np.where(dndlogDpSCONTROL_2 == np.nanmax(dndlogDpSCONTROL_2))
iNsALLSIP_2 = np.where(dndlogDpSALLSIP_2 == np.nanmax(dndlogDpSALLSIP_2))
iNsALLSIP_HM2_2 = np.where(dndlogDpSALLSIP_HM2_2 == np.nanmax(dndlogDpSALLSIP_HM2_2))
iNsDEMOTT_2 = np.where(dndlogDpSDEMOTT_2 == np.nanmax(dndlogDpSDEMOTT_2))

iNgCONTROL_2 = np.where(dndlogDpGCONTROL_2 == np.nanmax(dndlogDpGCONTROL_2))
iNgALLSIP_2 = np.where(dndlogDpGALLSIP_2 == np.nanmax(dndlogDpGALLSIP_2))
iNgALLSIP_HM2_2 = np.where(dndlogDpGALLSIP_HM2_2 == np.nanmax(dndlogDpGALLSIP_HM2_2))
iNgDEMOTT_2 = np.where(dndlogDpGDEMOTT_2 == np.nanmax(dndlogDpGDEMOTT_2))

iNrCONTROL_2 = np.where(dndlogDpRCONTROL_2 == np.nanmax(dndlogDpRCONTROL_2))
iNrALLSIP_2 = np.where(dndlogDpRALLSIP_2 == np.nanmax(dndlogDpRALLSIP_2))
iNrALLSIP_HM2_2 = np.where(dndlogDpRALLSIP_HM2_2 == np.nanmax(dndlogDpRALLSIP_HM2_2))
iNrDEMOTT_2 = np.where(dndlogDpRDEMOTT_2 == np.nanmax(dndlogDpRDEMOTT_2))

iNcCONTROL_2 = np.where(dndlogDpCCONTROL_2 == np.nanmax(dndlogDpCCONTROL_2))
iNcALLSIP_2 = np.where(dndlogDpCALLSIP_2 == np.nanmax(dndlogDpCALLSIP_2))
iNcALLSIP_HM2_2 = np.where(dndlogDpCALLSIP_HM2_2 == np.nanmax(dndlogDpCALLSIP_HM2_2))
iNcDEMOTT_2 = np.where(dndlogDpCDEMOTT_2 == np.nanmax(dndlogDpCDEMOTT))



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
lamda = ax[0,0].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/lambdaS_ALLSIP[spin_up:end,:].T), levs, cmap = cmap, norm=colors.LogNorm())
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
lamda = ax[0,1].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/lambdaR_ALLSIP[spin_up:end,:].T), levs, cmap = cmap)
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
lamda = ax[1,0].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/lambda_ALLSIP[spin_up:end,:].T), levs, cmap = cmap, norm=colors.LogNorm())
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
lamda = ax[1,1].contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/lambdaG_ALLSIP[spin_up:end,:].T), levs, cmap = cmap, norm=colors.LogNorm())
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
lamda = ax.contourf(wrf_time[spin_up:end], ZZmiddle/1000, to_np(1/lambdaC_ALLSIP[spin_up:end,:].T), levs, cmap = cmap)
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

L11 = axs[0,0].plot(Dp1*0.001,dndlogDpCONTROL,'-k',linewidth=2.5)
L31 = axs[0,0].plot(Dp1*0.001,dndlogDpALLSIP,'-',color='blue',linewidth=2.5)
L51 = axs[0,0].plot(Dp1*0.001,dndlogDpDEMOTT,'-',color='cyan',linewidth=2.5)
axs[0,0].semilogy([Dp1[iNiCONTROL]*0.001, Dp1[iNiCONTROL]*0.001], [10**(-7), dndlogDpCONTROL[iNiCONTROL]], 'k--')
axs[0,0].semilogy([Dp1[iNiDEMOTT]*0.001, Dp1[iNiDEMOTT]*0.001], [10**(-7), dndlogDpDEMOTT[iNiDEMOTT]], '--', color='cyan')
axs[0,0].semilogy([Dp1[iNiALLSIP]*0.001, Dp1[iNiALLSIP]*0.001], [10**(-7), dndlogDpALLSIP[iNiALLSIP]], '--', color='blue')
axs[0,0].set_ylabel("d($N_{ice}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[0,0].set_xlabel("$D_{p}$ (mm)")
axs[0,0].set_xlim(0.4*0.001,3000*0.001)
axs[0,0].set_ylim(1e-7,1e-1)
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,0].text(5.5*10**(-4),0.04,'(a)', fontsize=14)

L12 = axs[0,1].plot(Dp2*0.001,dndlogDpSCONTROL,'-k',linewidth=2.5,label='CONTROL')
L52 = axs[0,1].plot(Dp2*0.001,dndlogDpSDEMOTT,'-',color='cyan',linewidth=2.5,label='DEMOTT')
L32 = axs[0,1].plot(Dp2*0.001,dndlogDpSALLSIP,'-',color='blue',linewidth=2.5,label='ALLSIP')
axs[0,1].set_ylabel("d($N_{snow}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[0,1].semilogy([Dp2[iNsCONTROL]*0.001, Dp2[iNsCONTROL]*0.001], [10**(-7), dndlogDpSCONTROL[iNsCONTROL]], 'k--')
axs[0,1].semilogy([Dp2[iNsALLSIP]*0.001, Dp2[iNsALLSIP]*0.001], [10**(-7), dndlogDpSALLSIP[iNsALLSIP]], '--', color='blue')
axs[0,1].semilogy([Dp2[iNsDEMOTT]*0.001, Dp2[iNsDEMOTT]*0.001], [10**(-7), dndlogDpSDEMOTT[iNsDEMOTT]], '--', color='cyan')
axs[0,1].set_xlabel("$D_{p}$ (mm)")
axs[0,1].set_xlim(12*0.0001,30000*0.001)
axs[0,1].set_ylim(10**(-7),1e-1)
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[0,1].text(0.0015,0.04,'(b)', fontsize=14)

L13 = axs[0,2].plot(Dp3*0.001,dndlogDpGCONTROL,'-k',linewidth=2.5,label='CONTROL')
axs[0,2].semilogy([Dp3[iNgDEMOTT]*0.001, Dp3[iNgDEMOTT]*0.001], [10**(-12), dndlogDpGDEMOTT[iNgDEMOTT]], '--', color='cyan')
L53 = axs[0,2].plot(Dp3*0.001,dndlogDpGDEMOTT,'-',color='cyan',linewidth=2.5,label='DEMOTT')
L33 = axs[0,2].plot(Dp3*0.001,dndlogDpGALLSIP,'-',color='blue',linewidth=2.5,label='ALLSIP')
axs[0,2].set_ylabel("d($N_{graupel}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[0,2].semilogy([Dp3[iNgCONTROL]*0.001, Dp3[iNgCONTROL]*0.001], [10**(-12), dndlogDpGCONTROL[iNgCONTROL]], 'k--')
axs[0,2].semilogy([Dp3[iNgALLSIP]*0.001, Dp3[iNgALLSIP]*0.001], [10**(-12), dndlogDpGALLSIP[iNgALLSIP]], '--', color='blue')
axs[0,2].set_xlabel("$D_{p}$ (mm)")
axs[0,2].set_xlim(10*0.0001,10*1000*0.001)
axs[0,2].set_ylim(10**(-9),10**(-2))
axs[0,2].set_xscale('log')
axs[0,2].set_yscale('log')
axs[0,2].text(0.0012,0.003,'(c)', fontsize=14)
axs[0,2].legend(loc = 'upper right',ncol=1)


L11 = axs[1,0].plot(Dp1*0.001,dndlogDpCONTROL_2,'-k',linewidth=2.5)
L31 = axs[1,0].plot(Dp1*0.001,dndlogDpALLSIP_2,'-',color='blue',linewidth=2.5)
L51 = axs[1,0].plot(Dp1*0.001,dndlogDpDEMOTT_2,'-',color='cyan',linewidth=2.5)
axs[1,0].semilogy([Dp1[iNiCONTROL_2]*0.001, Dp1[iNiCONTROL_2]*0.001], [10**(-7), dndlogDpCONTROL_2[iNiCONTROL_2]], 'k--')
axs[1,0].semilogy([Dp1[iNiDEMOTT_2]*0.001, Dp1[iNiDEMOTT_2]*0.001], [10**(-7), dndlogDpDEMOTT_2[iNiDEMOTT_2]], '--', color='cyan')
axs[1,0].semilogy([Dp1[iNiALLSIP_2]*0.001, Dp1[iNiALLSIP_2]*0.001], [10**(-7), dndlogDpALLSIP_2[iNiALLSIP_2]], '--', color='blue')
axs[1,0].set_ylabel("d($N_{ice}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[1,0].set_xlabel("$D_{p}$ (mm)")
axs[1,0].set_xlim(0.4*0.001,3000*0.001)
axs[1,0].set_ylim(1e-7,1e-1)
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,0].text(5.5*10**(-4),0.04,'(d)', fontsize=14)

L12 = axs[1,1].plot(Dp2*0.001,dndlogDpSCONTROL_2,'-k',linewidth=2.5,label='CONTROL')
L52 = axs[1,1].plot(Dp2*0.001,dndlogDpSDEMOTT_2,'-',color='cyan',linewidth=2.5,label='DEMOTT')
L32 = axs[1,1].plot(Dp2*0.001,dndlogDpSALLSIP_2,'-',color='blue',linewidth=2.5,label='ALLSIP')
axs[1,1].set_ylabel("d($N_{snow}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[1,1].semilogy([Dp2[iNsCONTROL_2]*0.001, Dp2[iNsCONTROL_2]*0.001], [10**(-7), dndlogDpSCONTROL_2[iNsCONTROL_2]], 'k--')
axs[1,1].semilogy([Dp2[iNsALLSIP_2]*0.001, Dp2[iNsALLSIP_2]*0.001], [10**(-7), dndlogDpSALLSIP_2[iNsALLSIP_2]], '--', color='blue')
axs[1,1].semilogy([Dp2[iNsDEMOTT_2]*0.001, Dp2[iNsDEMOTT_2]*0.001], [10**(-7), dndlogDpSDEMOTT_2[iNsDEMOTT_2]], '--', color='cyan')
axs[1,1].set_xlabel("$D_{p}$ (mm)")
axs[1,1].set_xlim(12*0.0001,30000*0.001)
axs[1,1].set_ylim(10**(-7),1e-1)
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
axs[1,1].text(0.0015,0.04,'(e)', fontsize=14)

L13 = axs[1,2].plot(Dp3*0.001,dndlogDpGCONTROL_2,'-k',linewidth=2.5,label='CONTROL')
axs[1,2].semilogy([Dp3[iNgDEMOTT_2]*0.001, Dp3[iNgDEMOTT_2]*0.001], [10**(-12), dndlogDpGDEMOTT_2[iNgDEMOTT_2]], '--', color='cyan')
L53 = axs[1,2].plot(Dp3*0.001,dndlogDpGDEMOTT_2,'-',color='cyan',linewidth=2.5,label='DEMOTT')
L33 = axs[1,2].plot(Dp3*0.001,dndlogDpGALLSIP_2,'-',color='blue',linewidth=2.5,label='ALLSIP')
axs[1,2].set_ylabel("d($N_{graupel}$)/d(log$D_{p}$) ($\mathrm{L^{-1}}$)")
axs[1,2].semilogy([Dp3[iNgCONTROL_2]*0.001, Dp3[iNgCONTROL_2]*0.001], [10**(-12), dndlogDpGCONTROL_2[iNgCONTROL_2]], 'k--')
axs[1,2].semilogy([Dp3[iNgALLSIP_2]*0.001, Dp3[iNgALLSIP_2]*0.001], [10**(-12), dndlogDpGALLSIP_2[iNgALLSIP_2]], '--', color='blue')
axs[1,2].set_xlabel("$D_{p}$ (mm)")
axs[1,2].set_xlim(10*0.0001,10*1000*0.001)
axs[1,2].set_ylim(10**(-9),10**(-2))
axs[1,2].set_xscale('log')
axs[1,2].set_yscale('log')
axs[1,2].text(0.0012,0.003,'(f)', fontsize=14)

plt.show()

fig.savefig(namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
