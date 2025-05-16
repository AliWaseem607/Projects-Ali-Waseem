# -*- coding: utf-8 -*-

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.rcParams['font.size']=15


nc=Dataset('./Data/ERA5/era5_pt_20211217_22.nc')
i_pt = 2
pt_lev=nc.variables['lev'][i_pt].item()

latlow = 30
lathigh =70
lonlow = -10#
lonhigh = 50
dlat=10
dlon=10



def prepare_basemap():
    m = Basemap(
            llcrnrlat=latlow,urcrnrlat=lathigh,\
            llcrnrlon=lonlow,urcrnrlon=lonhigh,\
          resolution='i', projection='merc', lat_0 = 39.5, lon_0 = 22.25
           )

    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()

    # draw parallels.
    parallels = np.arange(latlow,lathigh,dlat) 
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)

    # draw meridians
    meridians = np.arange(lonlow,lonhigh,dlon) 
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)

    lon_helm = 22.196297
    lat_helm = 37.9839
    xhelm,yhelm = m(lon_helm,lat_helm)
    m.scatter(xhelm,yhelm,marker='*',color='y',edgecolor='k',s=180,zorder=100,linewidth=1)
    
    return m


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

fig = plt.figure(figsize=(15,9))

namefigure = figures_directory + '/FigureS10.png'

m=prepare_basemap()

x,y=m(nc.variables['lon'][:].data,nc.variables['lat'][120:].data)
xx, yy = np.meshgrid(x, y)
pv_data = nc.variables['PV'][0,i_pt,:,:]
im=m.pcolormesh(x,y,pv_data[120:,:],vmin=0,cmap='YlOrRd')#,vmax=1)
plt.rcParams['font.size']=15
fig.colorbar(im,extend='max',shrink=1,aspect=25,label=r'Potential vorticity [K m$^2$ kg$^{-1}$ s$^{-1}]$')
fig.savefig(namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
