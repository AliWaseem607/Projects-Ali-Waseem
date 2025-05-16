# Notes

## crsim improvements
Basically the major improvement is done by extracting the file as 
```ncks -d south_north,164,164 -d west_east,153,153 -d south_north_stag,164,165 -d west_east_stag,153,154 wrfout_Helmos_d03_2024-10-08_00\:00\:00.nc wrfout_Helmos_d03_2024-10-08_HAC_test_stag_2.nc```
Additionally we can test if we really need the staggered grid to be two different points we can also take that as a single point as given by the get_lng_lat command. After that the parameters can be set so that it ranges an x y of 1 to 1, this time though we need to put back in the changing of iterations of the time as that is faster than cutting this file again to extract the time in a temp.nc file. Some more tests should be run to see if getting a far away time is harder than getting a close time. 

## To Do


## Available Plots
- Cross section of windspeed with terrain and PBLH
- domain map
- Plot of PBLH at two stations
- Period multi plot with reflectivity, temp, rh, and boundary layer height

## Plots to make

## Measurment locations
Overall there are three measurment locations for the current CHOPIN campaign I will be referring to them as HAC, NPRK, and SPRK. NPRK and SPRK stand for north parking lot and south parking lot respectively. Note that NPRK contains two different measurment stations that so happen to fall into the same grid cell for extraction so that is why they have been combined into one. More information for each of the mesaurement stations is given below

### Station measurements
HAC: 

VL: from old campaign

NPRK: location 1 (Radars, the Doppler LiDAR and the CIMEL), location 2 (one of the CCN counters, an SMPS, a picaro and a HTDMA)

SPRK: Mobile LiDAR, the UAVs and the helikite

### Latitude Longitude
VL: latitude = 37.9995 longitude = 22.19329
HAC: latitude = 37.9843, longitude = 22.1963
Mobile LiDAR, the UAVs and the helikite: latitude = 38.005528, longitude = 22.198667
Radars, the Doppler LiDAR and the CIMEL: latitude = 38.007444, longitude = 22.196028
one of the CCN counters, an SMPS, a picaro and a HTDMA: latitude = 38.004889, longitude = 22.193944

### WRF domain extraction
HAC: west_east 153 south_north 164 west_east_stag 153 south_north_stag 165

VL: west_east 152 south_north 166 west_east_stag 153 south_north_stag 167

Mobile LiDAR, the UAVs and the helikite (SPRK)
west_east 153 south_north 167 west_east_stag 153 south_north_stag 167

Radars, the Doppler LiDAR and the CIMEL (NPRK)
west_east 152 south_north 167 west_east_stag 153 south_north_stag 167

one of the CCN counters, an SMPS, a picaro and a HTDMA (NPRK)
west_east 152 south_north 167 west_east_stag 153 south_north_stag 167


## Freezing WRF
CHOPIN_oct20-23, CHOPIN_oct24-27, CHOPIN_envelop_oct17-20_MYNN all did not run several times in a row at 6 hour and 3 hour wrfrst times

CHOPIN_clear_oct27-30_KEPS and CHOPIN_oct16-19 had to be restart once and then they ran

CHOPIN_envelop_oct17-20_KEPS did run with a wrfrst of 30 minutes, I then accidentally ran the real again, but we can check to see if this changes anything as well

### Required outputs for the forum
namelist.input, wrfinput and wrfbdy files