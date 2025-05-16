# %%
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import img_tiles
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes._axes import Axes
from matplotlib.pyplot import Circle
from netCDF4 import Dataset
from wrf import destagger, getvar, rh

sys.path.append("./")
from scalebar import scale_bar

# %%
# load radiosonde data
# # oct23 = pd.read_csv(
# #     "data/radiosondes/2024-10-23-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # oct26 = pd.read_csv(
# #     "data/radiosondes/2024-10-26-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # oct29 = pd.read_csv(
# #     "data/radiosondes/2024-10-29-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # oct31 = pd.read_csv(
# #     "data/radiosondes/2024-10-31-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # nov2 = pd.read_csv(
# #     "data/radiosondes/2024-11-02-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # nov5_09 = pd.read_csv(
# #     "data/radiosondes/2024-11-05-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # nov5_15 = pd.read_csv(
# #     "data/radiosondes/2024-11-05-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# # )
# # # load WRF data
# # wrf_oct29_KEPS = Dataset("/scratch/waseem/CHOPIN_clear_oct27-30_KEPS/wrfout_CHPN_d03_2024-10-27_full_domain_KEPS.nc")
# # wrf_oct29_MYNN = Dataset("/scratch/waseem/CHOPIN_clear_oct27-30_MYNN/wrfout_CHPN_d03_2024-10-27_18:00:00.nc")
# # wrf_oct29_YSU = Dataset("/scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d03_2024-10-27_18:00:00.nc")

# # wrf_oct31_YSU = Dataset("/scratch/waseem/CHOPIN_oct29-nov1/wrfout_CHPN_d03_2024-10-29_18:00:00.nc")
# # wrf_oct31_MYNN = Dataset("/scratch/waseem/CHOPIN_oct29-nov1_MYNN/wrfout_CHPN_d03_2024-10-29_18:00:00.nc")
# # wrf_nov5 = Dataset("/scratch/waseem/CHOPIN_nov3-6/wrfout_CHPN_d03_2024-11-03_00:00:00.nc")

# # # load ERA5 data
# # ERA5_oct29 = Dataset("/work/lapi/waseem/ERA5/ERA5_ncfiles/20241025-20241029.nc")
# # ERA5_oct31 = Dataset("/work/lapi/waseem/ERA5/ERA5_ncfiles/20241031-20241031.nc")
# # ERA5_nov5 = Dataset("/work/lapi/waseem/ERA5/ERA5_ncfiles/20241101-20241105.nc")

# Load radiosonde data
# oct23_09 = pd.read_csv(
#     "data/radiosondes/2024-10-23-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
# )
oct26_09 = pd.read_csv(
    "data/radiosondes/2024-10-26-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
oct29_15 = pd.read_csv(
    "data/radiosondes/2024-10-29-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
oct31_09 = pd.read_csv(
    "data/radiosondes/2024-10-31-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov2_09 = pd.read_csv(
    "data/radiosondes/2024-11-02-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov5_09 = pd.read_csv(
    "data/radiosondes/2024-11-05-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov5_15 = pd.read_csv(
    "data/radiosondes/2024-11-05-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov15_11 = pd.read_csv(
    "./data/radiosondes/2024-11-15-11.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov16_09 = pd.read_csv(
    "./data/radiosondes/2024-11-16-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov17_09 = pd.read_csv(
    "./data/radiosondes/2024-11-17-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov18_09 = pd.read_csv(
    "./data/radiosondes/2024-11-18-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov18_15 = pd.read_csv(
    "./data/radiosondes/2024-11-18-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov19_09 = pd.read_csv(
    "./data/radiosondes/2024-11-19-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov19_15 = pd.read_csv(
    "./data/radiosondes/2024-11-19-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov20_09 = pd.read_csv(
    "./data/radiosondes/2024-11-20-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov21_09 = pd.read_csv(
    "./data/radiosondes/2024-11-21-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov21_15 = pd.read_csv(
    "./data/radiosondes/2024-11-21-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov22_09 = pd.read_csv(
    "./data/radiosondes/2024-11-22-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov23_15 = pd.read_csv(
    "./data/radiosondes/2024-11-23-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov24_09 = pd.read_csv(
    "./data/radiosondes/2024-11-24-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov25_15 = pd.read_csv(
    "./data/radiosondes/2024-11-25-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov26_09 = pd.read_csv(
    "./data/radiosondes/2024-11-26-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov26_15 = pd.read_csv(
    "./data/radiosondes/2024-11-26-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov28_15 = pd.read_csv(
    "./data/radiosondes/2024-11-28-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov29_09 = pd.read_csv(
    "./data/radiosondes/2024-11-29-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov29_15 = pd.read_csv(
    "./data/radiosondes/2024-11-29-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov30_09 = pd.read_csv(
    "./data/radiosondes/2024-11-30-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
nov30_15 = pd.read_csv(
    "./data/radiosondes/2024-11-30-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)
dec01_15 = pd.read_csv(
    "./data/radiosondes/2024-12-01-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec02_09 = pd.read_csv(
    "./data/radiosondes/2024-12-02-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec02_15 = pd.read_csv(
    "./data/radiosondes/2024-12-02-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec05_09 = pd.read_csv(
    "./data/radiosondes/2024-12-05-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec05_15 = pd.read_csv(
    "./data/radiosondes/2024-12-05-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec06_09 = pd.read_csv(
    "./data/radiosondes/2024-12-06-09.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec06_15 = pd.read_csv(
    "./data/radiosondes/2024-12-06-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)

dec13_15 = pd.read_csv(
    "./data/radiosondes/2024-12-06-15.csv", parse_dates=["time"], date_format=r"%Y-%m-%d %H:%M:%S"
)


## load the WRF data
# wrf_poi3_CONTROL = Dataset("/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_06:00:00.nc") # 15, 16
# wrf_poi3_SIP = Dataset("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_06:00:00.nc") # 15, 16

# wrf_nov15_CONTROL = Dataset("/scratch/waseem/CHOPIN_nov15-18/wrfout_CHPN_d03_2024-11-15_00:00:00.nc") # 17

# wrf_nov17_CONTROL = Dataset("/scratch/waseem/CHOPIN_nov17-20/wrfout_CHPN_d03_2024-11-17_00:00:00.nc") # 18, 19

# wrf_nov19_CONTROL = Dataset("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_00:00:00.nc") # 20, 21
# wrf_nov19_SIP = Dataset("/scratch/waseem/CHOPIN_nov19-22_SIP/wrfout_CHPN_d03_2024-11-19_00:00:00.nc") # 20, 21

# wrf_nov21_CONTROL = Dataset("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_00:00:00.nc") # 22, 23
# wrf_nov21_SIP = Dataset("/scratch/waseem/CHOPIN_nov21-24_SIP/wrfout_CHPN_d03_2024-11-21_00:00:00.nc") # 22, 23

# wrf_nov28_CONTROL = Dataset("/scratch/waseem/CHOPIN_nov28-dec1_CONTROL/wrfout_CHPN_d03_2024-11-28_06:00:00.nc") # 29, 30
# wrf_nov28_SIP = Dataset("/scratch/waseem/CHOPIN_nov28-dec1_SIP/wrfout_CHPN_d03_2024-11-28_06:00:00.nc") # 29, 30

# wrf_nov30_CONTROL = Dataset("/scratch/waseem/CHOPIN_nov30-dec3_CONTROL/wrfout_CHPN_d03_2024-11-30_00:00:00.nc") # 01
# wrf_nov30_SIP = Dataset("/scratch/waseem/CHOPIN_nov30-dec3_SIP/wrfout_CHPN_d03_2024-11-30_00:00:00.nc") # 01

# # wrf_oct27_KEPS = Dataset("/scratch/waseem/CHOPIN_clear_oct27-30_KEPS/wrfout_CHPN_d03_2024-10-27_full_domain_KEPS.nc")
# # wrf_oct27_MYNN = Dataset("/scratch/waseem/CHOPIN_clear_oct27-30_MYNN/wrfout_CHPN_d03_2024-10-27_18:00:00.nc")
# # wrf_oct27_YSU = Dataset("/scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d03_2024-10-27_18:00:00.nc")

# # wrf_oct29_YSU = Dataset("/scratch/waseem/CHOPIN_oct29-nov1/wrfout_CHPN_d03_2024-10-29_18:00:00.nc")
# # wrf_oct29_MYNN = Dataset("/scratch/waseem/CHOPIN_oct29-nov1_MYNN/wrfout_CHPN_d03_2024-10-29_18:00:00.nc")
# # wrf_nov5 = Dataset("/scratch/waseem/CHOPIN_nov3-6/wrfout_CHPN_d03_2024-11-03_00:00:00.nc")

wrf_oct24 = Path("/scratch/waseem/CHOPIN_oct24-27/wrfout_CHPN_d03_2024-10-24_full_domain.nc")  # 26

wrf_oct27_KEPS = Path(
    "/scratch/waseem/CHOPIN_clear_oct27-30_KEPS/wrfout_CHPN_d03_2024-10-27_full_domain_KEPS.nc"
)  # 29
wrf_oct27_MYNN = Path(
    "/scratch/waseem/CHOPIN_clear_oct27-30_MYNN/wrfout_CHPN_d03_2024-10-27_18:00:00.nc"
)  # 29
wrf_oct27_YSU = Path("/scratch/waseem/CHOPIN_clear_oct27-30/wrfout_CHPN_d03_2024-10-27_18:00:00.nc")  # 29

wrf_oct29_YSU = Path("/scratch/waseem/CHOPIN_oct29-nov1/wrfout_CHPN_d03_2024-10-29_18:00:00.nc")  # 31
wrf_oct29_MYNN = Path("/scratch/waseem/CHOPIN_oct29-nov1_MYNN/wrfout_CHPN_d03_2024-10-29_18:00:00.nc")  # 31
wrf_nov3_YSU = Path("/scratch/waseem/CHOPIN_nov3-6/wrfout_CHPN_d03_2024-11-03_00:00:00.nc")  # nov 5
wrf_nov3_MYNN = Path("/scratch/waseem/CHOPIN_nov3-6_MYNN/wrfout_CHPN_d03_2024-11-03_00:00:00.nc")  # nov 5

wrf_poi3_CONTROL = Path(
    "/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_06:00:00.nc"
)  # 15, 16
wrf_poi3_SIP = Path("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_06:00:00.nc")  # 15, 16

wrf_nov15_CONTROL = Path("/scratch/waseem/CHOPIN_nov15-18/wrfout_CHPN_d03_2024-11-15_00:00:00.nc")  # 17

wrf_nov17_CONTROL = Path("/scratch/waseem/CHOPIN_nov17-20/wrfout_CHPN_d03_2024-11-17_00:00:00.nc")  # 18, 19

wrf_nov19_CONTROL = Path("/scratch/waseem/CHOPIN_nov19-22/wrfout_CHPN_d03_2024-11-19_00:00:00.nc")  # 20, 21
wrf_nov19_SIP = Path("/scratch/waseem/CHOPIN_nov19-22_SIP/wrfout_CHPN_d03_2024-11-19_00:00:00.nc")  # 20, 21

wrf_nov21_CONTROL = Path("/scratch/waseem/CHOPIN_nov21-24/wrfout_CHPN_d03_2024-11-21_00:00:00.nc")  # 22, 23
wrf_nov21_SIP = Path("/scratch/waseem/CHOPIN_nov21-24_SIP/wrfout_CHPN_d03_2024-11-21_00:00:00.nc")  # 22, 23

wrf_nov28_CONTROL = Path(
    "/scratch/waseem/CHOPIN_nov28-dec1_CONTROL/wrfout_CHPN_d03_2024-11-28_06:00:00.nc"
)  # 29, 30
wrf_nov28_SIP = Path("/scratch/waseem/CHOPIN_nov28-dec1_SIP/wrfout_CHPN_d03_2024-11-28_06:00:00.nc")  # 29, 30

wrf_nov30_CONTROL = Path(
    "/scratch/waseem/CHOPIN_nov30-dec3_CONTROL/wrfout_CHPN_d03_2024-11-30_00:00:00.nc"
)  # 01
wrf_nov30_SIP = Path("/scratch/waseem/CHOPIN_nov30-dec3_SIP/wrfout_CHPN_d03_2024-11-30_00:00:00.nc")  # 01 02

wrf_dec3_CONTROL = Path(
    "/scratch/waseem/CHOPIN_dec3-6_CONTROL/wrfout_CHPN_d03_2024-12-03_18:00:00.nc"
)  # 05 06
wrf_dec3_SIP = Path("/scratch/waseem/CHOPIN_dec3-6_SIP/wrfout_CHPN_d03_2024-12-03_18:00:00.nc")  # 05 06

wrf_gd_0_CONTROL = Path("/scratch/waseem/CHOPIN_gd_0_CONTROL/wrfout_CHPN_d03_2024-11-23_00:00:00.nc")  # 2409

wrf_gd_0_SIP = Path("/scratch/waseem/CHOPIN_gd_0_SIP/wrfout_CHPN_d03_2024-11-23_00:00:00.nc")  # 2409

wrf_gd_1_CONTROL_2 = Path(
    "/scratch/waseem/CHOPIN_gd_1_CONTROL_2/wrfout_CHPN_d03_2024-10-27_00:00:00.nc"
)  # 29


# %%


def get_sat_vapor_pressure(temperature):
    Rv = 461.5  # J kg-1 K-1
    T0 = 273.16  # K
    Lv = 2260000  # J/kg
    es0 = 611.73  # Pa
    es = es0 * np.exp((Lv / Rv) * ((1 / T0) - (1 / temperature)))

    return es


def get_qvapor(rh, pressure, temperature):
    Rv = 461.5  # J kg-1 K-1
    Rd = 287  # J/kg·K

    e = get_sat_vapor_pressure(temperature) * rh

    w = e * Rd / (Rv * (pressure - e))
    return w


def get_sh(rh, pressure, temperature):
    Rv = 461.5  # J kg-1 K-1
    Rd = 287  # J/kg·K

    e = get_sat_vapor_pressure(temperature) * rh

    w = e * Rd / (Rv * (pressure - e))

    return w / (w + 1)


def get_wrf_data(
    radiosonde: pd.DataFrame, ncfile: Dataset | Path, save_path: Path | None = None, reset: bool = False
):
    if not reset and save_path is not None:
        if save_path.exists():
            return pd.read_csv(save_path, parse_dates=["time"])

    print("Starting: ")
    print(f"Radiosonde: {radiosonde.iloc[0]['time']}")
    print(f"ncfile: {ncfile}")
    if isinstance(ncfile, Path):
        ncfile = Dataset(str(ncfile))

    assert not isinstance(ncfile, Path)

    height_change = radiosonde.geopot_height.diff()
    mask = height_change > (-1)
    radiosonde = radiosonde.loc[mask].reset_index(drop=True).copy()

    wrf_times = pd.Series(getvar(ncfile, "Times", timeidx=None, meta=False))
    RD = 287.0
    CP = 1004.5
    P1000MB = 100000.0
    RA = 287.15
    EPS = 0.622
    radiosonde["wrf_T"] = pd.Series(dtype="float")
    radiosonde["wrf_RH"] = pd.Series(dtype="float")
    radiosonde["wrf_wind_speed"] = pd.Series(dtype="float")
    radiosonde["wrf_wind_dir"] = pd.Series(dtype="float")
    radiosonde["wrf_SH"] = pd.Series(dtype="float")
    radiosonde["wrf_P"] = pd.Series(dtype="float")
    radiosonde["wrf_rho"] = pd.Series(dtype="float")
    for i, row in radiosonde.iterrows():
        time_deltas = wrf_times - row.time
        timeidx = np.argmin(np.abs([x.total_seconds() for x in time_deltas]))
        geopot_height_stag = (
            ncfile.variables["PH"][timeidx, :, row.d03_y, row.d03_x]
            + ncfile.variables["PHB"][timeidx, :, row.d03_y, row.d03_x] / 9.81
        )
        geopot_height = destagger(geopot_height_stag, stagger_dim=0)

        total_pressure = np.squeeze(
            ncfile.variables["P"][timeidx, :, row.d03_y, row.d03_x]
            + ncfile.variables["PB"][timeidx, :, row.d03_y, row.d03_x]
        )  # in mb
        potential_temp = np.squeeze(ncfile.variables["T"][timeidx, :, row.d03_y, row.d03_x] + 300.0)
        kinetic_temp = (total_pressure / P1000MB) ** (RD / CP) * potential_temp
        qvapor = ncfile.variables["QVAPOR"][timeidx, :, row.d03_y, row.d03_x]
        qvapor_at_height = np.interp(row.geopot_height, geopot_height, qvapor)
        relative_humidity = rh(qvapor, total_pressure, kinetic_temp)
        u_stag = ncfile.variables["U"][timeidx, :, int(row.d03_y), int(row.d03_x) : int(row.d03_x) + 2]
        v_stag = ncfile.variables["V"][timeidx, :, int(row.d03_y) : int(row.d03_y) + 2, int(row.d03_x)]
        # w_stag = ncfile.variables["W"][timeidx, :, row.d03_y, row.d03_x]
        u = np.squeeze(destagger(u_stag, stagger_dim=1))
        v = np.squeeze(destagger(v_stag, stagger_dim=1))
        u_at_height = np.interp(row.geopot_height, geopot_height, u)
        v_at_height = np.interp(row.geopot_height, geopot_height, v)
        # w = np.squeeze(destagger(w_stag, stagger_dim=0))
        wind_speed = np.sqrt(u**2 + v**2)
        wind_direction_cart = np.arctan2(-v_at_height, -u_at_height) / np.pi * 180
        wind_direction_polar = (90 - wind_direction_cart + 360) % 360
        virtual_temp = kinetic_temp * (EPS + qvapor) / (EPS * (1.0 + qvapor))
        total_pressure_at_height = np.interp(row.geopot_height, geopot_height, total_pressure)
        virtual_temp_at_height = np.interp(row.geopot_height, geopot_height, virtual_temp)
        rho_at_height = total_pressure_at_height / RA / virtual_temp_at_height

        radiosonde.loc[i, "wrf_T"] = np.interp(row.geopot_height, geopot_height, kinetic_temp) - 273.15
        radiosonde.loc[i, "wrf_RH"] = np.interp(row.geopot_height, geopot_height, relative_humidity)
        radiosonde.loc[i, "wrf_wind_speed"] = np.interp(row.geopot_height, geopot_height, wind_speed)
        radiosonde.loc[i, "wrf_wind_dir"] = wind_direction_polar
        radiosonde.loc[i, "wrf_SH"] = qvapor_at_height / (1 + qvapor_at_height)
        radiosonde.loc[i, "wrf_P"] = np.interp(row.geopot_height, geopot_height, total_pressure) * 100
        radiosonde.loc[i, "wrf_rho"] = rho_at_height

    radiosonde["qvapor"] = get_qvapor(radiosonde["RH"], radiosonde["P"] * 100, radiosonde["T"] + 273.15)
    radiosonde["virtual_temperature"] = (
        (radiosonde["T"] + 273.15) * (EPS + radiosonde["qvapor"]) / (EPS * (1.0 + radiosonde["qvapor"]))
    )
    radiosonde["rho"] = radiosonde["P"] / RA / radiosonde["virtual_temperature"]
    radiosonde["SH"] = radiosonde["qvapor"] / (1 + radiosonde["qvapor"])

    if save_path is not None:
        print("saved")
        radiosonde.to_csv(save_path, index=False)

    print()
    return radiosonde


def get_ERA5_data(radiosonde: pd.DataFrame, ncfile: Dataset):
    height_change = radiosonde.geopot_height.diff()
    mask = height_change > (-1)
    radiosonde = radiosonde.loc[mask].reset_index(drop=True).copy()
    radiosonde["ERA5_T"] = pd.Series(dtype="float")
    radiosonde["ERA5_RH"] = pd.Series(dtype="float")
    radiosonde["ERA5_wind_speed"] = pd.Series(dtype="float")
    radiosonde["ERA5_wind_dir"] = pd.Series(dtype="float")

    ERA5_lat_distance = np.abs(
        np.tile(ncfile.variables["latitude"][:].filled(), reps=(len(radiosonde.lat), 1))
        - radiosonde.lat.to_numpy().reshape((len(radiosonde.lat), 1))
    )
    ERA5_lat_idx = np.argmin(ERA5_lat_distance, axis=1)
    del ERA5_lat_distance

    ERA5_lon_distance = np.abs(
        np.tile(ncfile.variables["longitude"][:].filled(), reps=(len(radiosonde.lon), 1))
        - radiosonde.lon.to_numpy().reshape((len(radiosonde.lon), 1))
    )

    ERA5_lon_idx = np.argmin(ERA5_lon_distance, axis=1)
    del ERA5_lon_distance

    radiosonde_delta_times = (
        (radiosonde.time - pd.Timestamp(year=1970, month=1, day=1))
        .apply(lambda x: x.total_seconds())
        .to_numpy()
    )
    time_difference = np.tile(
        ncfile.variables["valid_time"][:].filled(), reps=(len(radiosonde.time), 1)
    ) - radiosonde_delta_times.reshape((len(radiosonde.time), 1))
    ERA5_time_idx = np.argmax(np.where(time_difference < 0, time_difference, -np.inf), axis=1)
    del time_difference
    del radiosonde_delta_times

    for i, row in radiosonde.iterrows():
        ERA5_relative_humidity = ncfile.variables["r"][ERA5_time_idx[i], :, ERA5_lat_idx[i], ERA5_lon_idx[i]]
        ERA5_temperature = ncfile.variables["t"][ERA5_time_idx[i], :, ERA5_lat_idx[i], ERA5_lon_idx[i]]
        geopot_height = ncfile.variables["z"][ERA5_time_idx[i], :, ERA5_lat_idx[i], ERA5_lon_idx[i]] / 9.81
        u = ncfile.variables["u"][ERA5_time_idx[i], :, ERA5_lat_idx[i], ERA5_lon_idx[i]]
        v = ncfile.variables["v"][ERA5_time_idx[i], :, ERA5_lat_idx[i], ERA5_lon_idx[i]]
        u_at_height = np.interp(row.geopot_height, geopot_height, u)
        v_at_height = np.interp(row.geopot_height, geopot_height, v)
        # w = np.squeeze(destagger(w_stag, stagger_dim=0))
        wind_speed = np.sqrt(u**2 + v**2)
        wind_direction_cart = np.arctan2(-v_at_height, -u_at_height) / np.pi * 180
        wind_direction_polar = (90 - wind_direction_cart + 360) % 360
        radiosonde.loc[i, "ERA5_T"] = np.interp(row.geopot_height, geopot_height, ERA5_temperature) - 273.15
        radiosonde.loc[i, "ERA5_RH"] = np.interp(row.geopot_height, geopot_height, ERA5_relative_humidity)
        radiosonde.loc[i, "ERA5_wind_speed"] = np.interp(row.geopot_height, geopot_height, wind_speed)
        radiosonde.loc[i, "ERA5_wind_dir"] = wind_direction_polar

    return radiosonde


# %%
wrf_oct24_oct26_09 = get_wrf_data(
    radiosonde=oct26_09,
    ncfile=wrf_oct24,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-26-09_wrf_added_YSU.csv"),
)

wrf_oct27_YSU_oct29_15 = get_wrf_data(
    radiosonde=oct29_15,
    ncfile=wrf_oct27_YSU,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-29-15_wrf_added_YSU.csv"),
)

wrf_oct27_MYNN_oct29_15 = get_wrf_data(
    radiosonde=oct29_15,
    ncfile=wrf_oct27_MYNN,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-29-15_wrf_added_MYNN.csv"),
)

wrf_oct27_KEPS_oct29_15 = get_wrf_data(
    radiosonde=oct29_15,
    ncfile=wrf_oct27_KEPS,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-29-15_wrf_added_KEPS.csv"),
)

wrf_gd_1_CONTROL_2_oct29_15 = get_wrf_data(
    radiosonde=oct29_15,
    ncfile=wrf_gd_1_CONTROL_2,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-29-15_wrf_added_gd_1_CONTROL_2.csv"),
)

wrf_oct29_YSU_oct31_09 = get_wrf_data(
    radiosonde=oct31_09,
    ncfile=wrf_oct29_YSU,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-31-09_wrf_added_YSU.csv"),
)

wrf_oct29_MYNN_oct31_09 = get_wrf_data(
    radiosonde=oct31_09,
    ncfile=wrf_oct29_MYNN,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-10-31-09_wrf_added_MYNN.csv"),
)

wrf_nov3_YSU_nov5_09 = get_wrf_data(
    radiosonde=nov5_09,
    ncfile=wrf_nov3_YSU,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-05-09_wrf_added_YSU.csv"),
)

wrf_nov3_MYNN_nov5_09 = get_wrf_data(
    radiosonde=nov5_09,
    ncfile=wrf_nov3_MYNN,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-05-09_wrf_added_MYNN.csv"),
)

wrf_nov3_YSU_nov5_15 = get_wrf_data(
    radiosonde=nov5_15,
    ncfile=wrf_nov3_YSU,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-05-15_wrf_added_YSU.csv"),
)

wrf_nov3_MYNN_nov5_15 = get_wrf_data(
    radiosonde=nov5_15,
    ncfile=wrf_nov3_MYNN,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-05-15_wrf_added_MYNN.csv"),
)

wrf_poi3_CONTROL_nov15_11 = get_wrf_data(
    radiosonde=nov15_11,
    ncfile=wrf_poi3_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-15-11_wrf_added_CONTROL.csv"),
)
wrf_poi3_SIP_nov15_11 = get_wrf_data(
    radiosonde=nov15_11,
    ncfile=wrf_poi3_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-15-11_wrf_added_SIP.csv"),
)
wrf_poi3_CONTROL_nov16_09 = get_wrf_data(
    radiosonde=nov16_09,
    ncfile=wrf_poi3_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-16-09_wrf_added_CONTROL.csv"),
)
wrf_poi3_SIP_nov16_09 = get_wrf_data(
    radiosonde=nov16_09,
    ncfile=wrf_poi3_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-16-09_wrf_added_SIP.csv"),
)
wrf_nov15_CONTROL_nov17_09 = get_wrf_data(
    radiosonde=nov17_09,
    ncfile=wrf_nov15_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-17-09_wrf_added_CONTROL.csv"),
)

wrf_nov17_CONTROL_nov18_09 = get_wrf_data(
    radiosonde=nov18_09,
    ncfile=wrf_nov17_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-18-09_wrf_added_CONTROL.csv"),
)
wrf_nov17_CONTROL_nov18_15 = get_wrf_data(
    radiosonde=nov18_15,
    ncfile=wrf_nov17_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-18-15_wrf_added_CONTROL.csv"),
)
wrf_nov17_CONTROL_nov19_09 = get_wrf_data(
    radiosonde=nov19_09,
    ncfile=wrf_nov17_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-19-09_wrf_added_CONTROL.csv"),
)
wrf_nov17_CONTROL_nov19_15 = get_wrf_data(
    radiosonde=nov19_15,
    ncfile=wrf_nov17_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-19-15_wrf_added_CONTROL.csv"),
)
wrf_nov19_CONTROL_nov20_09 = get_wrf_data(
    radiosonde=nov20_09,
    ncfile=wrf_nov19_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-20-09_wrf_added_CONTROL.csv"),
)
wrf_nov19_SIP_nov20_09 = get_wrf_data(
    radiosonde=nov20_09,
    ncfile=wrf_nov19_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-20-09_wrf_added_SIP.csv"),
)
wrf_nov19_CONTROL_nov21_09 = get_wrf_data(
    radiosonde=nov21_09,
    ncfile=wrf_nov19_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-21-09_wrf_added_CONTROL.csv"),
)
wrf_nov19_SIP_nov21_09 = get_wrf_data(
    radiosonde=nov21_09,
    ncfile=wrf_nov19_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-21-09_wrf_added_SIP.csv"),
)
wrf_nov19_CONTROL_nov21_15 = get_wrf_data(
    radiosonde=nov21_15,
    ncfile=wrf_nov19_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-21-15_wrf_added_CONTROL.csv"),
)
wrf_nov19_SIP_nov21_15 = get_wrf_data(
    radiosonde=nov21_15,
    ncfile=wrf_nov19_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-21-15_wrf_added_SIP.csv"),
)
wrf_nov21_CONTROL_nov22_09 = get_wrf_data(
    radiosonde=nov22_09,
    ncfile=wrf_nov21_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-22-09_wrf_added_CONTROL.csv"),
)
wrf_nov21_SIP_nov22_09 = get_wrf_data(
    radiosonde=nov22_09,
    ncfile=wrf_nov21_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-22-09_wrf_added_SIP.csv"),
)
wrf_nov21_CONTROL_nov23_15 = get_wrf_data(
    radiosonde=nov23_15,
    ncfile=wrf_nov21_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-23-15_wrf_added_CONTROL.csv"),
)
wrf_nov21_SIP_nov23_15 = get_wrf_data(
    radiosonde=nov23_15,
    ncfile=wrf_nov21_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-23-15_wrf_added_SIP.csv"),
)

wrf_gd_0_CONTROL_nov24_09 = get_wrf_data(
    radiosonde=nov24_09,
    ncfile=wrf_gd_0_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-24-09_wrf_added_CONTROL.csv"),
)
wrf_gd_0_SIP_nov24_09 = get_wrf_data(
    radiosonde=nov24_09,
    ncfile=wrf_gd_0_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-24-09_wrf_added_SIP.csv"),
)

wrf_nov28_CONTROL_nov29_09 = get_wrf_data(
    radiosonde=nov29_09,
    ncfile=wrf_nov28_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-29-09_wrf_added_CONTROL.csv"),
)
wrf_nov28_SIP_nov29_09 = get_wrf_data(
    radiosonde=nov29_09,
    ncfile=wrf_nov28_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-29-09_wrf_added_SIP.csv"),
)
wrf_nov28_CONTROL_nov29_15 = get_wrf_data(
    radiosonde=nov29_15,
    ncfile=wrf_nov28_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-29-15_wrf_added_CONTROL.csv"),
)
wrf_nov28_SIP_nov29_15 = get_wrf_data(
    radiosonde=nov29_15,
    ncfile=wrf_nov28_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-29-15_wrf_added_SIP.csv"),
)
wrf_nov28_CONTROL_nov30_09 = get_wrf_data(
    radiosonde=nov30_09,
    ncfile=wrf_nov28_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-30-09_wrf_added_CONTROL.csv"),
)
wrf_nov28_SIP_nov30_09 = get_wrf_data(
    radiosonde=nov30_09,
    ncfile=wrf_nov28_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-30-09_wrf_added_SIP.csv"),
)
wrf_nov28_CONTROL_nov30_15 = get_wrf_data(
    radiosonde=nov30_15,
    ncfile=wrf_nov28_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-30-15_wrf_added_CONTROL.csv"),
)
wrf_nov28_SIP_nov30_15 = get_wrf_data(
    radiosonde=nov30_15,
    ncfile=wrf_nov28_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-11-30-15_wrf_added_SIP.csv"),
)
wrf_nov30_CONTROL_dec01_15 = get_wrf_data(
    radiosonde=dec01_15,
    ncfile=wrf_nov30_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-01-15_wrf_added_CONTROL.csv"),
)
wrf_nov30_SIP_dec01_15 = get_wrf_data(
    radiosonde=dec01_15,
    ncfile=wrf_nov30_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-01-15_wrf_added_SIP.csv"),
)

wrf_nov30_CONTROL_dec02_15 = get_wrf_data(
    radiosonde=dec02_15,
    ncfile=wrf_nov30_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-02-15_wrf_added_CONTROL.csv"),
)
wrf_nov30_SIP_dec02_15 = get_wrf_data(
    radiosonde=dec02_15,
    ncfile=wrf_nov30_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-02-15_wrf_added_SIP.csv"),
)

wrf_nov30_CONTROL_dec02_09 = get_wrf_data(
    radiosonde=dec02_09,
    ncfile=wrf_nov30_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-02-09_wrf_added_CONTROL.csv"),
)
wrf_nov30_SIP_dec02_09 = get_wrf_data(
    radiosonde=dec02_09,
    ncfile=wrf_nov30_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-02-09_wrf_added_SIP.csv"),
)

wrf_dec3_CONTROL_dec05_09 = get_wrf_data(
    radiosonde=dec05_09,
    ncfile=wrf_dec3_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-05-09_wrf_added_CONTROL.csv"),
)

wrf_dec3_SIP_dec05_09 = get_wrf_data(
    radiosonde=dec05_09,
    ncfile=wrf_dec3_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-05-09_wrf_added_SIP.csv"),
)

wrf_dec3_CONTROL_dec05_15 = get_wrf_data(
    radiosonde=dec05_15,
    ncfile=wrf_dec3_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-05-15_wrf_added_CONTROL.csv"),
)

wrf_dec3_SIP_dec05_15 = get_wrf_data(
    radiosonde=dec05_15,
    ncfile=wrf_dec3_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-05-15_wrf_added_SIP.csv"),
)

wrf_dec3_CONTROL_dec06_09 = get_wrf_data(
    radiosonde=dec06_09,
    ncfile=wrf_dec3_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-06-09_wrf_added_CONTROL.csv"),
)

wrf_dec3_SIP_dec06_09 = get_wrf_data(
    radiosonde=dec06_09,
    ncfile=wrf_dec3_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-06-09_wrf_added_SIP.csv"),
)

wrf_dec3_CONTROL_dec06_15 = get_wrf_data(
    radiosonde=dec06_15,
    ncfile=wrf_dec3_CONTROL,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-06-15_wrf_added_CONTROL.csv"),
)

wrf_dec3_SIP_dec06_15 = get_wrf_data(
    radiosonde=dec06_15,
    ncfile=wrf_dec3_SIP,
    save_path=Path("./data/radiosondes/wrf_data_added/2024-12-06-15_wrf_added_SIP.csv"),
)


# %%

# # oct23_wrf_added = get_wrf_data(oct23, )
# # oct26_wrf_added = get_wrf_data(oct26, )
# oct29_wrf_added_YSU = get_wrf_data(oct29, wrf_oct29_YSU)
# oct29_wrf_added_KEPS = get_wrf_data(oct29, wrf_oct29_KEPS)
# oct29_wrf_added_MYNN = get_wrf_data(oct29, wrf_oct29_MYNN)
# oct31_wrf_added = get_wrf_data(oct31, wrf_oct31)
# # nov2_wrf_added = get_wrf_data(nov2,)
# nov5_09_wrf_added = get_wrf_data(nov5_09, wrf_nov5)
# nov5_15_wrf_added = get_wrf_data(nov5_15, wrf_nov5)

# # path = Path("data/radiosondes/wrf_data_added/2024-10-23-09_wrf_added_YSU.csv")
# # if not path.exists():
# #     oct29_wrf_added_YSU = get_wrf_data(oct29, wrf_oct29_YSU)
# #     oct29_wrf_added_YSU.to_csv(path, index=False)
# # else:
# #     oct29_wrf_added_YSU = pd.read_csv(path, parse_dates=["time"])

# # path = Path("data/radiosondes/wrf_data_added/2024-10-23-09_wrf_added_KEPS.csv")
# # if not path.exists():
# #     oct29_wrf_added_KEPS = get_wrf_data(oct29, wrf_oct29_KEPS)
# #     oct29_wrf_added_KEPS.to_csv(path, index=False)
# # else:
# #     oct29_wrf_added_KEPS = pd.read_csv(path, parse_dates=["time"])

# # path = Path("data/radiosondes/wrf_data_added/2024-10-23-09_wrf_added_MYNN.csv")
# # if not path.exists():
# #     oct29_wrf_added_MYNN = get_wrf_data(oct29, wrf_oct29_MYNN)
# #     oct29_wrf_added_MYNN.to_csv(path, index=False)
# # else:
# #     oct29_wrf_added_MYNN = pd.read_csv(path, parse_dates=["time"])

# # path = Path("data/radiosondes/wrf_data_added/2024-10-31-09_wrf_added_YSU.csv")
# # if not path.exists():
# #     oct31_wrf_added = get_wrf_data(oct31, wrf_oct31)
# #     oct31_wrf_added.to_csv(path, index=False)
# # else:
# #     oct31_wrf_added = pd.read_csv(path, parse_dates=["time"])

# # path = Path("data/radiosondes/wrf_data_added/2024-11-05-09_wrf_added_YSU.csv")
# # if not path.exists():
# #     nov5_09_wrf_added = get_wrf_data(nov5_09, wrf_nov5)
# #     nov5_09_wrf_added.to_csv(path, index=False)
# # else:
# #     nov5_09_wrf_added = pd.read_csv(path, parse_dates=["time"])


# # path = Path("data/radiosondes/wrf_data_added/2024-11-05-15_wrf_added_YSU.csv")
# # if not path.exists():
# #     nov5_15_wrf_added = get_wrf_data(nov5_15, wrf_nov5)
# #     nov5_15_wrf_added.to_csv(path, index=False)
# # else:
# #     nov5_15_wrf_added = pd.read_csv(path, parse_dates=["time"])

# # oct29_all_KEPS = get_ERA5_data(oct29_wrf_added_KEPS, ERA5_oct29)
# # oct29_all_MYNN = get_ERA5_data(oct29_wrf_added_MYNN, ERA5_oct29)
# # oct29_all_YSU = get_ERA5_data(oct29_wrf_added_YSU, ERA5_oct29)
# # oct31_all = get_ERA5_data(oct31_wrf_added, ERA5_oct31)
# # nov5_09_all = get_ERA5_data(nov5_09_wrf_added, ERA5_nov5)
# # nov5_15_all = get_ERA5_data(nov5_15_wrf_added, ERA5_nov5)


# %%
def plot_radiosonde_temp_comparison(
    ax: Axes,
    radiosonde: pd.DataFrame,
    plot_radiosonde: bool = True,
    wrf: bool = True,
    ERA5: bool = True,
    wrf_label: str = "WRF Simulation",
    wrf_color: str = "saddlebrown",
):
    if plot_radiosonde:
        im0 = ax.scatter(
            radiosonde["T"],
            radiosonde.geopot_height / 1000,
            c=radiosonde.fly_time / 60,
            cmap="jet",
            s=1,
            label="Radiosonde",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Fly time [min]")
    if wrf:
        ax.plot(
            radiosonde["wrf_T"],
            radiosonde.geopot_height / 1000,
            color=wrf_color,
            label=wrf_label,
            linestyle="-",
        )
    if ERA5:
        ax.plot(
            radiosonde["ERA5_T"],
            radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Geopotential Height [km]")


def plot_radiosonde_RH_comparison(
    ax: Axes,
    radiosonde: pd.DataFrame,
    plot_radiosonde: bool = True,
    wrf: bool = True,
    ERA5: bool = True,
    wrf_label: str = "WRF Simulation",
    wrf_color: str = "saddlebrown",
):
    if plot_radiosonde:
        im0 = ax.scatter(
            radiosonde["RH"],
            radiosonde.geopot_height / 1000,
            c=radiosonde.fly_time / 60,
            cmap="jet",
            s=1,
            label="Radiosonde",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Fly time [min]")
    if wrf:
        ax.plot(
            radiosonde["wrf_RH"],
            radiosonde.geopot_height / 1000,
            color=wrf_color,
            label=wrf_label,
            linestyle="-",
        )
    if ERA5:
        ax.plot(
            radiosonde["ERA5_RH"],
            radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    ax.legend()
    ax.set_xlabel("Relative Humidity [%]")
    ax.set_ylabel("Geopotential Height [km]")
    ax.set_xlim(0, 100)


def plot_radiosonde_wind_speed_comparison(
    ax: Axes,
    radiosonde: pd.DataFrame,
    plot_radiosonde: bool = True,
    wrf: bool = True,
    ERA5: bool = True,
    wrf_label: str = "WRF Simulation",
    wrf_color: str = "saddlebrown",
):
    if plot_radiosonde:
        im0 = ax.scatter(
            radiosonde["wind_speed"],
            radiosonde.geopot_height / 1000,
            c=radiosonde.fly_time / 60,
            cmap="jet",
            s=1,
            label="Radiosonde",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Fly time [min]")
    if wrf:
        ax.plot(
            radiosonde["wrf_wind_speed"],
            radiosonde.geopot_height / 1000,
            color=wrf_color,
            label=wrf_label,
            linestyle="-",
        )
    if ERA5:
        ax.plot(
            radiosonde["ERA5_wind_speed"],
            radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    ax.legend()
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Geopotential Height [km]")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(-xmax * 0.05, xmax)


def plot_radiosonde_wind_dir_comparison(
    ax: Axes,
    radiosonde: pd.DataFrame,
    fontsize: float = 11,
    plot_radiosonde: bool = True,
    wrf: bool = True,
    ERA5: bool = True,
    wrf_label: str = "WRF Simulation",
    wrf_color: str = "saddlebrown",
):
    max_range = radiosonde.geopot_height.max() / 1000
    plot_range = int(np.ceil(max_range))
    ax.set_ylim(-plot_range, plot_range)
    ax.set_xlim(-plot_range, plot_range)
    ax.axvline(0, ymin=-plot_range, ymax=plot_range, color="k", alpha=0.75)
    ax.axhline(0, xmin=-plot_range, xmax=plot_range, color="k", alpha=0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    last_tic_range = np.floor(plot_range / 2.5) * 2.5
    tic_steps = int((last_tic_range - 2.5) / 2.5 + 1)
    circles_radi = np.linspace(2.5, last_tic_range, tic_steps)
    circles = [Circle((0, 0), x, fill=False, color="grey", alpha=0.75) for x in circles_radi]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for circ in circles:
        ax.add_patch(circ)

    ax.add_patch(Circle((0, 0), plot_range, fill=False, color="k"))
    ax.set_yticks(-circles_radi, [str(x) for x in circles_radi])

    ax.text(-0.75, plot_range + 1, "N", fontdict={"size": fontsize})
    ax.text(-0.75, -plot_range - 2, "S", fontdict={"size": fontsize})

    ax.text(plot_range + 0.5, -1, "E", fontdict={"size": fontsize})
    ax.text(-plot_range - 3, -1, "W", fontdict={"size": fontsize})

    if plot_radiosonde:
        radiosonde_x = np.cos(radiosonde.wind_dir / 180 * np.pi)
        radiosonde_y = np.sin(radiosonde.wind_dir / 180 * np.pi)
        im0 = ax.scatter(
            radiosonde_x * radiosonde.geopot_height / 1000,
            radiosonde_y * radiosonde.geopot_height / 1000,
            c=radiosonde.fly_time / 60,
            cmap="jet",
            s=1,
            label="Radiosonde",
        )
        cbar = plt.colorbar(im0)
        cbar.set_label("Flight Time [min]")
    if wrf:
        wrf_wind_dir = radiosonde.wrf_wind_dir.to_numpy(dtype="float") / 180 * np.pi
        wrf_x = np.cos(wrf_wind_dir)
        wrf_y = np.sin(wrf_wind_dir)
        ax.plot(
            wrf_x * radiosonde.geopot_height / 1000,
            wrf_y * radiosonde.geopot_height / 1000,
            color=wrf_color,
            label=wrf_label,
            linestyle="-",
        )

    if ERA5:
        era5_wind_dir = radiosonde.ERA5_wind_dir.to_numpy(dtype="float") / 180 * np.pi
        era5_x = np.cos(era5_wind_dir)
        era5_y = np.sin(era5_wind_dir)
        ax.plot(
            era5_x * radiosonde.geopot_height / 1000,
            era5_y * radiosonde.geopot_height / 1000,
            color="magenta",
            label="ERA5",
            linestyle="--",
        )

    ax.set_ylabel("Radius is Geopot. Height [km]")
    ax.set_xlabel("Wind Direction", labelpad=30)
    ax.legend()


def plot_radiosonde_flight_path(ax: GeoAxes, radiosonde: pd.DataFrame):
    ax.scatter(radiosonde.lon, radiosonde.lat, c=radiosonde.fly_time, cmap="jet", s=0.5)
    ax.scatter(
        radiosonde.iloc[0]["lon"], radiosonde.iloc[0]["lat"], marker="s", color="blue", s=45, label="Start"
    )
    ax.scatter(
        radiosonde.iloc[-1]["lon"], radiosonde.iloc[-1]["lat"], marker="o", color="red", s=45, label="End"
    )
    ax.legend()
    buffer = 0.05
    min_lat = radiosonde.lat.min() - buffer / 2
    max_lat = radiosonde.lat.max() + buffer / 2
    min_lon = radiosonde.lon.min() - buffer
    max_lon = radiosonde.lon.max() + buffer
    tiler = img_tiles.GoogleTiles()
    dlat = max_lat - min_lat
    dlon = max_lon - min_lon
    min_axis = np.min([dlat, dlon])
    zoom_level = 10
    if min_axis < 0.2:
        zoom_level = 11
    if min_axis < 0.15:
        zoom_level = 12
    ax.add_image(tiler, zoom_level)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_title("Flight Path")


# # # dfs = [
# # #     oct29_all_YSU,
# # #     oct29_all_KEPS,
# # #     oct29_all_MYNN,
# # #     oct31_all,
# # #     nov5_09_all,
# # #     nov5_15_all,
# # # ]
# # # titles = ["Oct 29th, YSU", "Oct 29th, KEPS", "Oct 29th, MYNN", "Oct 31st", "Nov 5th 9am", "Nov 5th 3pm"]
# # # save_names = [
# # #     "radiosonde_comparison_2024-10-29-09_YSU.png",
# # #     "radiosonde_comparison_2024-10-29-09_KEPS.png",
# # #     "radiosonde_comparison_2024-10-29-09_MYNN.png",
# # #     "radiosonde_comparison_2024-10-31-09.png",
# # #     "radiosonde_comparison_2024-11-05-09.png",
# # #     "radiosonde_comparison_2024-11-05-15.png",
# # # ]

# plt.rcdefaults()
legend_size = 10
small_size = 12
medium_size = 13
large_size = 14
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=legend_size)  # legend fontsize
plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
plt.rc("axes", titlesize= medium_size)

# Plot the boundary layer comparison

fig = plt.figure(figsize=(9, 10.5))
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())

plot_radiosonde_temp_comparison(
    ax1, wrf_oct27_YSU_oct29_15, wrf_label="YSU", wrf_color="cornflowerblue", ERA5=False
)
plot_radiosonde_RH_comparison(
    ax2, wrf_oct27_YSU_oct29_15, wrf_label="YSU", wrf_color="cornflowerblue", ERA5=False
)
plot_radiosonde_wind_speed_comparison(
    ax3, wrf_oct27_YSU_oct29_15, wrf_label="YSU", wrf_color="cornflowerblue", ERA5=False
)
plot_radiosonde_wind_dir_comparison(
    ax4, wrf_oct27_YSU_oct29_15, wrf_label="YSU", wrf_color="cornflowerblue", ERA5=False
)

plot_radiosonde_temp_comparison(
    ax1, wrf_oct27_MYNN_oct29_15, wrf_label="MYNN", wrf_color="forestgreen", ERA5=False, plot_radiosonde=False
)
plot_radiosonde_RH_comparison(
    ax2, wrf_oct27_MYNN_oct29_15, wrf_label="MYNN", wrf_color="forestgreen", ERA5=False, plot_radiosonde=False
)
plot_radiosonde_wind_speed_comparison(
    ax3, wrf_oct27_MYNN_oct29_15, wrf_label="MYNN", wrf_color="forestgreen", ERA5=False, plot_radiosonde=False
)
plot_radiosonde_wind_dir_comparison(
    ax4, wrf_oct27_MYNN_oct29_15, wrf_label="MYNN", wrf_color="forestgreen", ERA5=False, plot_radiosonde=False
)

plot_radiosonde_temp_comparison(
    ax1, wrf_oct27_KEPS_oct29_15, wrf_label="KEPS", wrf_color="darkorchid", ERA5=False, plot_radiosonde=False
)
plot_radiosonde_RH_comparison(
    ax2, wrf_oct27_KEPS_oct29_15, wrf_label="KEPS", wrf_color="darkorchid", ERA5=False, plot_radiosonde=False
)
plot_radiosonde_wind_speed_comparison(
    ax3, wrf_oct27_KEPS_oct29_15, wrf_label="KEPS", wrf_color="darkorchid", ERA5=False, plot_radiosonde=False
)
plot_radiosonde_wind_dir_comparison(
    ax4, wrf_oct27_KEPS_oct29_15, wrf_label="KEPS", wrf_color="darkorchid", ERA5=False, plot_radiosonde=False
)

plot_radiosonde_flight_path(ax5, wrf_oct27_YSU_oct29_15)
scale_bar(ax5, location=(0.2, 0.05))

plt.tight_layout()
plt.savefig(f"figures/final_plots/radiosonde_bl_comparison", bbox_inches="tight", dpi=300)

# Plot general comparisons
radiosondes_metadata = pd.read_csv("./data/metadata_radiosondes.csv", parse_dates=["start_time", "end_time"])
radiosondes_metadata["sip"] = radiosondes_metadata.sip.apply(lambda x: True if x == 1 else False)

# # for df, title, save_name in zip(dfs, titles, save_names):
# #     fig = plt.figure(figsize=(9, 10.5))
# #     gs = fig.add_gridspec(3, 2)
# #     ax1 = fig.add_subplot(gs[0, 0])
# #     ax2 = fig.add_subplot(gs[0, 1])
# #     ax3 = fig.add_subplot(gs[1, 0])
# #     ax4 = fig.add_subplot(gs[1, 1])
# #     ax5 = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())
# #     plot_radiosonde_temp_comparison(ax1, df)
# #     plot_radiosonde_RH_comparison(ax2, df)
# #     plot_radiosonde_wind_speed_comparison(ax3, df)
# #     plot_radiosonde_wind_dir_comparison(ax4, df)
# #     plot_radiosonde_flight_path(ax5, df)
# #     scale_bar(ax5, location=(0.2, 0.05))

# #     plt.tight_layout()
# #     fig.suptitle(title, y=1.01)
# #     plt.savefig(f"figures/radiosonde_comparison/{save_name}", bbox_inches="tight")

# %%
# statistics
data = []
for df, scheme in zip(
    [wrf_oct27_YSU_oct29_15, wrf_oct27_MYNN_oct29_15, wrf_oct27_KEPS_oct29_15], ["YSU", "MYNN", "KEPS"]
):
    RMSE_T = np.sqrt(np.mean((df["T"] - df["wrf_T"]) ** 2))
    RMSE_WD = np.sqrt(np.mean((df["wind_dir"] - df["wrf_wind_dir"]) ** 2))
    RMSE_WS = np.sqrt(np.mean((df["wind_speed"] - df["wrf_wind_speed"]) ** 2))

    dZ = np.diff(df.geopot_height) / 2
    water_vapour = df.SH * df.rho * 10**3  # g m-3
    wrf_water_vapour = df.wrf_SH * df.wrf_rho * 10**3  # g m-3

    water_vapour_centered = np.diff(water_vapour) / 2 + water_vapour.iloc[:-1]
    wrf_water_vapour_centered = np.diff(wrf_water_vapour) / 2 + wrf_water_vapour.iloc[:-1]

    WVC = water_vapour_centered * dZ
    wrf_WVC = wrf_water_vapour_centered * dZ
    WVP = np.sum(WVC)
    wrf_WVP = np.sum(wrf_WVC)

    WVP_bias = wrf_WVP - WVP

    data.append(
        {
            "PBL Scheme": scheme,
            "RMSE Temperature [K]": RMSE_T,
            "RMSE Wind Speed [m/s]": RMSE_WS,
            "RMSE Wind Direction [deg]": RMSE_WD,
            "Integrated Water Vapour Error [g/m2]": WVP_bias,
        }
    )

results = pd.DataFrame(data)
results.set_index("PBL Scheme")
results.to_csv("./figures/final_plots/radiosonde_BL_RMSE.csv", float_format="%.2f")
