# %%

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import ll_to_xy


def get_new_lat_lon(
    wind_speed: float,
    wind_direction: float,
    geoheight_end: float,
    dt: float,
    lat: float,
    lon: float,
    geoheight_start: float,
) -> tuple[float, float]:
    """
    wind_speed is in m/s, wind direction is in degrees with 0 equal to north,
    geoheight start is the geopot_heightential height at the start of the measurment,
    geoheight end is the geopot_heightential height at the end of the measurment. dt
    is the amount of time of the measurement. lat is the starting latitude,
    lon is the starting longitude, both in degrees
    """
    R = 6378137  # Earth Radius at "sea level"
    wind_dir_rad = wind_direction / 180 * np.pi
    u = wind_speed * np.sin(wind_dir_rad + np.pi) * dt  # + np.pi because wind dir is where wind is from
    w = wind_speed * np.cos(wind_dir_rad + np.pi) * dt  # + np.pi because wind dir is where wind is from
    delta_lat_rad = np.arcsin(w / (R + (geoheight_start + geoheight_end) / 2))
    delta_lon_rad = np.arcsin(u / (R + (geoheight_start + geoheight_end) / 2))
    delta_lat_deg = delta_lat_rad * 180 / np.pi
    delta_lon_deg = delta_lon_rad * 180 / np.pi
    new_lat = lat + delta_lat_deg
    new_lon = lon + delta_lon_deg
    return new_lat, new_lon


def clean_txt_no_latlon(file_path, ncfile):
    cols = ["fly_time", "time", "P", "T", "RH", "wind_speed", "wind_dir", "geopot_height", "dewpoint"]
    cols_dtype = {
        "fly_time": "int",
        "time": "str",
        "P": "float",
        "T": "float",
        "RH": "float",
        "wind_speed": "float",
        "wind_dir": "float",
        "geopot_height": "float",
        "dewpoint": "float",
    }

    file = pd.read_csv(
        file_path,
        header=None,
        skiprows=20,
        skipfooter=10,
        sep=r"\s+",
        names=cols,
        dtype=cols_dtype,
        encoding="unicode_escape",
        engine="python",
        na_values="-",
    )

    file["lat"] = None
    file["lon"] = None
    file["d03_x"] = None
    file["d03_y"] = None
    with open(file_path, "r", encoding="cp1252") as f:
        text = f.read()
        text = text.replace("\t", " ").replace("\n", " ")

    lat = float(re.search(r"Latitude: *([0-9.]+)", text).groups()[0])
    lon = float(re.search(r"Longitude: *([0-9.]+)", text).groups()[0])
    file.loc[0, "lat"] = lat
    file.loc[0, "lon"] = lon
    x, y = ll_to_xy(ncfile, lat, lon, meta=False)
    file.loc[0, "d03_x"] = x
    file.loc[0, "d03_y"] = y
    base_time_string = file_path.parent.parent.stem
    year = int(base_time_string[:4])
    month = int(base_time_string[4:6])
    day = int(base_time_string[6:8])
    base_time = pd.Timestamp(year=year, month=month, day=day)
    file["time"] = file.time.apply(
        lambda x: base_time
        + pd.Timedelta(int(x.split(":")[0]), "h")
        + pd.Timedelta(int(x.split(":")[1]), "m")
        + pd.Timedelta(int(x.split(":")[2]), "second")
    )

    for i in range(1, len(file)):
        dt = file.loc[i, "fly_time"] - file.loc[i - 1, "fly_time"]
        new_lat, new_lon = get_new_lat_lon(
            wind_speed=file.loc[i, "wind_speed"],
            wind_direction=file.loc[i, "wind_dir"],
            geoheight_end=file.loc[i, "geopot_height"],
            geoheight_start=file.loc[i - 1, "geopot_height"],
            dt=dt,
            lat=file.loc[i - 1, "lat"],
            lon=file.loc[i - 1, "lon"],
        )
        file.loc[i, "lat"] = new_lat
        file.loc[i, "lon"] = new_lon
        if np.isnan(new_lat) or np.isnan(new_lon):
            file.loc[i, "d03_x"] = np.nan
            file.loc[i, "d03_y"] = np.nan
            continue
        x, y = ll_to_xy(ncfile, new_lat, new_lon, meta=False)
        file.loc[i, "d03_x"] = x
        file.loc[i, "d03_y"] = y

    new_file_name = (
        f"{base_time_string[:4]}-{base_time_string[4:6]}-{base_time_string[6:8]}-{base_time_string[8:]}.csv"
    )
    save_path = Path("/home/waseem/CHOPIN_analysis/data/radiosondes/", new_file_name)

    file.to_csv(save_path, index=False)


def clean_txt_latlon(file_path: Path, ncfile: Dataset):
    cols = [
        "fly_time",
        "time",
        "P",
        "T",
        "RH",
        "wind_speed",
        "wind_dir",
        "lon",
        "lat",
        "geopot_height",
        "dewpoint",
    ]
    cols_dtype = {
        "fly_time": "int",
        "time": "str",
        "P": "float",
        "T": "float",
        "RH": "float",
        "wind_speed": "float",
        "wind_dir": "float",
        "lat": "float",
        "lon": "float",
        "geopot_height": "float",
        "dewpoint": "float",
    }
    file = pd.read_csv(
        file_path,
        header=None,
        skiprows=20,
        skipfooter=10,
        sep=r"\s+",
        names=cols,
        dtype=cols_dtype,
        encoding="unicode_escape",
        engine="python",
        na_values="-",
    )

    file["d03_x"] = None
    file["d03_y"] = None

    base_time_string = file_path.parent.parent.stem
    year = int(base_time_string[:4])
    month = int(base_time_string[4:6])
    day = int(base_time_string[6:8])
    base_time = pd.Timestamp(year=year, month=month, day=day)
    file["time"] = file.time.apply(
        lambda x: base_time
        + pd.Timedelta(int(x.split(":")[0]), "h")
        + pd.Timedelta(int(x.split(":")[1]), "m")
        + pd.Timedelta(int(x.split(":")[2]), "second")
    )

    for i in range(0, len(file)):
        if pd.isna(file.loc[i, "lat"]) or pd.isna(file.loc[i, "lon"]):
            file.loc[i, "d03_x"] = np.nan
            file.loc[i, "d03_y"] = np.nan
            continue

        x, y = ll_to_xy(ncfile, file.loc[i, "lat"], file.loc[i, "lon"], meta=False)
        file.loc[i, "d03_x"] = int(x)
        file.loc[i, "d03_y"] = int(y)

    new_file_name = (
        f"{base_time_string[:4]}-{base_time_string[4:6]}-{base_time_string[6:8]}-{base_time_string[8:]}.csv"
    )
    save_path = Path("/home/waseem/CHOPIN_analysis/data/radiosondes/", new_file_name)

    file.to_csv(save_path, index=False)


def add_latlon_comparison(file_path: Path):
    cols = [
        "fly_time",
        "time",
        "P",
        "T",
        "RH",
        "wind_speed",
        "wind_dir",
        "lon",
        "lat",
        "geopot_height",
        "dewpoint",
    ]
    cols_dtype = {
        "fly_time": "int",
        "time": "str",
        "P": "float",
        "T": "float",
        "RH": "float",
        "wind_speed": "float",
        "wind_dir": "float",
        "lat": "float",
        "lon": "float",
        "geopot_height": "float",
        "dewpoint": "float",
    }
    file = pd.read_csv(
        file_path,
        header=None,
        skiprows=20,
        skipfooter=10,
        sep=r"\s+",
        names=cols,
        dtype=cols_dtype,
        encoding="unicode_escape",
        engine="python",
        na_values="-",
    )

    file["lat_calculated"] = None
    file["lon_calculated"] = None
    with open(file_path, "r", encoding="cp1252") as f:
        text = f.read()
        text = text.replace("\t", " ").replace("\n", " ")

    lat = float(re.search(r"Latitude: *([0-9.]+)", text).groups()[0])
    lon = float(re.search(r"Longitude: *([0-9.]+)", text).groups()[0])
    file.loc[0, "lat_calculated"] = lat
    file.loc[0, "lon_calculated"] = lon
    base_time_string = file_path.parent.parent.stem
    year = int(base_time_string[:4])
    month = int(base_time_string[4:6])
    day = int(base_time_string[6:8])
    base_time = pd.Timestamp(year=year, month=month, day=day)
    file["time"] = file.time.apply(
        lambda x: base_time
        + pd.Timedelta(int(x.split(":")[0]), "h")
        + pd.Timedelta(int(x.split(":")[1]), "m")
        + pd.Timedelta(int(x.split(":")[2]), "second")
    )

    for i in range(1, len(file)):
        dt = file.loc[i, "fly_time"] - file.loc[i - 1, "fly_time"]
        new_lat, new_lon = get_new_lat_lon(
            wind_speed=file.loc[i, "wind_speed"],
            wind_direction=file.loc[i, "wind_dir"],
            geoheight_end=file.loc[i, "geopot_height"],
            geoheight_start=file.loc[i - 1, "geopot_height"],
            dt=dt,
            lat=file.loc[i - 1, "lat_calculated"],
            lon=file.loc[i - 1, "lon_calculated"],
        )
        file.loc[i, "lat_calculated"] = new_lat
        file.loc[i, "lon_calculated"] = new_lon

    new_file_name = (
        f"{base_time_string[:4]}-{base_time_string[4:6]}-{base_time_string[6:8]}-{base_time_string[8:]}.csv"
    )
    save_path = Path("/home/waseem/CHOPIN_analysis/data/radiosondes/latlon_comparison_2", new_file_name)

    file.to_csv(save_path, index=False)


# %%

# Read radiosonde data
# 2024102309 - no useable data
# 2024102509 - Using summary II at a low resolution
# 2024102609 - We have the raw data
# 2024102616 - no useable data
# 2024102915 - we have the raw data
# 2024103109 - we have the raw data
# 2024110209 - we have the raw data
# 2024110509 - we have the raw data
# 2024110515 - we have the raw data

###########################
##### OLD CLEANING ########
###########################
# # # #   PRES.     TIME     HGT/MSL  TEMP.   RH     DEWP   W.D    W.S
# # # #   mb     HH:MM:SS   Meter     °C      %      °C    Deg.  Knots
# # # path_1 = r"/scratch/waseem/Radiosondes/2024102509/20241025090000000000_SummaryII.txt"

# # # # Time   	UTC Time   	P       	T    	Hu         	Ws      	Wd   	Geopot_height  	Dewp.
# # # # [sec]     [HH:mm:ss] 	[hPa]    	[°C]    [%]        	[m/s]   	[°]    	[m]     	[°C]

# # # path_2 = (
# # #     r"/scratch/waseem/Radiosondes/2024102609/SOUNDING DATA/20241026090000000000_UPP_RAW_%%%%%_2024102609.txt"
# # # )
# # # path_3 = (
# # #     r"/scratch/waseem/Radiosondes/2024102915/SOUNDING DATA/20241029150022021731_UPP_RAW_%%%%%_2024102915.txt"
# # # )
# # # path_4 = (
# # #     r"/scratch/waseem/Radiosondes/2024103109/SOUNDING DATA/20241031090022015930_UPP_RAW_%%%%%_2024103109.txt"
# # # )
# # # path_5 = (
# # #     r"/scratch/waseem/Radiosondes/2024110209/SOUNDING DATA/20241102090022017598_UPP_RAW_%%%%%_2024110209.txt"
# # # )
# # # path_6 = (
# # #     r"/scratch/waseem/Radiosondes/2024110509/SOUNDING DATA/20241105090022021759_UPP_RAW_%%%%%_2024110509.txt"
# # # )
# # # path_7 = (
# # #     r"/scratch/waseem/Radiosondes/2024110515/SOUNDING DATA/20241105150022016508_UPP_RAW_%%%%%_2024110515.txt"
# # # )

# # # ncfile_path = r"/scratch/waseem/CHOPIN_oct24-27/wrfout_CHPN_d03_2024-10-24_full_domain.nc"
# # # ncfile = Dataset(ncfile_path)
# # # # %%
# # # cols = ["P", "time", "geopot_height", "T", "RH", "dewpoint", "wind_dir", "wind_speed"]
# # # file_1 = pd.read_csv(path_1, header=None, skiprows=19, nrows=9, sep=r"\s+").reset_index(drop=True)
# # # file_1.columns = cols
# # # file_1.wind_speed = file_1.wind_speed * 0.514444
# # # file_1["fly_time"] = file_1.time.apply(
# # #     lambda x: int(x.split(":")[0]) * 60 * 60 + int(x.split(":")[1]) * 60 + int(x.split(":")[2])
# # # )

# # # base_time_string = path_1.split("/")[4]
# # # year = int(base_time_string[:4])
# # # month = int(base_time_string[4:6])
# # # day = int(base_time_string[6:8])

# # # base_time = pd.Timestamp(year=year, month=month, day=day)
# # # file_1["time"] = file_1.time.apply(
# # #     lambda x: base_time
# # #     + pd.Timedelta(int(x.split(":")[0]), "h")
# # #     + pd.Timedelta(int(x.split(":")[1]), "m")
# # #     + pd.Timedelta(int(x.split(":")[2]), "second")
# # # )
# # # cols_dtype = {
# # #     "fly_time": "int",
# # #     "time": "datetime64[ns]",
# # #     "P": "float",
# # #     "T": "float",
# # #     "RH": "float",
# # #     "wind_speed": "float",
# # #     "wind_dir": "float",
# # #     "geopot_height": "float",
# # #     "dewpoint": "float",
# # # }
# # # file_1.astype(cols_dtype)

# # # cols = ["fly_time", "time", "P", "T", "RH", "wind_speed", "wind_dir", "geopot_height", "dewpoint"]
# # # cols_dtype = {
# # #     "fly_time": "int",
# # #     "time": "str",
# # #     "P": "float",
# # #     "T": "float",
# # #     "RH": "float",
# # #     "wind_speed": "float",
# # #     "wind_dir": "float",
# # #     "geopot_height": "float",
# # #     "dewpoint": "float",
# # # }
# # # file_2 = pd.read_csv(
# # #     path_2,
# # #     header=None,
# # #     skiprows=20,
# # #     skipfooter=10,
# # #     sep=r"\s+",
# # #     names=cols,
# # #     dtype=cols_dtype,
# # #     encoding="unicode_escape",
# # #     engine="python",
# # #     na_values="-",
# # # )
# # # file_3 = pd.read_csv(
# # #     path_3,
# # #     header=None,
# # #     skiprows=20,
# # #     skipfooter=10,
# # #     sep=r"\s+",
# # #     names=cols,
# # #     dtype=cols_dtype,
# # #     encoding="unicode_escape",
# # #     engine="python",
# # #     na_values="-",
# # # )
# # # file_4 = pd.read_csv(
# # #     path_4,
# # #     header=None,
# # #     skiprows=20,
# # #     skipfooter=10,
# # #     sep=r"\s+",
# # #     names=cols,
# # #     dtype=cols_dtype,
# # #     encoding="unicode_escape",
# # #     engine="python",
# # #     na_values="-",
# # # )
# # # file_5 = pd.read_csv(
# # #     path_5,
# # #     header=None,
# # #     skiprows=20,
# # #     skipfooter=10,
# # #     sep=r"\s+",
# # #     names=cols,
# # #     dtype=cols_dtype,
# # #     encoding="unicode_escape",
# # #     engine="python",
# # #     na_values="-",
# # # )
# # # file_6 = pd.read_csv(
# # #     path_6,
# # #     header=None,
# # #     skiprows=20,
# # #     skipfooter=10,
# # #     sep=r"\s+",
# # #     names=cols,
# # #     dtype=cols_dtype,
# # #     encoding="unicode_escape",
# # #     engine="python",
# # #     na_values="-",
# # # )
# # # file_7 = pd.read_csv(
# # #     path_7,
# # #     header=None,
# # #     skiprows=20,
# # #     skipfooter=10,
# # #     sep=r"\s+",
# # #     names=cols,
# # #     dtype=cols_dtype,
# # #     encoding="unicode_escape",
# # #     engine="python",
# # #     na_values="-",
# # # )

# # # file_1["lat"] = None
# # # file_1["lon"] = None
# # # file_1["d03_x"] = None
# # # file_1["d03_y"] = None

# # # file_1.loc[0, "lat"] = 38.007503
# # # file_1.loc[0, "lon"] = 22.196147
# # # x, y = ll_to_xy(ncfile, 38.007503, 22.196147, meta=False)
# # # file_1.loc[0, "d03_x"] = x
# # # file_1.loc[0, "d03_y"] = y

# # # for path, file in zip(
# # #     [path_2, path_3, path_4, path_5, path_6, path_7], [file_2, file_3, file_4, file_5, file_6, file_7]
# # # ):
# # #     file["lat"] = None
# # #     file["lon"] = None
# # #     file["d03_x"] = None
# # #     file["d03_y"] = None
# # #     with open(path, "r", encoding="cp1252") as f:
# # #         text = f.read()
# # #         text = text.replace("\t", " ").replace("\n", " ")

# # #     lat = float(re.search(r"Latitude: *([0-9.]+)", text).groups()[0])
# # #     lon = float(re.search(r"Longitude: *([0-9.]+)", text).groups()[0])
# # #     file.loc[0, "lat"] = lat
# # #     file.loc[0, "lon"] = lon
# # #     x, y = ll_to_xy(ncfile, lat, lon, meta=False)
# # #     file.loc[0, "d03_x"] = x
# # #     file.loc[0, "d03_y"] = y
# # #     base_time_string = path.split("/")[4]
# # #     year = int(base_time_string[:4])
# # #     month = int(base_time_string[4:6])
# # #     day = int(base_time_string[6:8])
# # #     base_time = pd.Timestamp(year=year, month=month, day=day)
# # #     file["time"] = file.time.apply(
# # #         lambda x: base_time
# # #         + pd.Timedelta(int(x.split(":")[0]), "h")
# # #         + pd.Timedelta(int(x.split(":")[1]), "m")
# # #         + pd.Timedelta(int(x.split(":")[2]), "second")
# # #     )

# %%
# Calculate all lat and lon where available.


# # # for file in [file_1, file_2, file_3, file_4, file_5, file_6, file_7]:
# # #     for i in range(1, len(file)):
# # #         dt = file.loc[i, "fly_time"] - file.loc[i - 1, "fly_time"]
# # #         new_lat, new_lon = get_new_lat_lon(
# # #             wind_speed=file.loc[i, "wind_speed"],
# # #             wind_direction=file.loc[i, "wind_dir"],
# # #             geoheight_end=file.loc[i, "geopot_height"],
# # #             geoheight_start=file.loc[i - 1, "geopot_height"],
# # #             dt=dt,
# # #             lat=file.loc[i - 1, "lat"],
# # #             lon=file.loc[i - 1, "lon"],
# # #         )
# # #         file.loc[i, "lat"] = new_lat
# # #         file.loc[i, "lon"] = new_lon
# # #         if np.isnan(new_lat) or np.isnan(new_lon):
# # #             file.loc[i, "d03_x"] = np.nan
# # #             file.loc[i, "d03_y"] = np.nan
# # #             continue
# # #         x, y = ll_to_xy(ncfile, new_lat, new_lon, meta=False)
# # #         file.loc[i, "d03_x"] = x
# # #         file.loc[i, "d03_y"] = y

# # # # %%
# # # mask_1 = np.any(file_1.isna(), axis=1)
# # # mask_2 = np.any(file_2.isna(), axis=1)
# # # mask_3 = np.any(file_3.isna(), axis=1)
# # # mask_4 = np.any(file_4.isna(), axis=1)
# # # mask_5 = np.any(file_5.isna(), axis=1)
# # # mask_6 = np.any(file_6.isna(), axis=1)
# # # mask_7 = np.any(file_7.isna(), axis=1)

# # # # %%
# # # file_1.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-10-23-09.csv", index=False)
# # # file_2.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-10-26-09.csv", index=False)
# # # file_3.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-10-29-15.csv", index=False)
# # # file_4.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-10-31-09.csv", index=False)
# # # file_5.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-11-02-09.csv", index=False)
# # # file_6.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-11-05-09.csv", index=False)
# # # file_7.to_csv("/home/waseem/CHOPIN_analysis/data/radiosondes/2024-11-05-15.csv", index=False)

# # # # %%
# # # maxs_x = [
# # #     file_1.d03_x.max(),
# # #     file_2.d03_x.max(),
# # #     file_3.d03_x.max(),
# # #     file_4.d03_x.max(),
# # #     file_5.d03_x.max(),
# # #     file_6.d03_x.max(),
# # #     file_7.d03_x.max(),
# # # ]
# # # mins_x = [
# # #     file_1.d03_x.min(),
# # #     file_2.d03_x.min(),
# # #     file_3.d03_x.min(),
# # #     file_4.d03_x.min(),
# # #     file_5.d03_x.min(),
# # #     file_6.d03_x.min(),
# # #     file_7.d03_x.min(),
# # # ]
# # # print(np.max(maxs_x))
# # # print(np.min(mins_x))


# # # maxs_y = [
# # #     file_1.d03_y.max(),
# # #     file_2.d03_y.max(),
# # #     file_3.d03_y.max(),
# # #     file_4.d03_y.max(),
# # #     file_5.d03_y.max(),
# # #     file_6.d03_y.max(),
# # #     file_7.d03_y.max(),
# # # ]
# # # mins_y = [
# # #     file_1.d03_y.min(),
# # #     file_2.d03_y.min(),
# # #     file_3.d03_y.min(),
# # #     file_4.d03_y.min(),
# # #     file_5.d03_y.min(),
# # #     file_6.d03_y.min(),
# # #     file_7.d03_y.min(),
# # # ]
# # # print(np.max(maxs_y))
# # # print(np.min(mins_y))

###########################
###########################
###########################

# %%

# Read radiosonde data
# 2024111511 - Raw data with no lat lon
# 2024111609 - Raw data with no lat lon
# 2024111709 - Raw data with no lat lon
# 2024111809 - Raw data with no lat lon
# 2024111815 - Raw data with no lat lon
# 2024111909 - Raw data with lat lon
# 2024111915 - Raw data with lat lon
# 2024112009 - Raw data with lat lon
# 2024112109 - Raw data with lat lon
# 2024112115 - Raw data with lat lon
# 2024112209 - Raw data with lat lon
# 2024112315 - Raw data with lat lon
# 2024112409 - Raw data with lat lon
# 2024112515 - Raw data with lat lon
# 2024112609 - Raw data with lat lon
# 2024112614 - no data
# 2024112615 - Raw data with lat lon
# 2024112815 - Raw data with lat lon
# 2024112909 - Raw data with lat lon
# 2024112915 - Raw data with lat lon
# 2024113009 - Raw data with lat lon
# 2024113015 - Raw data with lat lon
# 2024120115 - Raw data with lat lon
# %%
radiosondes_path = Path("/scratch/waseem/Radiosondes_3/")
sample_ncfile_path = Path("/scratch/waseem/CHOPIN_nov11-14/wrfout_CHPN_d03_2024-11-11_00:00:00.nc")
ncfile = Dataset(str(sample_ncfile_path))

no_lat_lon = [ "2024102609", "2024102915",
"2024103109",
"2024110209",
"2024110509",
"2024110515", "2024111511", "2024111609", "2024111709", "2024111809", "2024111815",]
lat_lon = [
    "2024111909",
    "2024111915",
    "2024112009",
    "2024112109",
    "2024112115",
    "2024112209",
    "2024112315",
    "2024112409",
    "2024112515",
    "2024112609",
    "2024112615",
    "2024112815",
    "2024112909",
    "2024112915",
    "2024113009",
    "2024113015",
    "2024120115",
    "2024120209",
    "2024120215",
    "2024120509",
    "2024120515",
    "2024120609",
    "2024120615",
    "2024121315",
]

for path in radiosondes_path.iterdir():
    print(f"cleaning {path}")
    if path.stem in no_lat_lon:
        file_path = Path(glob.glob(str(path) + "/SOUNDING DATA/*.txt")[0])
        clean_txt_no_latlon(file_path, ncfile)
    if path.stem in lat_lon:
        file_path = Path(glob.glob(str(path) + "/SOUNDING DATA/*.txt")[0])
        clean_txt_latlon(file_path, ncfile)

# %%
# # # # verificaiton of lat lon finding
# # # radiosondes_path = Path("/scratch/waseem/Radiosondes_3/")

# # # lat_lon = [
# # #     "2024111909",
# # #     "2024111915",
# # #     "2024112009",
# # #     "2024112109",
# # #     "2024112115",
# # #     "2024112209",
# # #     "2024112315",
# # #     "2024112409",
# # #     "2024112515",
# # #     "2024112609",
# # #     "2024112615",
# # #     "2024112815",
# # #     "2024112909",
# # #     "2024112915",
# # #     "2024113009",
# # #     "2024113015",
# # #     "2024120115",
# # #     "2024120209",
# # #     "2024120215",
# # #     "2024120509",
# # #     "2024120515",
# # #     "2024120609",
# # #     "2024120615",
# # #     "2024121315",
# # # ]

# # # for path in radiosondes_path.iterdir():
# # #     if path.stem in lat_lon:
# # #         file_path = Path(glob.glob(str(path) + "/SOUNDING DATA/*.txt")[0])
# # #         add_latlon_comparison(file_path)
