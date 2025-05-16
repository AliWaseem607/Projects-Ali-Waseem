# %%
import glob
import re
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar

# %%


def update_WRF_metadate() -> None:
    metadata = pd.read_csv(
        "/home/waseem/CHOPIN_analysis/data/metadata.csv",
        parse_dates=["start_time", "end_time", "true_start_time"],
    )
    wrfout_path = glob.glob("/home/waseem/CHOPIN_analysis/data/**/wrfout/*.nc", recursive=True)
    stations = ["NPRK", "SPRK", "HAC", "VL"]

    for file in wrfout_path:
        if re.search("NPRK", file) is None:
            continue
        ncfile = Dataset(file, mode="a")
        if not "Zhh_BASTA" in ncfile.variables.keys():
            print(file)
            print("No BASTA")
        if not "Zhh_MIRA" in ncfile.variables.keys():
            print(file)
            print("No MIRA")

    new_data = []
    for file in wrfout_path:
        if re.search("MIRACUT", file) is not None:
            continue

        file_path = Path(file)
        if str(file_path.absolute()) in set(metadata.file_path):
            continue

        station = None
        for station_name in stations:
            if re.search(station_name, file_path.stem) is not None:
                station = station_name
                break
        ncfile = Dataset(str(file_path.absolute()))
        mp = ncfile.MP_PHYSICS
        bl = ncfile.BL_PBL_PHYSICS
        SIP = ncfile.SIP
        times = pd.Series(getvar(ncfile, "Times", meta=False, timeidx=None))  # type: ignore
        ncfile.close()
        new_row = {
            "file_path": str(file_path.absolute()),
            "start_time": times.iloc[0],
            "end_time": times.iloc[-1],
            "spinup": 24,
            "true_start_time": times.iloc[0] + pd.Timedelta(24, "h"),
            "mp": mp,
            "bl": bl,
            "sip": SIP,
            "station": station,
        }
        new_data.append(new_row)

    if len(new_data) > 0:
        metadata = pd.concat([metadata, pd.DataFrame(new_data)])
        metadata.sort_values("start_time", inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        metadata.to_csv("/home/waseem/CHOPIN_analysis/data/metadata.csv", index=False)


def update_MIRA_metadata() -> None:
    metadata = pd.read_csv(
        "/home/waseem/CHOPIN_analysis/data/metadata_MIRA.csv", parse_dates=["start_time", "end_time"]
    )
    MIRA_paths = glob.glob("/scratch/waseem/MIRA_insitu/gathered_data/*.nc")

    new_data = []
    for file in MIRA_paths:
        file_path = Path(file)
        if str(file_path.absolute()) in set(metadata.file_path):
            continue
        ncfile = Dataset(str(file_path.absolute()))
        times = pd.to_datetime(
            ncfile.variables["time"][:], origin=pd.Timestamp(year=1970, month=1, day=1), unit="s"
        )
        ncfile.close()
        new_row = {"file_path": str(file_path.absolute()), "start_time": times[0], "end_time": times[-1]}
        new_data.append(new_row)

    if len(new_data) > 0:
        metadata = pd.concat([metadata, pd.DataFrame(new_data)])
        metadata.sort_values("start_time", inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        metadata.to_csv("/home/waseem/CHOPIN_analysis/data/metadata_MIRA.csv", index=False)


def update_MIRA_metadata_trimmed() -> None:
    metadata = pd.read_csv(
        "/home/waseem/CHOPIN_analysis/data/metadata_MIRA_trimmed.csv", parse_dates=["start_time", "end_time"]
    )
    MIRA_paths = glob.glob("/scratch/waseem/MIRA_insitu/trimmed_data/*.nc")

    new_data = []
    for file in MIRA_paths:
        file_path = Path(file)
        if str(file_path.absolute()) in set(metadata.file_path):
            continue
        ncfile = Dataset(str(file_path.absolute()))
        times = pd.to_datetime(
            ncfile.variables["time"][:], origin=pd.Timestamp(year=1970, month=1, day=1), unit="s"
        )
        ncfile.close()
        new_row = {"file_path": str(file_path.absolute()), "start_time": times[0], "end_time": times[-1]}
        new_data.append(new_row)

    if len(new_data) > 0:
        metadata = pd.concat([metadata, pd.DataFrame(new_data)])
        metadata.sort_values("start_time", inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        metadata.to_csv("/home/waseem/CHOPIN_analysis/data/metadata_MIRA_trimmed.csv", index=False)


def update_BASTA_metadata() -> None:
    metadata = pd.read_csv("/home/waseem/CHOPIN_analysis/data/metadata_BASTA.csv")
    BASTA_paths = glob.glob("/scratch/waseem/BASTA_insitu/12m5/*.nc")

    new_data = []
    for file in BASTA_paths:
        file_path = Path(file)
        if str(file_path.absolute()) in set(metadata.file_path):
            continue
        ncfile = Dataset(str(file_path.absolute()))
        file_start_time = pd.Timestamp(year=int(ncfile.year), month=int(ncfile.month), day=int(ncfile.day))
        times = pd.to_datetime(ncfile.variables["time"][:], unit="s", origin=file_start_time)
        ncfile.close()
        new_row = {"file_path": str(file_path.absolute()), "start_time": times[0], "end_time": times[-1]}
        new_data.append(new_row)

    if len(new_data) > 0:
        metadata = pd.concat([metadata, pd.DataFrame(new_data)])
        metadata.sort_values("start_time", inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        metadata.to_csv("/home/waseem/CHOPIN_analysis/data/metadata_BASTA.csv", index=False)


# %%
update_WRF_metadate()
print("updated wrf")
update_MIRA_metadata()
print("updated MIRA")
update_MIRA_metadata_trimmed()
print("updated trimmed MIRA")
update_BASTA_metadata()
print("updated BASTA")
