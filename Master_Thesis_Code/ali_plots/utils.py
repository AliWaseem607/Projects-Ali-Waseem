import datetime
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset  # type: ignore


def get_variable_namelist(variable_name: str, namelist_text: str) -> list[str]:
    """
    Function to obtain the values of a variable from the namelist.input file.
    """
    pattern_all_values = rf"{variable_name} *= *([0-9. ,]+)+"
    result_all_values = re.search(pattern_all_values, namelist_text)
    assert result_all_values is not None
    all_values = result_all_values.groups()[0]

    pattern_individual_values = r"([0-9.]+)"
    return re.findall(pattern_individual_values, all_values)


def get_start_time(namelist_input_path: Path) -> np.datetime64:
    """
    Function to obtain the start time for a wrf simulation for the 3rd domain
    """
    with open(namelist_input_path, "r") as f:
        text = f.read()
    start_year = get_variable_namelist("start_year", text)[2]
    start_month = get_variable_namelist("start_month", text)[2]
    start_day = get_variable_namelist("start_day", text)[2]
    start_hour = get_variable_namelist("start_hour", text)[2]
    start_minute = get_variable_namelist("start_minute", text)[2]

    return np.datetime64(f"{start_year}-{start_month}-{start_day}T{start_hour}:{start_minute}")


def get_run_time(namelist_input_path: Path) -> np.timedelta64:
    """
    Function to obtain the run time for a wrf simulation.
    """

    with open(namelist_input_path, "r") as f:
        text = f.read()

    run_days = get_variable_namelist("run_days", text)[0]
    run_hours = get_variable_namelist("run_hours", text)[0]
    run_minutes = get_variable_namelist("run_minutes", text)[0]
    run_seconds = get_variable_namelist("run_seconds", text)[0]

    return (
        np.timedelta64(run_days, "D")
        + np.timedelta64(run_hours, "h")
        + np.timedelta64(run_minutes, "m")
        + np.timedelta64(run_seconds, "s")
    )


def get_wrf_times(namelist_input_path: Path, spinup_time: np.timedelta64) -> np.ndarray:
    """
    Obtains the times of the wrf outputs
    """
    with open(namelist_input_path, "r") as f:
        text = f.read()
    history_interval = int(get_variable_namelist("history_interval", text)[2])
    start_time = get_start_time(namelist_input_path)
    run_time = get_run_time(namelist_input_path)
    end_time = start_time + run_time + np.timedelta64(history_interval, "m")
    history_interval_seconds = history_interval * 60
    all_times = np.arange(start_time, end_time, dtype="datetime64[s]")[::history_interval_seconds]
    spinup_end = start_time + spinup_time
    spin_up_indx = all_times <= spinup_end
    wrf_times = all_times[~spin_up_indx]

    return wrf_times

def load_radar_data(radar_path:Path, time_path:Path, start_time:np.datetime64, end_time:np.datetime64) -> np.ndarray:
    all_data = np.load(radar_path)
    all_times = np.load(time_path).astype('datetime64[ms]')
    mask = (all_times>=start_time) & (all_times<=end_time)
    return all_data[mask, :]

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