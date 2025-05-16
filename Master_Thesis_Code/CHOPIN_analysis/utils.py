from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from netCDF4 import Dataset  # type: ignore


def get_plot_start_time(first_time: pd.Timestamp) -> pd.Timestamp:
    year = first_time.year
    month = first_time.month
    day = first_time.day
    hour = first_time.hour
    start_time_hour = 0
    if hour >= 6:
        start_time_hour = 6
    if hour >= 12:
        start_time_hour = 12
    if hour >= 18:
        start_time_hour = 18

    return pd.Timestamp(year=year, month=month, day=day, hour=start_time_hour)


def get_plot_end_time(last_time: pd.Timestamp) -> pd.Timestamp:
    year = last_time.year
    month = last_time.month
    day = last_time.day
    hour = last_time.hour
    end_time_hour = 0
    add_day = False
    if hour < 18:
        end_time_hour = 18
    if hour < 12:
        end_time_hour = 12
    if hour < 6:
        end_time_hour = 6
    if hour >= 18:
        end_time_hour = 0
        add_day = True

    end_time = pd.Timestamp(year=year, month=month, day=day, hour=end_time_hour)
    if add_day:
        end_time += pd.Timedelta(1, "D")
    return end_time

def get_middled_geopotential_height(dataset: Dataset) -> np.ndarray:
    PHB = np.squeeze(dataset.variables["PHB"][0, :])
    PH = np.squeeze(dataset.variables["PH"][0, :])
    HGT = np.squeeze(dataset.variables["HGT"][0])
    ZZ = (PH + PHB) / 9.81 - HGT
    return np.diff(ZZ) / 2 + ZZ[:-1]

def set_up_dated_x_axis(axs: list[Axes], tick_locs, tick_labels):
    for ax in axs:
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([])
    
    axs[-1].set_xticklabels(tick_labels)
    axs[-1].set_xlabel("Date [UTC]")


def get_LWC(tk: np.ndarray, qvapor: np.ndarray, pressure: np.ndarray, cloud_water: np.ndarray):
    """
    Calculates the liquid water content or ice water content.
    """
    RA = 287.15
    EPS = 0.622
    tv = tk * (EPS + qvapor) / (EPS * (1.0 + qvapor))
    rho = pressure / RA / tv
    lwc = cloud_water * rho * 10**3
    lwc[lwc < 10**-7] = np.nan
    return lwc


def get_LWC_rh(tk: np.ndarray, rh: np.ndarray, pressure: np.ndarray, cloud_water: np.ndarray):
    """
    Calculates the liquid water content or ice water content using RH instead of 
    water vapour mixing ratio
    """
    RA = 287.15
    EPS = 0.622
    TP_T = 273.16
    TP_P = 611.657  # Pa
    Lv = 2260 * 1000  # J/kg
    Rv = 461.51  # JK−1kg−1
    e_sat = TP_P * np.exp(Lv / Rv * (1 / TP_T - 1 / tk))

    e = rh / 100 * e_sat
    qvapor = RA / Rv * e / (pressure * 100 - e)
    tv = tk * (EPS + qvapor) / (EPS * (1.0 + qvapor))
    rho = pressure / RA / tv
    lwc = cloud_water * rho * 10**3
    lwc[lwc < 10**-7] = np.nan
    return lwc