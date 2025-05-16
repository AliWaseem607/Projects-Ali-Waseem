# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import glob
import warnings
from pathlib import Path

import cmasher as cmr
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from netCDF4 import Dataset  # type: ignore
from wrf import ALL_TIMES, getvar, to_np

from utils import get_wrf_times


def plot_microphysics(
    CONTROL: Dataset,
    DEMOTT: Dataset,
    ALLSIP: Dataset,
    namelist_path: Path,
    spinup_time: np.timedelta64,
    save_path: Path,
    wprof_data_path: Path,
) -> None:
    """
    Main function to plot the microphysics in the model. outputs figures 2 and 3 from vivi's paper
    control, demott, and allsip datapaths are the paths to the extracted files for the radar locattion.
    spin up time is the time delta between the first time and the first time for the simulation.
    save path is where the figures will be saved
    wprof data path is the path to the datafolder which contains the wprof data.
    """
    assert ALL_TIMES is not None

    wrf_time = get_wrf_times(namelist_path, spinup_time=spinup_time)
    tick_start_time = pd.Timestamp(wrf_time[0]).to_pydatetime() - datetime.timedelta(hours=4)
    tick_end_time = pd.Timestamp(wrf_time[-1]).to_pydatetime() + datetime.timedelta(hours=2)

    tick_locs = mdates.drange(tick_start_time, tick_end_time, datetime.timedelta(hours=6))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

    ### WRF constants
    RA = 287.15
    RD = 287.0
    CP = 1004.5
    P1000MB = 100000.0
    EPS = 0.622

    presCONTROL = np.squeeze(
        CONTROL.variables["P"][:] + CONTROL.variables["PB"][:]
    )  # pressure and purtubation pressure
    thetCONTROL = np.squeeze(CONTROL.variables["T"][:] + 300.0)  # potential temperature
    qvCONTROL = np.squeeze(CONTROL.variables["QVAPOR"][:])
    tkCONTROL = (presCONTROL / P1000MB) ** (RD / CP) * thetCONTROL
    tCONTROL = tkCONTROL - 273.15
    tvCONTROL = tkCONTROL * (EPS + qvCONTROL) / (EPS * (1.0 + qvCONTROL))  # virtual temperature
    rhoCONTROL = presCONTROL / RA / tvCONTROL
    icncCONTROL = (
        np.squeeze(
            (
                CONTROL.variables["QNICE"][:]
                + CONTROL.variables["QNSNOW"][:]
                + CONTROL.variables["QNGRAUPEL"][:]
            )
        )
        * rhoCONTROL
        * 10**-3
    )  # L-1
    iwcCONTROL = (
        np.squeeze(
            (CONTROL.variables["QICE"][:] + CONTROL.variables["QSNOW"][:] + CONTROL.variables["QGRAUP"][:])
        )
        * rhoCONTROL
        * 10**3
    )  # gm-3
    lwcCONTROL = (
        np.squeeze((CONTROL.variables["QCLOUD"][:] + CONTROL.variables["QRAIN"][:])) * rhoCONTROL * 10**3
    )  # gm-3
    brCONTROL = np.squeeze(CONTROL.variables["DNI_BR"][:]) * rhoCONTROL * 10**-3  # L-1s-1
    hmCONTROL = np.squeeze(CONTROL.variables["DNI_HM"][:]) * rhoCONTROL * 10**-3  # L-1s-1
    dsCONTROL = (
        np.squeeze(
            (CONTROL.variables["DNI_DS1"][:])
            + (CONTROL.variables["DNI_DS2"][:])
            + (CONTROL.variables["DNS_BF1"][:])
            + (CONTROL.variables["DNG_BF1"][:])
        )
        * rhoCONTROL
        * 10**-3
    )  # L-1s-1
    sbCONTROL = (
        np.squeeze((CONTROL.variables["DNI_SBS"][:]) + (CONTROL.variables["DNI_SBG"][:]))
        * rhoCONTROL
        * 10**-3
    )  # L-1s-1
    pipCONTROL = (
        np.squeeze(
            (CONTROL.variables["DNI_CON"][:])
            + (CONTROL.variables["DNI_IMM"][:])
            + (CONTROL.variables["DNI_NUC"][:])
            + (CONTROL.variables["DNS_CCR"][:])
        )
        * rhoCONTROL
        * 10**-3
    )  # L-1s-1
    aggCONTROL = abs(np.squeeze(CONTROL.variables["DNS_AGG"][:])) * rhoCONTROL * 10**-3  # L-1s-1
    rimCONTROL = np.squeeze(CONTROL.variables["DQC_RIM"][:]) * rhoCONTROL * 10**3  # gm-3s-1
    depCONTROL = (
        np.squeeze(
            CONTROL.variables["DQI_DEP"][:]
            + CONTROL.variables["DQS_DEP"][:]
            + CONTROL.variables["DQG_DEP"][:]
        )
        * rhoCONTROL
        * 10**3
    )  # gm-3s-1
    rhiCONTROL = np.squeeze(CONTROL.variables["DSI_TEN"][:]) * 100
    pblhCONTROL = np.squeeze(CONTROL.variables["PBLH"][:]) / 1000  # PBLH in km

    PHB = np.squeeze(CONTROL.variables["PHB"][0, :])
    PH = np.squeeze(CONTROL.variables["PH"][0, :])
    HGT = np.squeeze(CONTROL.variables["HGT"][0])
    ZZASL = (PH + PHB) / 9.81
    ZZ = (PH + PHB) / 9.81 - HGT
    ZZ_km = ZZ / 1000
    dz = np.zeros((len(ZZ) - 1))
    ZZmiddle = np.zeros((len(ZZ) - 1))

    kk = 0

    for jj in range(len(ZZ) - 1):
        dz[kk] = (ZZ[kk + 1] - ZZ[kk]) / 2
        ZZmiddle[kk] = dz[kk] + ZZ[kk]
        kk = kk + 1

    presDEMOTT = np.squeeze(DEMOTT.variables["P"][:] + DEMOTT.variables["PB"][:])
    thetDEMOTT = np.squeeze(DEMOTT.variables["T"][:] + 300.0)
    qvDEMOTT = np.squeeze(DEMOTT.variables["QVAPOR"][:])
    tkDEMOTT = (presDEMOTT / P1000MB) ** (RD / CP) * thetDEMOTT
    tDEMOTT = tkDEMOTT - 273.15
    tvDEMOTT = tkDEMOTT * (EPS + qvDEMOTT) / (EPS * (1.0 + qvDEMOTT))
    rhoDEMOTT = presDEMOTT / RA / tvDEMOTT
    icncDEMOTT = (
        np.squeeze(
            (DEMOTT.variables["QNICE"][:] + DEMOTT.variables["QNSNOW"][:] + DEMOTT.variables["QNGRAUPEL"][:])
        )
        * rhoDEMOTT
        * 10**-3
    )  # L-1
    iwcDEMOTT = (
        np.squeeze(
            (DEMOTT.variables["QICE"][:] + DEMOTT.variables["QSNOW"][:] + DEMOTT.variables["QGRAUP"][:])
        )
        * rhoDEMOTT
        * 10**3
    )  # gm-3
    lwcDEMOTT = (
        np.squeeze((DEMOTT.variables["QCLOUD"][:] + DEMOTT.variables["QRAIN"][:])) * rhoDEMOTT * 10**3
    )  # gm-3
    brDEMOTT = np.squeeze(DEMOTT.variables["DNI_BR"][:]) * rhoDEMOTT * 10**-3  # L-1s-1
    hmDEMOTT = np.squeeze(DEMOTT.variables["DNI_HM"][:]) * rhoDEMOTT * 10**-3  # L-1s-1
    dsDEMOTT = (
        np.squeeze(
            (DEMOTT.variables["DNI_DS1"][:])
            + (DEMOTT.variables["DNI_DS2"][:])
            + (DEMOTT.variables["DNS_BF1"][:])
            + (DEMOTT.variables["DNG_BF1"][:])
        )
        * rhoDEMOTT
        * 10**-3
    )  # L-1s-1
    sbDEMOTT = (
        np.squeeze((DEMOTT.variables["DNI_SBS"][:]) + (DEMOTT.variables["DNI_SBG"][:])) * rhoDEMOTT * 10**-3
    )  # L-1s-1
    pipDEMOTT = (
        np.squeeze(
            (DEMOTT.variables["DNI_CON"][:])
            + (DEMOTT.variables["DNI_IMM"][:])
            + (DEMOTT.variables["DNI_NUC"][:])
            + (DEMOTT.variables["DNS_CCR"][:])
        )
        * rhoDEMOTT
        * 10**-3
    )  # L-1s-1
    aggDEMOTT = abs(np.squeeze(DEMOTT.variables["DNS_AGG"][:])) * rhoDEMOTT * 10**-3  # L-1s-1
    rimDEMOTT = np.squeeze(DEMOTT.variables["DQC_RIM"][:]) * rhoDEMOTT * 10**3  # gm-3s-1
    depDEMOTT = (
        np.squeeze(
            DEMOTT.variables["DQI_DEP"][:] + DEMOTT.variables["DQS_DEP"][:] + DEMOTT.variables["DQG_DEP"][:]
        )
        * rhoDEMOTT
        * 10**3
    )  # gm-3s-1
    rhiDEMOTT = np.squeeze(DEMOTT.variables["DSI_TEN"][:]) * 100
    CLDFRADMT = np.squeeze((DEMOTT.variables["CLDFRA"][:]))
    pblhDEMOTT = np.squeeze(DEMOTT.variables["PBLH"][:]) / 1000  # PBLH in km

    PHB2 = np.squeeze(DEMOTT.variables["PHB"][0, :])
    PH2 = np.squeeze(DEMOTT.variables["PH"][0, :])
    HGT2 = np.squeeze(DEMOTT.variables["HGT"][0])
    ZZASL2 = (PH2 + PHB2) / 9.81
    ZZ2 = (PH2 + PHB2) / 9.81 - HGT2
    ZZ_km2 = ZZ2 / 1000
    dz2 = np.zeros((len(ZZ2) - 1))
    ZZmiddle2 = np.zeros((len(ZZ2) - 1))

    kk = 0

    for jj in range(len(ZZ2) - 1):
        dz2[kk] = (ZZ2[kk + 1] - ZZ2[kk]) / 2
        ZZmiddle2[kk] = dz2[kk] + ZZ2[kk]
        kk = kk + 1

    presALLSIP = np.squeeze(ALLSIP.variables["P"][:] + ALLSIP.variables["PB"][:])
    thetALLSIP = np.squeeze(ALLSIP.variables["T"][:] + 300.0)
    qvALLSIP = np.squeeze(ALLSIP.variables["QVAPOR"][:])
    tkALLSIP = (presALLSIP / P1000MB) ** (RD / CP) * thetALLSIP
    tALLSIP = tkALLSIP - 273.15
    tvALLSIP = tkALLSIP * (EPS + qvALLSIP) / (EPS * (1.0 + qvALLSIP))
    rhoALLSIP = presALLSIP / RA / tvALLSIP
    icncALLSIP = (
        np.squeeze(
            (ALLSIP.variables["QNICE"][:] + ALLSIP.variables["QNSNOW"][:] + ALLSIP.variables["QNGRAUPEL"][:])
        )
        * rhoALLSIP
        * 10**-3
    )  # L-1
    iwcALLSIP = (
        np.squeeze(
            (ALLSIP.variables["QICE"][:] + ALLSIP.variables["QSNOW"][:] + ALLSIP.variables["QGRAUP"][:])
        )
        * rhoALLSIP
        * 10**3
    )  # gm-3
    lwcALLSIP = (
        np.squeeze((ALLSIP.variables["QCLOUD"][:] + ALLSIP.variables["QRAIN"][:])) * rhoALLSIP * 10**3
    )  # gm-3
    brALLSIP = np.squeeze(ALLSIP.variables["DNI_BR"][:]) * rhoALLSIP * 10**-3  # L-1s-1
    hmALLSIP = np.squeeze(ALLSIP.variables["DNI_HM"][:]) * rhoALLSIP * 10**-3  # L-1s-1
    dsALLSIP = (
        np.squeeze(
            (ALLSIP.variables["DNI_DS1"][:])
            + (ALLSIP.variables["DNI_DS2"][:])
            + (ALLSIP.variables["DNS_BF1"][:])
            + (ALLSIP.variables["DNG_BF1"][:])
        )
        * rhoALLSIP
        * 10**-3
    )  # L-1s-1
    sbALLSIP = (
        np.squeeze((ALLSIP.variables["DNI_SBS"][:]) + (ALLSIP.variables["DNI_SBG"][:])) * rhoALLSIP * 10**-3
    )  # L-1s-1
    pipALLSIP = (
        np.squeeze(
            (ALLSIP.variables["DNI_CON"][:])
            + (ALLSIP.variables["DNI_IMM"][:])
            + (ALLSIP.variables["DNI_NUC"][:])
            + (ALLSIP.variables["DNS_CCR"][:])
        )
        * rhoALLSIP
        * 10**-3
    )  # L-1s-1
    aggALLSIP = abs(np.squeeze(ALLSIP.variables["DNS_AGG"][:])) * rhoALLSIP * 10**-3  # L-1s-1
    rimALLSIP = np.squeeze(ALLSIP.variables["DQC_RIM"][:]) * rhoALLSIP * 10**3  # gm-3s-1
    depALLSIP = (
        np.squeeze(
            ALLSIP.variables["DQI_DEP"][:] + ALLSIP.variables["DQS_DEP"][:] + ALLSIP.variables["DQG_DEP"][:]
        )
        * rhoALLSIP
        * 10**3
    )  # gm-3s-1
    rhiALLSIP = np.squeeze(ALLSIP.variables["DSI_TEN"][:]) * 100
    pblhALLSIP = np.squeeze(ALLSIP.variables["PBLH"][:]) / 1000  # PBLH in km

    PHB3 = np.squeeze(ALLSIP.variables["PHB"][0, :])
    PH3 = np.squeeze(ALLSIP.variables["PH"][0, :])
    HGT3 = np.squeeze(ALLSIP.variables["HGT"][0])
    ZZASL3 = (PH3 + PHB3) / 9.81
    ZZ3 = (PH3 + PHB3) / 9.81 - HGT3
    ZZ_km3 = ZZ3 / 1000
    dz3 = np.zeros((len(ZZ3) - 1))
    ZZmiddle3 = np.zeros((len(ZZ3) - 1))

    kk = 0

    for jj in range(len(ZZ3) - 1):
        dz3[kk] = (ZZ3[kk + 1] - ZZ3[kk]) / 2
        ZZmiddle3[kk] = dz3[kk] + ZZ3[kk]
        kk = kk + 1

    icncCONTROL[icncCONTROL <= 10 ** (-5)] = np.nan
    lwcCONTROL[lwcCONTROL <= 10 ** (-6)] = np.nan
    iwcCONTROL[iwcCONTROL <= 10 ** (-6)] = np.nan
    pipCONTROL[pipCONTROL <= 0] = np.nan
    aggCONTROL[aggCONTROL <= 0] = np.nan
    zstagCONTROL = np.squeeze(getvar(CONTROL, "zstag", timeidx=ALL_TIMES))
    zstagCONTROL = zstagCONTROL[:]
    dzCONTROL = np.diff(zstagCONTROL, axis=1)
    lwpCONTROL = np.nansum(lwcCONTROL * dzCONTROL, axis=1)

    icncDEMOTT[icncDEMOTT <= 10 ** (-5)] = np.nan
    lwcDEMOTT[lwcDEMOTT <= 10 ** (-6)] = np.nan
    iwcDEMOTT[iwcDEMOTT <= 10 ** (-6)] = np.nan
    pipDEMOTT[pipDEMOTT <= 0] = np.nan
    aggDEMOTT[aggDEMOTT <= 0] = np.nan
    zstagDEMOTT = np.squeeze(getvar(DEMOTT, "zstag", timeidx=ALL_TIMES))
    zstagDEMOTT = zstagDEMOTT[:]
    dzDEMOTT = np.diff(zstagDEMOTT, axis=1)
    lwpDEMOTT = np.nansum(lwcDEMOTT * dzDEMOTT, axis=1)

    icncALLSIP[icncALLSIP <= 10 ** (-5)] = np.nan
    lwcALLSIP[lwcALLSIP <= 10 ** (-6)] = np.nan
    iwcALLSIP[iwcALLSIP <= 10 ** (-6)] = np.nan
    pipALLSIP[pipALLSIP <= 0] = np.nan
    aggALLSIP[aggALLSIP <= 0] = np.nan
    zstagALLSIP = np.squeeze(getvar(ALLSIP, "zstag", timeidx=ALL_TIMES))
    zstagALLSIP = zstagALLSIP[:]
    dzALLSIP = np.diff(zstagALLSIP, axis=1)
    lwpALLSIP = np.nansum(lwcALLSIP * dzALLSIP, axis=1)

    ###Remove spin-up time
    icncALLSIP = icncALLSIP[-len(wrf_time) :]
    lwcALLSIP = lwcALLSIP[-len(wrf_time) :]
    tALLSIP = tALLSIP[-len(wrf_time) :]
    iwcALLSIP = iwcALLSIP[-len(wrf_time) :]
    brALLSIP = brALLSIP[-len(wrf_time) :]
    hmALLSIP = hmALLSIP[-len(wrf_time) :]
    dsALLSIP = dsALLSIP[-len(wrf_time) :]
    sbALLSIP = sbALLSIP[-len(wrf_time) :]
    pipALLSIP = pipALLSIP[-len(wrf_time) :]
    aggALLSIP = aggALLSIP[-len(wrf_time) :]
    rimALLSIP = rimALLSIP[-len(wrf_time) :]
    depALLSIP = depALLSIP[-len(wrf_time) :]
    rhiALLSIP = rhiALLSIP[-len(wrf_time) :]
    lwpALLSIP = lwpALLSIP[-len(wrf_time) :]
    pblhALLSIP = pblhALLSIP[-len(wrf_time) :]

    icncCONTROL = icncCONTROL[-len(wrf_time) :]
    lwcCONTROL = lwcCONTROL[-len(wrf_time) :]
    tCONTROL = tCONTROL[-len(wrf_time) :]
    iwcCONTROL = iwcCONTROL[-len(wrf_time) :]
    brCONTROL = brCONTROL[-len(wrf_time) :]
    hmCONTROL = hmCONTROL[-len(wrf_time) :]
    dsCONTROL = dsCONTROL[-len(wrf_time) :]
    sbCONTROL = sbCONTROL[-len(wrf_time) :]
    pipCONTROL = pipCONTROL[-len(wrf_time) :]
    aggCONTROL = aggCONTROL[-len(wrf_time) :]
    rimCONTROL = rimCONTROL[-len(wrf_time) :]
    depCONTROL = depCONTROL[-len(wrf_time) :]
    rhiCONTROL = rhiCONTROL[-len(wrf_time) :]
    lwpCONTROL = lwpCONTROL[-len(wrf_time) :]
    pblhCONTROL = pblhCONTROL[-len(wrf_time) :]

    icncDEMOTT = icncDEMOTT[-len(wrf_time) :]
    lwcDEMOTT = lwcDEMOTT[-len(wrf_time) :]
    tDEMOTT = tDEMOTT[-len(wrf_time) :]
    iwcDEMOTT = iwcDEMOTT[-len(wrf_time) :]
    brDEMOTT = brDEMOTT[-len(wrf_time) :]
    hmDEMOTT = hmDEMOTT[-len(wrf_time) :]
    dsDEMOTT = dsDEMOTT[-len(wrf_time) :]
    sbDEMOTT = sbDEMOTT[-len(wrf_time) :]
    pipDEMOTT = pipDEMOTT[-len(wrf_time) :]
    aggDEMOTT = aggDEMOTT[-len(wrf_time) :]
    rimDEMOTT = rimDEMOTT[-len(wrf_time) :]
    depDEMOTT = depDEMOTT[-len(wrf_time) :]
    rhiDEMOTT = rhiDEMOTT[-len(wrf_time) :]
    CLDFRADMT = CLDFRADMT[-len(wrf_time) :]
    lwpDEMOTT = lwpDEMOTT[-len(wrf_time) :]
    pblhDEMOTT = pblhDEMOTT[-len(wrf_time) :]

    ########################
    # W-prof & LWP data    #
    ########################
    f = str(wprof_data_path) + "/*ZEN_LV2*"
    # Change this path

    Skewness_plot = None
    wprof_time = np.array([])
    LWP = None

    for element in sorted(glob.glob(f)):
        nc = Dataset(element)
        dt = [datetime.datetime.fromtimestamp(tt) for tt in nc.variables["Time"]]

        if dt[-1] < datetime.datetime(2021, 12, 17, 20, 40):
            continue

        if dt[-1] > datetime.datetime(2021, 12, 19, 13, 00):
            continue

        thresh = 30

        Rgates = nc.variables["Rgate"][:] / 1000
        Skew = nc.variables["Spectral-skewness"][:]
        ql = nc.variables["Liquid-water-path"][:]  # gm-2
        SNR = 10 * np.log10(nc.variables["Ze"][:] / nc.variables["Linear-sensitivity-vert"][:])

        Skew[SNR < thresh] = np.nan

        if Skewness_plot is None:
            Skewness_plot = Skew
        else:
            Skewness_plot = np.concatenate((Skewness_plot, Skew), axis=0)

        if LWP is None:
            LWP = ql
        else:
            LWP = np.concatenate((LWP, ql), axis=0)

        wprof_time = np.append(wprof_time, dt)  # type: ignore

    radar_time = np.array(wprof_time, dtype="datetime64[s]")

    ### Plotting ###
    plt.rc("font", size=14)
    plt.rc("axes", titlesize=14)
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    plt.rc("legend", fontsize=14)
    plt.rc("figure", titlesize=14)

    fig, axs = plt.subplots(4, 1, figsize=(14, 14))

    namefigure = str(Path(save_path, "/Figure2.png"))

    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.95, hspace=0.20)

    levs = np.logspace(-2, 3, 10)
    br = 1e-2  # L-1s-1
    hm = 1e-4  # L-1s-1
    br2 = 1e-3  # L-1s-1
    agg = 1e-5  # L-1s-1
    rim = 1e-5  # gm-3s-1
    sb = 1e-4  # L-1s-1
    ds = 1e-5  # L-1s-1

    br_label = r"BR$_{rate}$ > 10$^{-2}$ [L$^{-1}$ s$^{-1}$]"
    hm_label = r"HM$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
    br_label2 = r"BR$_{rate}$ > 10$^{-3}$ [L$^{-1}$ s$^{-1}$]"
    sb_label = r"SUBBR$_{rate}$ > 10$^{-4}$ [L$^{-1}$ s$^{-1}$]"
    ds_label = r"DS$_{rate}$ > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
    agg_label = r"Aggregation > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"

    icnc = axs[0].contourf(wrf_time, ZZmiddle / 1000, to_np(icncCONTROL.T), levs, norm=colors.LogNorm())
    agg_contour = axs[0].contour(
        wrf_time, ZZmiddle / 1000, aggCONTROL.T, levels=[agg], colors="red", linewidths=2
    )
    rhi = axs[0].contour(
        wrf_time,
        ZZmiddle / 1000,
        rhiCONTROL.T,
        levels=[100.1],
        colors="k",
        linewidths=0.8,
        linestyles="solid",
    )
    # rhi.collections[0].set_hatch('////')
    cs = axs[0].contour(
        wrf_time, ZZmiddle / 1000, tCONTROL.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    axs[0].clabel(cs, inline=True, fontsize=12, fmt="%d$^\circ$C", colors="dimgrey")  # type: ignore
    cbar = fig.colorbar(icnc, ax=axs[0], aspect=15)
    cbar.ax.set_yscale("log")
    cbar.set_label("ICNC CONTROL [$\mathrm{L^{-1}}$]", fontsize=16)  # type: ignore
    axs[0].set_ylabel("Altitude [km]")
    axs[0].set_ylim(0, 5)
    axs[0].set_yticks([0, 1, 2, 3, 4, 5])
    axs[0].set_xticks(tick_locs)
    axs[0].set_xticklabels([])
    axs[0].set_xlim(wrf_time[0], wrf_time[-1])
    br_handle = Line2D([], [], color="darkviolet", linewidth=2)
    hm_handle = Line2D([], [], color="darkcyan", linewidth=2)
    br_handle2 = Line2D([], [], color="darkviolet", linestyle="--", linewidth=2)
    sb_handle = Line2D([], [], color="magenta", linewidth=2)
    ds_handle = Line2D([], [], color="cyan", linewidth=2)
    agg_handle = Line2D([], [], color="red", linewidth=2)
    axs[0].legend(
        handles=[br_handle, br_handle2, sb_handle, ds_handle, agg_handle],
        labels=[br_label, br_label2, sb_label, ds_label, agg_label],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        frameon=False,
    )  # labelcolor='white', bbox_to_anchor=(1.11, 0.92)
    text_loc = np.datetime64("2021-12-19T10:30:00")
    text = axs[0].text(text_loc, 4.5, "(a)", fontsize=14)

    icnc = axs[1].contourf(wrf_time, ZZmiddle2 / 1000, to_np(icncDEMOTT.T), levs, norm=colors.LogNorm())
    cs = axs[1].contour(
        wrf_time, ZZmiddle2 / 1000, tDEMOTT.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    agg_contour = axs[1].contour(
        wrf_time, ZZmiddle2 / 1000, aggDEMOTT.T, levels=[agg], colors="red", linewidths=2
    )
    axs[1].clabel(cs, inline=True, fontsize=12, fmt="%d$^\circ$C", colors="dimgrey")  # type: ignore
    cbar = fig.colorbar(icnc, ax=axs[1], aspect=15)
    cbar.ax.set_yscale("log")
    cbar.set_label("ICNC DEMOTT [$\mathrm{L^{-1}}$]", fontsize=16)  # type: ignore
    axs[1].set_ylabel("Altitude [km]")
    axs[1].set_ylim(0, 5)
    axs[1].set_yticks([0, 1, 2, 3, 4, 5])
    axs[1].set_xticks(tick_locs)
    axs[1].set_xticklabels([])
    axs[1].set_xlim(wrf_time[0], wrf_time[-1])
    text = axs[1].text(text_loc, 4.5, "(b)", fontsize=14)

    icnc = axs[2].contourf(wrf_time, ZZmiddle3 / 1000, to_np(icncALLSIP.T), levs, norm=colors.LogNorm())
    br_contour = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, brALLSIP.T, levels=[br], colors="darkviolet", linewidths=2
    )
    hm_contour = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, hmALLSIP.T, levels=[hm], colors="darkcyan", linewidths=2
    )
    br_contour2 = axs[2].contour(
        wrf_time,
        ZZmiddle3 / 1000,
        brALLSIP.T,
        levels=[br2],
        colors="darkviolet",
        linewidths=2,
        linestyles="dashed",
    )
    sb_contour = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, sbALLSIP.T, levels=[sb], colors="magenta", linewidths=2
    )
    ds_contour = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, dsALLSIP.T, levels=[ds], colors="cyan", linewidths=2
    )
    cs = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, tALLSIP.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    axs[2].clabel(cs, inline=True, fontsize=12, fmt="%d$^\circ$C", colors="dimgrey")  # type: ignore
    cbar = fig.colorbar(icnc, ax=axs[2], aspect=15)
    cbar.ax.set_yscale("log")
    cbar.set_label("ICNC ALLSIP [$\mathrm{L^{-1}}$]", fontsize=16)  # type: ignore
    axs[2].set_xlim(wrf_time[0], wrf_time[-1])
    axs[2].set_ylabel("Altitude [km]")
    axs[2].set_ylim(0, 5)
    axs[2].set_yticks([0, 1, 2, 3, 4, 5])
    axs[2].set_xticklabels([])
    axs[2].set_xlim(wrf_time[0], wrf_time[-1])
    text = axs[2].text(text_loc, 4.5, "(c)", fontsize=14)

    cs = axs[3].contour(
        wrf_time, ZZmiddle / 1000, (tCONTROL.T), levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    im4 = axs[3].pcolormesh(wprof_time, Rgates, Skewness_plot.T, vmin=-1, vmax=1, cmap="seismic") #type: ignore
    axs[3].clabel(cs, inline=True, fontsize=12, fmt="%d$^\circ$C", colors="dimgrey")  # type: ignore
    cbar = fig.colorbar(im4, ax=axs[3], aspect=15)
    cbar.set_label("Skewness", fontsize=16)
    axs[3].set_ylabel("Altitude" + "\n [km]")
    axs[3].set_ylim(0, 5)
    axs[3].set_yticks([0, 1, 2, 3, 4, 5])
    axs[3].set_xticks(tick_locs)
    axs[3].set_xticklabels(tick_labels)
    axs[3].set_xlabel("Time [UTC]")
    axs[3].grid()
    axs[3].set_xlim(wrf_time[0], wrf_time[-1])
    text = axs[3].text(text_loc, 4.5, "(d)", fontsize=14)

    # plt.show()

    fig.savefig(
        namefigure, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight"
    )

    cmap2 = cmr.get_sub_cmap("Blues", 0.15, 1.0)

    fig, axs = plt.subplots(4, 1, figsize=(14, 14))

    namefigure2 = str(Path(save_path, "/Figure3.png"))

    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.95, hspace=0.20)

    levs = np.logspace(-2, 3, 10)

    dep = 1e-5
    rim = 1e-5

    rim_label = r"Riming > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]"
    dep_label = r"Deposition > 10$^{-5}$ [g m$^{-3}$ s$^{-1}$]"

    lwc = axs[0].contourf(
        wrf_time, ZZmiddle / 1000, to_np(lwcCONTROL.T), levels=np.linspace(0, 1, 9), cmap=cmap2
    )
    rim_contour = axs[0].contour(
        wrf_time, ZZmiddle / 1000, rimCONTROL.T, levels=[rim], colors="#FFDB58", linewidths=2
    )
    dep_contour = axs[0].contour(
        wrf_time, ZZmiddle / 1000, depCONTROL.T, levels=[dep], colors="coral", linewidths=2
    )
    cs = axs[0].contour(
        wrf_time, ZZmiddle / 1000, tCONTROL.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    rim_handle = Line2D([], [], color="#FFDB58", linewidth=2)
    dep_handle = Line2D([], [], color="coral", linewidth=2)
    axs[0].clabel(cs, inline=True, fontsize=10, fmt="%d$^\circ$C", colors="dimgrey")  # type: ignore
    cbar = fig.colorbar(lwc, ax=axs[0], aspect=15)
    cbar.set_label("LWC CONTROL [$\mathrm{gm^{-3}}$]", fontsize=16)  # type: ignore
    cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
    axs[0].set_ylabel("Altitude [km]")
    axs[0].set_ylim(0, 5)
    axs[0].set_yticks([0, 1, 2, 3, 4, 5])
    axs[0].set_xticks(tick_locs)
    axs[0].set_xticklabels([])
    axs[0].set_xlim(wrf_time[0], wrf_time[-1])
    axs[0].legend(
        handles=[rim_handle, dep_handle],
        labels=[rim_label, dep_label],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.36, 1.25),
        frameon=False,
    )
    text_loc = np.datetime64("2021-12-19T10:30:00")
    text = axs[0].text(text_loc, 4.5, "(a)", fontsize=14)

    lwc = axs[1].contourf(
        wrf_time, ZZmiddle2 / 1000, to_np(lwcDEMOTT.T), levels=np.linspace(0, 1, 9), cmap=cmap2
    )
    rim_contour = axs[1].contour(
        wrf_time, ZZmiddle2 / 1000, rimDEMOTT.T, levels=[rim], colors="#FFDB58", linewidths=2
    )
    dep_contour = axs[1].contour(
        wrf_time, ZZmiddle2 / 1000, depDEMOTT.T, levels=[dep], colors="coral", linewidths=2
    )
    cs = axs[1].contour(
        wrf_time, ZZmiddle2 / 1000, tDEMOTT.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    axs[1].clabel(cs, inline=True, fontsize=10, fmt="%d$^\circ$C", colors="dimgrey")  # type: ignore
    cbar = fig.colorbar(lwc, ax=axs[1], aspect=15)
    cbar.set_label("LWC DEMOTT [$\mathrm{gm^{-3}}$]", fontsize=16)  # type: ignore
    cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
    axs[1].set_ylabel("Altitude [km]")
    axs[1].set_ylim(0, 5)
    axs[1].set_yticks([0, 1, 2, 3, 4, 5])
    axs[1].set_xticks(tick_locs)
    axs[1].set_xticklabels([])
    axs[1].set_xlim(wrf_time[0], wrf_time[-1])
    text = axs[1].text(text_loc, 4.5, "(b)", fontsize=14)

    lwc = axs[2].contourf(
        wrf_time, ZZmiddle3 / 1000, to_np(lwcALLSIP.T), levels=np.linspace(0, 1, 9), cmap=cmap2
    )
    rim_contour = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, rimALLSIP.T, levels=[rim], colors="#FFDB58", linewidths=2
    )
    dep_contour = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, depALLSIP.T, levels=[dep], colors="coral", linewidths=2
    )
    cs = axs[2].contour(
        wrf_time, ZZmiddle3 / 1000, tALLSIP.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    axs[2].clabel(cs, inline=True, fontsize=10, fmt="%d$^\circ$C", colors="dimgrey")  # type:ignore
    cbar = fig.colorbar(lwc, ax=axs[2], aspect=15)
    cbar.set_label("LWC ALLSIP [$\mathrm{gm^{-3}}$]", fontsize=16)  # type:ignore
    cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
    axs[2].set_ylabel("Altitude [km]")
    axs[2].set_ylim(0, 5)
    axs[2].set_yticks([0, 1, 2, 3, 4, 5])
    axs[2].set_xticks(tick_locs)
    axs[2].set_xticklabels([])
    axs[2].set_xlim(wrf_time[0], wrf_time[-1])
    text = axs[2].text(text_loc, 4.5, "(c)", fontsize=14)

    axs[3].plot(wprof_time, LWP, "-", color="darkgray", linewidth=2, label="Observations")
    axs[3].plot(wrf_time, lwpCONTROL, color="black", linewidth=2, label="CONTROL")
    axs[3].plot(wrf_time, lwpDEMOTT, color="cyan", linewidth=2, label="DEMOTT")
    axs[3].plot(wrf_time, lwpALLSIP, color="blue", linewidth=2, label="ALLSIP")
    cbar = fig.colorbar(cs, ax=axs[3], aspect=15)
    cbar.set_ticks(np.linspace(0, 1, 5))  # type: ignore
    cbar.ax.set_visible(False)
    axs[3].set_ylabel(r"LWP [gm$^{-2}$]")
    axs[3].set_ylim(0, 2000)
    axs[3].set_xlabel("Time [UTC]")
    axs[3].set_xticks(tick_locs)
    axs[3].set_xticklabels(tick_labels)
    axs[3].legend(ncol=4, loc="upper left")
    axs[3].grid()
    axs[3].set_xlim(wrf_time[0], wrf_time[-1])
    text = axs[3].text(text_loc, 1800, "(d)", fontsize=14)

    # plt.show()

    fig.savefig(
        namefigure2, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    levs = np.logspace(-2, 3, 10)

    namefigure3 = str(Path(save_path, "/FigureS5.png"))

    icnc = ax.contourf(wrf_time, ZZmiddle / 1000, to_np(icncALLSIP.T), levs, norm=colors.LogNorm())
    pbl = ax.plot(wrf_time, pblhALLSIP, marker="o", color="black", markerfacecolor="black", markersize="6")
    agg = 0.00001  # L-1s-1
    agg_contour = plt.contour(
        wrf_time, ZZmiddle / 1000, aggALLSIP.T, levels=[agg], colors="red", linewidths=2
    )
    rhi = plt.contour(
        wrf_time, ZZmiddle / 1000, rhiALLSIP.T, levels=[101], colors="k", linewidths=0.8, linestyles="solid"
    )
    rhi.collections[0].set_hatch("////")
    cs = plt.contour(
        wrf_time, ZZmiddle / 1000, tALLSIP.T, levels=np.arange(-50, 0, 5), colors="dimgray", linewidths=1
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt="%d$^\circ$C")  # type: ignore
    cbar = fig.colorbar(icnc, pad=0.01)
    cbar.ax.set_yscale("log")
    cbar.set_label("ICNC ALLSIP [$\mathrm{L^{-1}}$]", fontsize=16)  # type: ignore
    ax.set_xlim(wrf_time[0], wrf_time[-1])
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel("Time [UTC]")
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlim(wrf_time[0], wrf_time[-1])
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    agg_label = r"Aggregation > 10$^{-5}$ [L$^{-1}$ s$^{-1}$]"
    agg_handle = Line2D([], [], color="red", linewidth=2)
    plt.legend(
        handles=[agg_handle],
        labels=[agg_label],
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.15, 1.11),
        frameon=False,
    )
    plt.grid()
    plt.subplots_adjust(top=0.85, bottom=0.12, left=0.05, right=1.0, wspace=0.3)
    # plt.show()
    fig.savefig(
        namefigure3, dpi=300, format="png", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight"
    )


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


    plot_microphysics(
        CONTROL=CONTROL, 
        DEMOTT=DEMOTT, 
        ALLSIP=ALLSIP, 
        namelist_path=namelist_input, 
        spinup_time=spinup_time, 
        save_path=save_path,
        wprof_data_path=wprof_path)