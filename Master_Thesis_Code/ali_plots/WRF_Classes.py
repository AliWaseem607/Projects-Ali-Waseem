import functools
import re
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset  # type: ignore
from scipy.special import gamma
from wrf import ALL_TIMES, getvar

# from utils import get_wrf_times


class WRFConstants:
    def __init__(
        self, 
        wrf_directory:Path,
    ) -> None:
        self.RA=287.15 # Ideal gas constant for air
        self.RD=287.0 # Idea gas constant for dry air?
        self.CP=1004.5 # Specific heat capcity for air
        self.P1000MB=100000.0 # Reference Pressure
        self.EPS=0.622 # Molar mass vapour/Molar mass dry air
        self.pi = 3.14159


        # Obtain parameters from module_mp_morr_two_moment.F
        with open(Path(wrf_directory, "phys", "module_mp_morr_two_moment.F")) as f:
            text = f.read()
        
        varaibles = [
            "NDCNST", 
            "LAMMINR", 
            "LAMMAXR",
            "LAMMINI",
            "LAMMAXI",
            "LAMMINS",
            "LAMMAXS",
            "LAMMING",
            "LAMMAXG",
            "RHOW",
            "RHOI",
            "RHOSN",
            "RHOG",

        ]
        values_dict = {}
        for variable in varaibles:            
            values_dict[variable] = self._find_variable(variable, text)


        self.NDCNST = values_dict["NDCNST"] # Constant Droplet Concentration cm-3
        self.LAMRMIN = values_dict["LAMMINR"]*10**-6 # Exponential size distribution parameter
        self.LAMRMAX = values_dict["LAMMAXR"]*10**-6 # Exponential size distribution parameter
        self.LAMIMIN = values_dict["LAMMINI"]*10**-6 # Exponential size distribution parameter
        self.LAMIMAX = values_dict["LAMMAXI"]*10**-6 # Exponential size distribution parameter
        self.LAMSMIN = values_dict["LAMMINS"]*10**-6 # Exponential size distribution parameter
        self.LAMSMAX = values_dict["LAMMAXS"]*10**-6 # Exponential size distribution parameter
        self.LAMGMIN = values_dict["LAMMING"]*10**-6 # Exponential size distribution parameter
        self.LAMGMAX = values_dict["LAMMAXG"]*10**-6 # Exponential size distribution parameter
        self.pice = values_dict["RHOI"]
        self.psnow = values_dict["RHOSN"]
        self.pgraupel = values_dict["RHOG"]
        self.prain = values_dict["RHOW"]
        self.pcloud = values_dict["RHOW"]

        self.heta = 0.0005714 * self.NDCNST + 0.2714
        self.miu = 1/(self.heta)**2 - 1
        self.miu = max(self.miu, 2)
        self.miu = min(self.miu, 10)
        self.alpha = self.pi*pcloud/6

        self.LAMCMIN = (self.miu+1)/(60) ; 
        self.LAMCMAX = (self.miu+1)  ##given in um-1
    
    def _find_variable(self, variable_name:str, module_text:str ) -> float:
        pattern = fr"\n.*[^!] *{variable_name} * = *([0-9a-zA-Z_.)(+/*-]+)"
        value = re.findall(pattern, module_text)[-1]
        assert isinstance(value, str)
        other_variable_check = re.finditer("([a-zA-Z]+)", value)
        if other_variable_check is not None:
            for var_result in other_variable_check:
                sub_var = var_result.groups()[0]
                if sub_var =="E" or sub_var == "e":
                    continue
                assert isinstance(sub_var, str)
                sub_value =  re.findall(fr"\n.*[^!] *{sub_var} * = *([0-9a-zA-Z_.)(+/*-]+)", module_text)[0]
                value = value.replace(sub_var, sub_value)

        return eval(value)

class SizeDistributionParameters:
    def __init__(self, Dp_min, Dp_max, npts):
        self.Dp_min = Dp_min
        self.Dp_max = Dp_max
        self.npts = npts
        self.logDp = np.linspace(np.log10(Dp_min), np.log10(Dp_max), num=npts)
        self.u8_arr = np.array([self.logDp[0], self.logDp[-1]])
        self.diff_u8 = np.diff(self.u8_arr)
        self.dlogDp = self.diff_u8/(npts-1)
        self.Dp = 10**self.logDp
        self.dDp = self.Dp*(10**self.dlogDp-1)



class WRFSimulationMeanSizeDistribution:
    def __init__(
            self, 
            wrfout_path:Path, 
            # namelist_path:Path, 
            # spinup_time:np.timedelta64,
            constants: WRFConstants,
        ) -> None:

        dataset = Dataset(wrfout_path)
        # time = get_wrf_times(namelist_path, spinup_time)

        # set up variables
        # assert ALL_TIMES is not None
        pres = np.squeeze(getvar(dataset,"pres",timeidx=ALL_TIMES, meta=False))
        tv = np.zeros(pres.shape)
        for i in range(pres.shape[0]):
            tv[i,:] = np.squeeze(getvar(dataset,"pres",timeidx=i, meta=False))

        rho = pres/constants.RA/tv
        lwc = np.squeeze(dataset.variables["QCLOUD"]) + np.squeeze(dataset.variables["QRAIN"])
        lwc = lwc*rho*10**3
        iwc = (
                np.squeeze(dataset.variables["QICE"]) + 
                np.squeeze(dataset.variables["QSNOW"]) + 
                np.squeeze(dataset.variables["QGRAUP"])
                )
        iwc = iwc*rho*10**3
        ICNC = (
                np.squeeze(dataset.variables["QNICE"]) + 
                np.squeeze(dataset.variables["QNSNOW"]) + 
                np.squeeze(dataset.variables["QNGRAUPEL"])
                )
        ICNC = ICNC*rho*10**-3  # L-1
        NICE = np.squeeze(dataset.variables["QNICE"], ALL_TIMES)*rho # kg/m3
        QICE = np.squeeze(dataset.variables["QICE"], ALL_TIMES)*rho # kg/m3
        NGRAUPEL = np.squeeze(dataset.variables["QNGRAUPEL"], ALL_TIMES)*rho # kg/m3
        QGRAUPEL = np.squeeze(dataset.variables["QGRAUP"], ALL_TIMES)*rho # kg/m3
        NSNOW = np.squeeze(dataset.variables["QNSNOW"], ALL_TIMES)*rho # kg/m3
        QSNOW = np.squeeze(dataset.variables["QSNOW"], ALL_TIMES)*rho # kg/m3
        NRAIN = np.squeeze(dataset.variables["QNRAIN"], ALL_TIMES)*rho # kg/m3
        QRAIN = np.squeeze(dataset.variables["QRAIN"], ALL_TIMES)*rho # kg/m3
        QCLOUD = np.squeeze(dataset.variables["QCLOUD"], ALL_TIMES)*rho # kg/m3

        LAMBDA_I = np.zeros(pres.shape)
        N0_I = np.zeros(pres.shape)
        LAMBDA_S = np.zeros(pres.shape)
        N0_S = np.zeros(pres.shape) 
        LAMBDA_G = np.zeros(pres.shape)
        N0_G = np.zeros(pres.shape)
        LAMBDA_R = np.zeros(pres.shape)
        N0_R = np.zeros(pres.shape)
        LAMBDA_C = np.zeros(pres.shape)
        N0_C = np.zeros(pres.shape)

        LAMBDA_I[QICE<=0] = np.nan
        N0_I[QICE<=0] = np.nan
        mask = QICE>0
        LAMBDA_I[mask] = ((constants.pi*constants.pice*NICE[mask]/QICE[mask])**(1/3))*10**(-6) # um-1
        np.clip(LAMBDA_I, a_min=constants.LAMIMIN, a_max=constants.LAMIMAX, out=LAMBDA_I)
        N0_I = NICE*LAMBDA_I*10**(-6) # cm-3 um-1

        LAMBDA_S[QSNOW<=0] = np.nan
        N0_S[QSNOW<=0] = np.nan
        mask = QSNOW>0
        LAMBDA_S[mask] = ((constants.pi*constants.psnow*NSNOW[mask]/QSNOW[mask])**(1/3))*10**(-6) # um-1
        np.clip(LAMBDA_S, a_min=constants.LAMSMIN, a_max=constants.LAMSMAX, out=LAMBDA_S)
        N0_S = NSNOW*LAMBDA_I*10**(-6) # cm-3 um-1

        LAMBDA_G[QGRAUPEL<0] = np.nan
        N0_G[QGRAUPEL<0] = np.nan
        mask = QGRAUPEL>0
        LAMBDA_G[mask] = ((constants.pi*constants.pgraupel*NGRAUPEL[mask]/QGRAUPEL[mask])**(1/3))*10**(-6) # um-1
        np.clip(LAMBDA_G, a_min=constants.LAMGMIN, a_max=constants.LAMGMAX, out=LAMBDA_G)
        N0_G = NGRAUPEL*LAMBDA_G*10**(-6) # cm-3 um-1

        LAMBDA_R[QRAIN<=0] = np.nan
        N0_R[QRAIN<=0] = np.nan
        mask = QRAIN>0
        LAMBDA_R[mask] = ((constants.pi*constants.prain*NRAIN[mask]/QRAIN[mask])**(1/3))*10**(-6) # um-1
        np.clip(LAMBDA_R, a_min=constants.LAMRMIN, a_max=constants.LAMRMAX, out=LAMBDA_R)
        N0_R = NRAIN*LAMBDA_R*10**(-6) # cm-3 um-1

        LAMBDA_C[QCLOUD<=0] = np.nan
        N0_C[QCLOUD<=0] = np.nan
        mask = QCLOUD>0
        LAMBDA_C[mask] = (
                    (
                        (constants.alpha*constants.NDCNST*(10**6)*gamma(constants.miu+4))/
                        (QCLOUD[mask]*gamma(constants.miu+1))
                    )**(1/3))*10**(-6) #um-1
        np.clip(LAMBDA_I, a_min=constants.LAMCMIN, a_max=constants.LAMCMAX, out=LAMBDA_C)
        N0_C = (constants.NDCNST*(10**6)*(LAMBDA_C**(constants.miu+1))/gamma(constants.miu+1))*(10**(-6)) #cm-3 um-1


        # Assign variables to self
        self.netCDF = dataset
        self.pres = pres 
        self.tv = tv 
        self.rho = rho 
        self.lwc = lwc 
        self.iwc = iwc 
        self.ICNC = ICNC 
        self.NICE = NICE 
        self.QICE = QICE 
        self.NGRAUPEL = NGRAUPEL 
        self.QGRAUPEL = QGRAUPEL 
        self.NSNOW = NSNOW 
        self.QSNOW = QSNOW 
        self.NRAIN = NRAIN 
        self.QRAIN = QRAIN 
        self.QCLOUD = QCLOUD 
        self.LAMBDA_I = LAMBDA_I 
        self.N0_I = N0_I 
        self.LAMBDA_S = LAMBDA_S 
        self.N0_S = N0_S 
        self.LAMBDA_G = LAMBDA_G 
        self.N0_G = N0_G 
        self.LAMBDA_R = LAMBDA_R 
        self.N0_R = N0_R 
        self.LAMBDA_C = LAMBDA_C 
        self.N0_C = N0_C 
    
    

WRFConstants(Path("/home/waseem/Helmos_all_sip/WRF"))
# WRFConstants(Path("/home/waseem/Helmos_control/WRF"))
# WRFConstants(Path("/home/waseem/Helmos_control_demott/WRF"))