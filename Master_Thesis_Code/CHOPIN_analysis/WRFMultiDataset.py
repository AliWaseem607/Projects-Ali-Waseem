# %%
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore
from wrf import getvar


class Station(Enum):
    HAC = "HAC"
    NPRK = "NPRK"
    SPRK = "SPRK"
    VL = "VL"


# %%


class WRFDataset:
    def __init__(self, ncfile_path: Path, spinup_hours: float = 24) -> None:
        self._ncfile = Dataset(ncfile_path)
        times = pd.Series(getvar(self._ncfile, "Times", meta=False, timeidx=None))  # type: ignore
        self._start_time = times.iloc[0] + pd.Timedelta(spinup_hours, "h")
        self._end_time = times.iloc[-1]
        self._mask = times >= self._start_time
        self._times = times[self._mask]
        self._mp_phys = self._ncfile.MP_PHYSICS
        self._bl_phys = self._ncfile.BL_PBL_PHYSICS

    @property
    def times(self) -> pd.Series:
        return self._times

    @property
    def ncfile(self) -> Dataset:
        return self._ncfile

    @property
    def start_time(self) -> pd.Timestamp:
        return self._start_time

    @property
    def end_time(self) -> pd.Timestamp:
        return self._end_time

    @property
    def mp_phys(self) -> int:
        return self._mp_phys

    @property
    def bl_phys(self) -> int:
        return self._bl_phys

    @cached_property
    def ZZ(self):
        PHB = np.squeeze(self._ncfile.variables["PHB"])[0, :]
        PH = np.squeeze(self._ncfile.variables["PH"])[0, :]
        HGT = np.squeeze(self._ncfile.variables["HGT"])[0]
        ZZ = (PH + PHB) / 9.81 - HGT

        # Clean up
        del PHB
        del PH
        del HGT

        return np.diff(ZZ) / 2 + ZZ[:-1]

    @cached_property
    def kinetic_temp(self):
        """
        The kinetic temperature in Kelvin
        """
        RD = 287.0
        CP = 1004.5
        P1000MB = 100000.0
        total_pressure = np.squeeze(
            self._ncfile.variables["P"][self._mask, :] + self._ncfile.variables["PB"][self._mask, :]
        )  # in mb
        potential_temp = np.squeeze(self._ncfile.variables["T"][self._mask, :] + 300.0)
        kinetic_temp = (total_pressure / P1000MB) ** (RD / CP) * potential_temp
        del total_pressure
        del potential_temp
        return kinetic_temp

    @cached_property
    def virtual_temp(self):
        """
        The virtual temperature in Kelvin
        """
        EPS = 0.622
        qvapor = np.squeeze(self._ncfile.variables["QVAPOR"][self._mask, :])
        virtual_temp = self.kinetic_temp * (EPS + qvapor) / (EPS * (1.0 + qvapor))
        del qvapor
        return virtual_temp

    @property
    def rho(self) -> np.ndarray:
        RA = 287.15
        total_pressure = np.squeeze(
            self._ncfile.variables["P"][self._mask, :] + self._ncfile.variables["PB"][self._mask, :]
        )  # in mb
        return total_pressure / RA / self.virtual_temp

    @cached_property
    def lwc(self):
        """
        liquid water content in g/m3
        """
        lwc = (
            np.squeeze(
                self._ncfile.variables["QCLOUD"][self._mask, :]
                + self._ncfile.variables["QRAIN"][self._mask, :]
            )
            * self.rho
            * 10**3
        )  # gm-3
        lwc[lwc <= 10 ** (-5)] = np.nan
        return lwc

    @cached_property
    def lwp(self):
        """
        liquid water path
        """
        # RA=287.15
        # total_pressure = np.squeeze(self._ncfile.variables['P'] + self._ncfile.variables['PB']) #in mb
        # rho = total_pressure/RA/self.virtual_temp
        # lwc = np.squeeze(self._ncfile.variables['QCLOUD'] + self._ncfile.variables['QRAIN'])*rho*10**3 #gm-3
        # lwc[lwc <= 10**(-5)] = np.nan
        zstag = np.squeeze(getvar(self._ncfile, "zstag", timeidx=None, meta=False)[self._mask, :])  # type: ignore
        dz = np.diff(zstag, axis=1)
        lwp = np.nansum(self.lwc * dz, axis=1)
        return lwp

    @cached_property
    def iwc(self):
        """
        Ice water content in g/m3
        """
        iwc = (
            np.squeeze(
                self._ncfile.variables["QICE"][self._mask, :]
                + self._ncfile.variables["QSNOW"][self._mask, :]
                + self._ncfile.variables["QGRAUP"][self._mask, :]
            )
            * self.rho
            * 10**3
        )  # gm-3
        iwc[iwc <= 10 ** (-5)] = np.nan
        return iwc

    @property
    def icnc(self) -> np.ndarray:
        """
        Ice crystal number concentration [L-1]
        """
        icnc = (
            np.squeeze(
                (
                    self._ncfile.variables["QNICE"][self._mask, :]
                    + self._ncfile.variables["QNSNOW"][self._mask, :]
                    + self._ncfile.variables["QNGRAUPEL"][self._mask, :]
                )
            )
            * self.rho
            * 10**-3
        )  # L-1

        return icnc

    @property
    def breakup(self) -> np.ndarray:
        """
        Ice break up tendency [L-1 s-1]
        """
        return np.squeeze(self._ncfile.variables["DNI_BR"][self._mask, :]) * self.rho * 10**-3  # L-1s-1

    @property
    def hallett_mossop(self) -> np.ndarray:
        """
        Rime splintering process tendency [L-1 s-1]
        """
        return np.squeeze(self._ncfile.variables["DNI_HM"][self._mask, :]) * self.rho * 10**-3  # L-1s-1

    @property
    def droplet_shatter(self) -> np.ndarray:
        """
        Droplet shattering tendency [L-1 s-1]
        """
        ds = (
            np.squeeze(
                self._ncfile.variables["DNI_DS1"][self._mask, :]
                + self._ncfile.variables["DNI_DS2"][self._mask, :]
                + self._ncfile.variables["DNS_BF1"][self._mask, :]
                + self._ncfile.variables["DNG_BF1"][self._mask, :]
            )
            * self.rho
            * 10**-3
        )  # L-1s-1

        return ds

    @property
    def sublimation_breakup(self) -> np.ndarray:
        """
        Sublimation break up tendency [L-1 s-1]
        """
        sb = (
            np.squeeze(
                self._ncfile.variables["DNI_SBS"][self._mask, :]
                + self._ncfile.variables["DNI_SBG"][self._mask, :]
            )
            * self.rho
            * 10**-3
        )  # L-1s-1
        return sb

    @property
    def primary_ice(self) -> np.ndarray:
        """
        Primary ice process tendency [L-1 s-1]
        """
        pip = (
            np.squeeze(
                self._ncfile.variables["DNI_CON"][self._mask, :]
                + self._ncfile.variables["DNI_IMM"][self._mask, :]
                + self._ncfile.variables["DNI_NUC"][self._mask, :]
                + self._ncfile.variables["DNS_CCR"][self._mask, :]
            )
            * self.rho
            * 10**-3
        )  # L-1s-1

        return pip

    @property
    def aggregation(self) -> np.ndarray:
        """
        Aggregation tendency [L-1 s-1]
        """
        return (
            np.abs(np.squeeze(self._ncfile.variables["DNS_AGG"][self._mask, :])) * self.rho * 10**-3
        )  # L-1s-1

    @property
    def rimming(self) -> np.ndarray:
        """
        Rimming tendency [g m-3 s-1]
        """
        return np.squeeze(self._ncfile.variables["DQC_RIM"][self._mask, :]) * self.rho * 10**3  # gm-3s-1

    @property
    def deposition(self) -> np.ndarray:
        """
        Deposition tendency [g m-3 s-1]
        """
        dep = (
            np.squeeze(
                self._ncfile.variables["DQI_DEP"][self._mask, :]
                + self._ncfile.variables["DQS_DEP"][self._mask, :]
                + self._ncfile.variables["DQG_DEP"][self._mask, :]
            )
            * self.rho
            * 10**3
        )  # gm-3s-1
        return dep

    @property
    def RH_I(self) -> np.ndarray:
        """
        Ice saturation ratio (relative humidity over ice)
        """
        return np.squeeze(self._ncfile.variables["DSI_TEN"][self._mask, :]) * 100

    def variables(self, var_name: str) -> np.ndarray:
        """
        Uses the `Dataset.variable` method on the ncfile.
        """
        return np.squeeze(self.ncfile.variables[var_name][self._mask, :])

    def getvar(self, var_name: str, timeidx: int | None = None, meta: bool = False) -> np.ndarray:
        """
        Uses the wrf-python getvar method on the ncfile.
        """
        data = getvar(self.ncfile, var_name, timeidx=timeidx, meta=meta)  # type: ignore
        if len(data.shape) > 2:
            return data[self._mask, :]  # type: ignore
        if len(data.shape) == 1:
            return data[self._mask]  # type: ignore

        raise NotImplementedError


class MaskedDataset:
    def __init__(self, ncfile: Dataset, mask: np.ndarray):
        self._ncfile = ncfile
        self._mask = mask

    @property
    def ncfile(self) -> Dataset:
        return self._ncfile

    @property
    def mask(self) -> np.ndarray:
        return self._mask


class WRFMultiDataset:
    def __init__(
        self,
        metadata_path: Path,
        start_time: pd.Timestamp | None,
        end_time: pd.Timestamp | None,
        station: str,
        mp_phys: int,
        bl_phys: int,
        sip: bool,
    ) -> None:
        dtypes = {
            "file_path": "str",
            "spinup": "int",
            "mp": "int",
            "bl": "int",
            "station": "str",
            "sip": "bool",
        }
        metadata = pd.read_csv(
            metadata_path, parse_dates=["start_time", "end_time", "true_start_time"], dtype=dtypes
        ).drop_duplicates(subset=["start_time", "end_time", "spinup", "mp", "bl", "sip", "station"])

        self._start_time = start_time
        self._end_time = end_time
        self._station = station
        self._mp_phys = mp_phys
        self._bl_phys = bl_phys
        self._sip = sip

        mask_start_time = np.ones(len(metadata), dtype="bool")
        mask_end_time = np.ones(len(metadata), dtype="bool")

        if start_time is not None:
            mask_start_time = metadata.end_time > start_time
        if end_time is not None:
            mask_end_time = metadata.true_start_time < end_time

        mask_mp_phys = metadata.mp == mp_phys
        mask_bl_phys = metadata.bl == bl_phys
        mask_station = metadata.station == station
        mask_sip = metadata.sip == sip

        total_filter = mask_start_time & mask_end_time & mask_mp_phys & mask_bl_phys & mask_station & mask_sip

        metadata_filtered = metadata[total_filter].sort_values("start_time").reset_index(drop=True)

        self._metadata = metadata_filtered.copy()

        if start_time is not None:
            metadata_filtered["functional_end_time"] = metadata_filtered["end_time"]
            for i in range(len(metadata_filtered)):
                if i == (len(metadata_filtered) - 1):
                    continue

                next_true_start_time = metadata_filtered.loc[i + 1, "start_time"] + pd.Timedelta(metadata_filtered.loc[i + 1, "spinup"], "h")  # type: ignore
                if metadata_filtered.loc[i, "end_time"] > next_true_start_time:  # type: ignore
                    metadata_filtered.loc[i, "functional_end_time"] = next_true_start_time - pd.Timedelta(
                        5, "minutes"
                    )

            mask_overlap = metadata_filtered.functional_end_time > start_time
            self._metadata = (
                metadata_filtered[mask_overlap].sort_values("start_time").reset_index(drop=True).copy()
            )

        # dict to store the datasets
        self._ncfiles: list[MaskedDataset] = []
        nrows = len(self._metadata)

        for i, row in self._metadata.iterrows():
            ncfile = Dataset(row.file_path)
            times = pd.Series(getvar(ncfile, "Times", meta=False, timeidx=None))  # type: ignore

            wrf_mask_start_time = np.ones(len(times), dtype="bool")
            wrf_mask_end_time = np.ones(len(times), dtype="bool")
            wrf_mask_latest_file = np.ones(len(times), dtype="bool")

            wrf_mask_spinup = times >= times.iloc[0] + pd.Timedelta(row.spinup, "h")

            if self._start_time is not None:
                wrf_mask_start_time = times >= self._start_time
            if self._end_time is not None:
                wrf_mask_end_time = times <= self._end_time
            if i < (nrows - 1):  # type: ignore
                wrf_mask_latest_file = times < (self._metadata.loc[i + 1, "start_time"] + pd.Timedelta(self._metadata.loc[i + 1, "spinup"], "h"))  # type: ignore

            mask = wrf_mask_spinup & wrf_mask_start_time & wrf_mask_end_time & wrf_mask_latest_file
            self._ncfiles.append(MaskedDataset(ncfile=ncfile, mask=mask))

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def mp_phys(self):
        return self._mp_phys

    @property
    def bl_phys(self):
        return self._bl_phys

    @property
    def station(self):
        return self._station

    @property
    def times(self) -> pd.Series:
        return pd.Series(self.getvar("Times"))

    def variables(self, var_name: str) -> np.ndarray:
        """
        Uses the `Dataset.variable` method on the ncfiles.
        """
        data = []
        for masked_ncfile in self._ncfiles:
            data.append(masked_ncfile.ncfile.variables[var_name][masked_ncfile.mask].filled())

        # 0 is the time axis
        return np.squeeze(np.concatenate(data, axis=0))

    def getvar(self, var_name: str, timeidx: int | None = None, meta: bool = False) -> np.ndarray:
        """
        Uses the wrf-python getvar method on the ncfiles.
        """
        if timeidx is None:
            data = []
            for masked_ncfile in self._ncfiles:
                data.append(
                    getvar(masked_ncfile.ncfile, var_name, timeidx=None, meta=meta)[masked_ncfile.mask]  # type: ignore
                )

        if timeidx is not None:
            temp_idx = timeidx
            for masked_ncfile in self._ncfiles:
                length = np.sum(masked_ncfile.mask)
                if length > temp_idx:  # contains the data we need
                    truth_locations = np.argwhere(masked_ncfile.mask)
                    return getvar(
                        masked_ncfile.ncfile, var_name, timeidx=truth_locations[temp_idx][0], meta=meta  # type: ignore
                    )
                temp_idx -= length

            raise RuntimeError(f"timeidx out of bounds for object with {timeidx-temp_idx} times.")

        return np.concatenate(data, axis=0)

    @cached_property
    def ZZ(self) -> np.ndarray:
        PHB = self.variables("PHB")[0, :]
        PH = self.variables("PH")[0, :]
        HGT = self.variables("HGT")[0]
        ZZ = (PH + PHB) / 9.81 - HGT

        # Clean up
        del PHB
        del PH
        del HGT

        return np.diff(ZZ) / 2 + ZZ[:-1]

    @cached_property
    def kinetic_temp(self) -> np.ndarray:
        """
        The kinetic temperature in Kelvin
        """
        RD = 287.0
        CP = 1004.5
        P1000MB = 100000.0
        total_pressure = np.squeeze(self.variables("P") + self.variables("PB"))  # in mb
        potential_temp = np.squeeze(self.variables("T") + 300.0)
        kinetic_temp = (total_pressure / P1000MB) ** (RD / CP) * potential_temp
        del total_pressure
        del potential_temp
        return kinetic_temp

    @cached_property
    def virtual_temp(self) -> np.ndarray:
        """
        The virtual temperature in Kelvin
        """
        EPS = 0.622
        qvapor = np.squeeze(self.variables("QVAPOR"))
        virtual_temp = self.kinetic_temp * (EPS + qvapor) / (EPS * (1.0 + qvapor))
        del qvapor
        return virtual_temp

    @cached_property
    def rho(self) -> np.ndarray:
        RA = 287.15
        total_pressure = np.squeeze(self.variables("P") + self.variables("PB"))  # in mb
        return total_pressure / RA / self.virtual_temp

    @cached_property
    def lwc(self) -> np.ndarray:
        """
        liquid water content in g/m3
        """
        lwc = np.squeeze(self.variables("QCLOUD") + self.variables("QRAIN")) * self.rho * 10**3  # gm-3
        lwc[lwc <= 10 ** (-5)] = np.nan
        return lwc

    @cached_property
    def lwp(self) -> np.ndarray:
        """
        liquid water path
        """
        # RA=287.15
        # total_pressure = np.squeeze(self.variables('P') + self.variables('PB')) #in mb
        # rho = total_pressure/RA/self.virtual_temp
        # lwc = np.squeeze(self.variables('QCLOUD') + self.variables('QRAIN'))*rho*10**3 #gm-3
        # lwc[lwc <= 10**(-5)] = np.nan
        zstag = np.squeeze(self.getvar("zstag"))
        dz = np.diff(zstag, axis=1)
        lwp = np.nansum(self.lwc * dz, axis=1)
        return lwp

    @cached_property
    def iwc(self) -> np.ndarray:
        """
        Ice water content in g/m3
        """
        iwc = (
            np.squeeze(self.variables("QICE") + self.variables("QSNOW") + self.variables("QGRAUP"))
            * self.rho
            * 10**3
        )  # gm-3
        iwc[iwc <= 10 ** (-5)] = np.nan
        return iwc

    @property
    def icnc(self) -> np.ndarray:
        """
        Ice crystal number concentration [L-1]
        """
        icnc = (
            np.squeeze((self.variables("QNICE") + self.variables("QNSNOW") + self.variables("QNGRAUPEL")))
            * self.rho
            * 10**-3
        )  # L-1

        return icnc

    @property
    def breakup(self) -> np.ndarray:
        """
        Ice break up tendency [L-1 s-1]
        """
        return np.squeeze(self.variables("DNI_BR")) * self.rho * 10**-3  # L-1s-1

    @property
    def hallett_mossop(self) -> np.ndarray:
        """
        Rime splintering process tendency [L-1 s-1]
        """
        return np.squeeze(self.variables("DNI_HM")) * self.rho * 10**-3  # L-1s-1

    @property
    def droplet_shatter(self) -> np.ndarray:
        """
        Droplet shattering tendency [L-1 s-1]
        """
        ds = (
            np.squeeze(
                self.variables("DNI_DS1")
                + self.variables("DNI_DS2")
                + self.variables("DNS_BF1")
                + self.variables("DNG_BF1")
            )
            * self.rho
            * 10**-3
        )  # L-1s-1

        return ds

    @property
    def sublimation_breakup(self) -> np.ndarray:
        """
        Sublimation break up tendency [L-1 s-1]
        """
        sb = np.squeeze(self.variables("DNI_SBS") + self.variables("DNI_SBG")) * self.rho * 10**-3  # L-1s-1
        return sb

    @property
    def primary_ice(self) -> np.ndarray:
        """
        Primary ice process tendency [L-1 s-1]
        """
        pip = (
            np.squeeze(
                self.variables("DNI_CON")
                + self.variables("DNI_IMM")
                + self.variables("DNI_NUC")
                + self.variables("DNS_CCR")
            )
            * self.rho
            * 10**-3
        )  # L-1s-1

        return pip

    @property
    def aggregation(self) -> np.ndarray:
        """
        Aggregation tendency [L-1 s-1]
        """
        return np.abs(np.squeeze(self.variables("DNS_AGG"))) * self.rho * 10**-3  # L-1s-1

    @property
    def rimming(self) -> np.ndarray:
        """
        Rimming tendency [g m-3 s-1]
        """
        return np.squeeze(self.variables("DQC_RIM")) * self.rho * 10**3  # gm-3s-1

    @property
    def deposition(self) -> np.ndarray:
        """
        Deposition tendency [g m-3 s-1]
        """
        dep = (
            np.squeeze(self.variables("DQI_DEP") + self.variables("DQS_DEP") + self.variables("DQG_DEP"))
            * self.rho
            * 10**3
        )  # gm-3s-1
        return dep

    @property
    def RH_I(self) -> np.ndarray:
        """
        Ice saturation ratio (relative humidity over ice)
        """
        return np.squeeze(self.variables("DSI_TEN")) * 100


class WRFMultiDatasetFactory:
    def __init__(self, metadata_path: Path) -> None:
        self._metadata_path = metadata_path

    def get_dataset(
        self,
        start_time: pd.Timestamp | None,
        end_time: pd.Timestamp | None,
        station: str,
        mp_phys: int,
        bl_phys: int,
        sip: bool,
    ) -> WRFMultiDataset:
        return WRFMultiDataset(
            metadata_path=self._metadata_path,
            start_time=start_time,
            end_time=end_time,
            mp_phys=mp_phys,
            bl_phys=bl_phys,
            station=station,
            sip=sip,
        )

    def describe_data(self) -> None:
        dtypes = {
            "file_path": "str",
            "spinup": "int",
            "mp": "int",
            "bl": "int",
            "station": "str",
            "sip": "bool",
        }
        metadata = pd.read_csv(
            self._metadata_path, parse_dates=["start_time", "end_time", "true_start_time"], dtype=dtypes
        )

        data = {}
        for (mp, bl, sip), group_unsorted in metadata.groupby(["mp", "bl", "sip"]):
            starts = []
            ends = []
            last_end = pd.Timestamp(0)
            group = (
                group_unsorted.loc[group_unsorted.station == "NPRK"]
                .drop_duplicates(subset=["start_time", "end_time", "spinup", "mp", "bl", "sip", "station"])
                .sort_values("start_time")
                .reset_index(drop=True)
            )

            for i in range(len(group)):
                row_start_time = group.loc[i, "true_start_time"]
                row_end_time = group.loc[i, "end_time"]

                if row_start_time > last_end + pd.Timedelta(5, "minutes"):  # type: ignore
                    starts.append(row_start_time)
                    if last_end != pd.Timestamp(0):
                        ends.append(last_end)
                    last_end = row_end_time

                else:
                    if row_end_time > last_end:  # type: ignore
                        last_end = row_end_time

            ends.append(last_end)
            data[f"mp:{mp}, bl:{bl}, SIP:{sip}"] = {"starts": starts, "ends": ends}

        for key, val in data.items():
            print(key)
            info_string = "Avaliable Time Periods:\n"
            for st, en in zip(val["starts"], val["ends"]):
                info_string += f"\n{st} to {en}"
            info_string += "\n"
            print(info_string)

    def get_setups(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[tuple]:
        dtypes = {
            "file_path": "str",
            "spinup": "int",
            "mp": "int",
            "bl": "int",
            "station": "str",
            "sip": "bool",
        }
        metadata = pd.read_csv(
            self._metadata_path, parse_dates=["start_time", "end_time", "true_start_time"], dtype=dtypes
        ).drop_duplicates(subset=["start_time", "end_time", "spinup", "mp", "bl", "sip", "station"])

        mask_start_time = metadata.end_time > start_time
        mask_end_time = metadata.true_start_time < end_time
        mask_station = metadata.station == "NPRK"

        total_filter = mask_start_time & mask_end_time & mask_station
        metadata_filtered = metadata[total_filter].sort_values("start_time").reset_index(drop=True)

        groups = metadata_filtered.groupby(["mp", "bl", "sip"])

        return [x for x in groups.groups]

    def get_periods(self, station: str, mp_phys: int, bl_phys: int, sip: bool) -> list[tuple]:
        dtypes = {
            "file_path": "str",
            "spinup": "int",
            "mp": "int",
            "bl": "int",
            "station": "str",
            "sip": "bool",
        }
        metadata = pd.read_csv(
            self._metadata_path, parse_dates=["start_time", "end_time", "true_start_time"], dtype=dtypes
        ).drop_duplicates(subset=["start_time", "end_time", "spinup", "mp", "bl", "sip", "station"])

        mask_mp_phys = metadata.mp == mp_phys
        mask_bl_phys = metadata.bl == bl_phys
        mask_station = metadata.station == station
        mask_sip = metadata.sip == sip

        total_filter = mask_mp_phys & mask_bl_phys & mask_station & mask_sip

        metadata_filtered = metadata[total_filter].sort_values("start_time").reset_index(drop=True)

        metadata = metadata_filtered.copy()

        starts = []
        ends = []
        last_end = pd.Timestamp(0)
        group = metadata.sort_values("start_time").reset_index(drop=True)

        for i in range(len(group)):
            row_start_time = group.loc[i, "true_start_time"]
            row_end_time = group.loc[i, "end_time"]

            if row_start_time > last_end + pd.Timedelta(5, "minutes"):  # type: ignore
                starts.append(row_start_time)
                if last_end != pd.Timestamp(0):
                    ends.append(last_end)
                last_end = row_end_time

            else:
                if row_end_time > last_end:  # type: ignore
                    last_end = row_end_time

        ends.append(last_end)

        return [(start, end) for start, end in zip(starts, ends)]


class MIRAMultiDataset:
    def __init__(self, metadata_path: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
        dtypes = {"file_path": "str"}
        metadata = pd.read_csv(metadata_path, parse_dates=["start_time", "end_time"], dtype=dtypes)
        self._start_time = start_time
        self._end_time = end_time

        mask_start_time = metadata.end_time > start_time
        mask_end_time = metadata.start_time < end_time

        total_filter = mask_start_time & mask_end_time

        metadata_filtered = metadata[total_filter].sort_values("start_time").reset_index(drop=True)

        self._metadata = metadata_filtered.copy()

        # list to store the datasets
        self._ncfiles: list[MaskedDataset] = []

        time_data = []
        for _, row in self._metadata.iterrows():
            ncfile = Dataset(row.file_path, mode="r")
            times = pd.Series(
                pd.to_datetime(
                    ncfile.variables["time"][:], origin=pd.Timestamp(year=1970, month=1, day=1), unit="s"
                )
            )

            mask_start_time = np.ones(len(times), dtype="bool")
            mask_end_time = np.ones(len(times), dtype="bool")

            mask_start_time = times >= self._start_time
            mask_end_time = times <= self._end_time

            mask = mask_end_time & mask_start_time
            self._ncfiles.append(MaskedDataset(ncfile=ncfile, mask=mask))  # type: ignore

            time_data.append(times[mask])

        self._times = pd.concat(time_data)  # type: ignore

    def variables(self, var_name: str) -> np.ndarray:
        """
        Uses the `Dataset.variable` method on the ncfiles.
        """
        data = []
        for masked_ncfile in self._ncfiles:
            data.append(masked_ncfile.ncfile.variables[var_name][masked_ncfile.mask].filled())

        # 0 is the time axis
        return np.squeeze(np.concatenate(data, axis=0))

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def times(self) -> pd.Series:
        return self._times

    @cached_property
    def range(self) -> np.ndarray:
        first_ncfile = self._ncfiles[0].ncfile.variables["range"]
        return np.squeeze(first_ncfile)

    @cached_property
    def refl(self) -> np.ndarray:
        elv = self.variables("elv")
        mask_elv = elv < 85
        refl = self.variables("Z")
        refl[mask_elv] = np.nan
        return 10 * np.log10(refl)

    @cached_property
    def skewness(self) -> np.ndarray:
        elv = self.variables("elv")
        mask_elv = elv < 85
        skewness = self.variables("SKWg")
        skewness[mask_elv] = np.nan
        return skewness

    @cached_property
    def lwc(self) -> np.ndarray:
        elv = self.variables("elv")
        mask_elv = elv < 85
        lwc = self.variables("LWC")
        lwc[mask_elv] = np.nan
        return lwc

    @cached_property
    def lwp(self) -> np.ndarray:
        return np.nansum(self.lwc, axis=1)

    @cached_property
    def clean_mask(self) -> np.ndarray | None:
        if "clean_mask" in self._ncfiles[0].ncfile.variables.keys():
            return self.variables("clean_mask") == 1
        else:
            return None

    # @staticmethod
    # def resample_data(data:np.ndarray, data_times:pd.Series, times:pd.Series, resample_function: Callable = np.nansum) -> np.ndarray:
    #     """
    #     `data` is an array with time being the first dimension. `data_times` contains
    #     the times at which the `data` was taken `times` is a pd.series that has
    #     Timestamp values that are at a lower resolution than the Times of the Mira
    #     radar. The resample_function is the function that determines how the times
    #     will be combined.
    #     """


class MIRAMultiDatasetFactory:
    def __init__(self, metadata_path: Path) -> None:
        self._metadata_path = metadata_path

    def get_dataset(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> MIRAMultiDataset:
        return MIRAMultiDataset(metadata_path=self._metadata_path, start_time=start_time, end_time=end_time)


class BASTAMultiDataset:
    def __init__(self, metadata_path: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
        dtypes = {"file_path": "str"}
        metadata = pd.read_csv(metadata_path, parse_dates=["start_time", "end_time"], dtype=dtypes)
        self._start_time = start_time
        self._end_time = end_time

        mask_start_time = metadata.end_time > start_time
        mask_end_time = metadata.start_time < end_time

        total_filter = mask_start_time & mask_end_time

        metadata_filtered = metadata[total_filter].sort_values("start_time").reset_index(drop=True)

        self._metadata = metadata_filtered.copy()

        # list to store the datasets
        self._ncfiles: list[MaskedDataset] = []

        time_data = []
        for _, row in self._metadata.iterrows():
            ncfile = Dataset(row.file_path)
            file_start_time = pd.Timestamp(
                year=int(ncfile.year), month=int(ncfile.month), day=int(ncfile.day)
            )
            times = pd.Series(pd.to_datetime(ncfile.variables["time"][:], unit="s", origin=file_start_time))

            mask_start_time = np.ones(len(times), dtype="bool")
            mask_end_time = np.ones(len(times), dtype="bool")

            mask_start_time = times >= self._start_time
            mask_end_time = times <= self._end_time

            mask = mask_end_time & mask_start_time
            self._ncfiles.append(MaskedDataset(ncfile=ncfile, mask=mask))  # type: ignore

            time_data.append(times[mask])

        self._times = pd.concat(time_data)  # type: ignore

    def variables(self, var_name: str) -> np.ndarray:
        """
        Uses the `Dataset.variable` method on the ncfiles.
        """
        data = []
        for masked_ncfile in self._ncfiles:
            data.append(masked_ncfile.ncfile.variables[var_name][masked_ncfile.mask].filled())

        # 0 is the time axis
        return np.squeeze(np.concatenate(data, axis=0))

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def times(self):
        return self._times

    @cached_property
    def range(self):
        first_ncfile = self._ncfiles[0].ncfile.variables["range"]
        return np.squeeze(first_ncfile)

    @staticmethod
    def get_calib_refl(radar_refl: np.ndarray, radar_range: np.ndarray) -> np.ndarray:
        calib_db = -186

        calib_refl = (
            radar_refl + 20.0 * np.log10(np.tile(radar_range, (radar_refl.shape[0], 1)), dtype="f") + calib_db
        )

        return calib_refl


class BASTAMultiDatasetFactory:
    def __init__(self, metadata_path: Path) -> None:
        self._metadata_path = metadata_path

    def get_dataset(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> BASTAMultiDataset:
        return BASTAMultiDataset(metadata_path=self._metadata_path, start_time=start_time, end_time=end_time)


# %%
