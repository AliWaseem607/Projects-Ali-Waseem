# %%
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar

sys.path.append("./")
from WRFMultiDataset import MIRAMultiDataset, MIRAMultiDatasetFactory, WRFDataset

# %%


class d03SIPPeriod:
    def __init__(
        self,
        control_ncfile_path: Path,
        sip_ncfile_path: Path,
        save_name: str,
        # MIRA_factory: MIRAMultiDatasetFactory,
        spinup_hours: float = 24,
        west_east_start: int = 51,
        west_east_end: int = 252,
        south_north_start: int = 51,
        south_north_end: int = 252,
    ):
        self.control = Dataset(str(control_ncfile_path), mode="r")
        self.control_times = pd.Series(getvar(self.control, "Times", timeidx=None, meta=False))  # type: ignore
        self.control_times.name = "control_times"
        self.sip = Dataset(str(sip_ncfile_path), mode="r")
        self.sip_times = pd.Series(getvar(self.sip, "Times", timeidx=None, meta=False))  # type: ignore
        self.sip_times.name = "sip_times"
        self.start_time = self.control_times[0] + pd.Timedelta(spinup_hours, "h")
        # self.MIRA = MIRA_factory.get_dataset(start_time=start_time, end_time=end_time)

        self.control_mask = self.control_times >= self.start_time
        self.sip_mask = self.sip_times >= self.start_time

        self.we_start = west_east_start
        self.we_end = west_east_end
        self.sn_start = south_north_start
        self.sn_end = south_north_end

        self.save_path = Path("/scratch/waseem/d03_stats")
        self.save_name = save_name

    def save_statistics(self) -> None:
        time_steps: int = 50
        total_times_control = np.sum(self.control_mask)
        steps = total_times_control / time_steps
        start_idx = np.argwhere(self.control_mask)[0][0]

        agg_mean_list_control = []
        agg_sum_list_control = []

        for i in range(int(np.ceil(steps))):
            end_idx = start_idx + time_steps
            if end_idx > len(self.control_mask):
                end_idx = len(self.control_mask)

            DNS_AGG_control = self.control.variables["DNS_AGG"][
                start_idx : end_idx + 1, :, self.sn_start : self.sn_end + 1, self.we_start : self.we_end + 1
            ].filled()
            DNS_AGG_control[DNS_AGG_control == 0] = np.nan
            agg_mean_list_control.append(np.nanmean(np.nanmean(DNS_AGG_control, axis=2), axis=2))
            agg_sum_list_control.append(np.nansum(np.nansum(DNS_AGG_control, axis=2), axis=2))

            start_idx = end_idx

        del DNS_AGG_control

        agg_mean_control = np.concatenate(agg_mean_list_control, axis=0)
        del agg_mean_list_control
        agg_sum_control = np.concatenate(agg_sum_list_control, axis=0)
        del agg_sum_list_control

        np.save(
            Path(self.save_path, f"{self.save_name}_CONTROL.npy"),
            np.stack([agg_mean_control, agg_sum_control]),
        )
        self.control_times.to_csv(Path(self.save_path, f"{self.save_name}_CONTROL_times.csv"), index=False)

        del agg_mean_control
        del agg_sum_control

        start_idx = np.argwhere(self.sip_mask)[0][0]

        agg_mean_list_sip = []
        agg_sum_list_sip = []

        for i in range(int(np.ceil(steps))):
            end_idx = start_idx + time_steps
            if end_idx > len(self.sip_mask):
                end_idx = len(self.sip_mask)

            DNS_AGG_sip = self.sip.variables["DNS_AGG"][
                start_idx : end_idx + 1, :, self.sn_start : self.sn_end + 1, self.we_start : self.we_end + 1
            ].filled()
            DNS_AGG_sip[DNS_AGG_sip == 0] = np.nan
            agg_mean_list_sip.append(np.nanmean(np.nanmean(DNS_AGG_sip, axis=2), axis=2))
            agg_sum_list_sip.append(np.nansum(np.nansum(DNS_AGG_sip, axis=2), axis=2))

            start_idx = end_idx

        del DNS_AGG_sip

        agg_mean_sip = np.concatenate(agg_mean_list_sip, axis=0)
        del agg_mean_list_sip
        agg_sum_sip = np.concatenate(agg_sum_list_sip, axis=0)
        del agg_sum_list_sip

        np.save(Path(self.save_path, f"{self.save_name}_SIP.npy"), np.stack([agg_mean_sip, agg_sum_sip]))
        self.sip_times.to_csv(Path(self.save_path, f"{self.save_name}_SIP_times.csv"), index=False)

        del agg_mean_sip
        del agg_sum_sip

    def save_tendency_statistics(self, time_steps: int = 5) -> None:
        total_times_control = np.sum(self.control_mask)
        steps = total_times_control / time_steps
        start_idx = np.argwhere(self.control_mask)[0][0]

        PIP_mean_list_control = []
        PIP_sum_list_control = []

        PIP_mean_list_sip = []
        PIP_sum_list_sip = []
        HM_mean_list_sip = []
        HM_sum_list_sip = []
        DS_mean_list_sip = []
        DS_sum_list_sip = []
        SUBBR_mean_list_sip = []
        SUBBR_sum_list_sip = []
        BR_mean_list_sip = []
        BR_sum_list_sip = []

        pip_v_sip_0_10_list = []
        pip_v_sip_10_20_list = []
        pip_v_sip_20_30_list = []
        pip_v_sip_30_38_list = []
        pip_v_sip_30_40_list = []

        RD = 287.0
        CP = 1004.5
        P1000MB = 100000.0
        EPS = 0.622
        RA = 287.15
        # # start with COntrol
        # for i in range(int(np.ceil(steps))):
        #     end_idx = start_idx + time_steps
        #     if end_idx > len(self.control_mask):
        #         end_idx = len(self.control_mask)

        #     total_pressure = np.squeeze(
        #         self.control.variables["P"][
        #             start_idx : end_idx + 1,
        #             :,
        #             self.sn_start : self.sn_end + 1,
        #             self.we_start : self.we_end + 1,
        #         ]
        #         + self.control.variables["PB"][
        #             start_idx : end_idx + 1,
        #             :,
        #             self.sn_start : self.sn_end + 1,
        #             self.we_start : self.we_end + 1,
        #         ]
        #     ).filled()

        #     potential_temp = np.squeeze(
        #         self.control.variables["T"][
        #             start_idx : end_idx + 1,
        #             :,
        #             self.sn_start : self.sn_end + 1,
        #             self.we_start : self.we_end + 1,
        #         ]
        #         + 300.0
        #     ).filled()
        #     kinetic_temp = (total_pressure / P1000MB) ** (RD / CP) * potential_temp
        #     del potential_temp

        #     qvapor = np.squeeze(
        #         self.control.variables["QVAPOR"][
        #             start_idx : end_idx + 1,
        #             :,
        #             self.sn_start : self.sn_end + 1,
        #             self.we_start : self.we_end + 1,
        #         ]
        #     ).filled()
        #     virtual_temp = kinetic_temp * (EPS + qvapor) / (EPS * (1.0 + qvapor))
        #     del qvapor
        #     del kinetic_temp

        #     rho = total_pressure / RA / virtual_temp
        #     del total_pressure
        #     del virtual_temp

        #     pip = (
        #         np.squeeze(
        #             self.control.variables["DNI_CON"][
        #                 start_idx : end_idx + 1,
        #                 :,
        #                 self.sn_start : self.sn_end + 1,
        #                 self.we_start : self.we_end + 1,
        #             ]
        #             + self.control.variables["DNI_IMM"][
        #                 start_idx : end_idx + 1,
        #                 :,
        #                 self.sn_start : self.sn_end + 1,
        #                 self.we_start : self.we_end + 1,
        #             ]
        #             + self.control.variables["DNI_NUC"][
        #                 start_idx : end_idx + 1,
        #                 :,
        #                 self.sn_start : self.sn_end + 1,
        #                 self.we_start : self.we_end + 1,
        #             ]
        #             + self.control.variables["DNS_CCR"][
        #                 start_idx : end_idx + 1,
        #                 :,
        #                 self.sn_start : self.sn_end + 1,
        #                 self.we_start : self.we_end + 1,
        #             ]
        #         ).filled()
        #         * rho
        #         * 10**-3
        #     )  # L-1s-1

        #     del rho

        #     pip[pip == 0] = np.nan
        #     PIP_mean_list_control.append(np.nanmean(np.nanmean(pip, axis=2), axis=2))
        #     PIP_sum_list_control.append(np.nansum(np.nansum(pip, axis=2), axis=2))
        #     del pip
        #     start_idx = end_idx

        # PIP_mean_control = np.concatenate(PIP_mean_list_control, axis=0)
        # del PIP_mean_list_control
        # PIP_sum_control = np.concatenate(PIP_sum_list_control, axis=0)
        # del PIP_sum_list_control

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_CONTROL_PIP.npy"),
        #     np.stack([PIP_mean_control, PIP_sum_control]),
        # )
        # del PIP_mean_control
        # del PIP_sum_control

        # Run SIP calculations
        for i in range(int(np.ceil(steps))):
            end_idx = start_idx + time_steps
            if end_idx > len(self.sip_mask):
                end_idx = len(self.sip_mask)

            total_pressure = np.squeeze(
                self.sip.variables["P"][
                    start_idx : end_idx + 1,
                    :,
                    self.sn_start : self.sn_end + 1,
                    self.we_start : self.we_end + 1,
                ]
                + self.sip.variables["PB"][
                    start_idx : end_idx + 1,
                    :,
                    self.sn_start : self.sn_end + 1,
                    self.we_start : self.we_end + 1,
                ]
            ).filled()

            potential_temp = np.squeeze(
                self.sip.variables["T"][
                    start_idx : end_idx + 1,
                    :,
                    self.sn_start : self.sn_end + 1,
                    self.we_start : self.we_end + 1,
                ]
                + 300.0
            ).filled()
            kinetic_temp = (total_pressure / P1000MB) ** (RD / CP) * potential_temp
            del potential_temp

            qvapor = np.squeeze(
                self.sip.variables["QVAPOR"][
                    start_idx : end_idx + 1,
                    :,
                    self.sn_start : self.sn_end + 1,
                    self.we_start : self.we_end + 1,
                ]
            ).filled()
            virtual_temp = kinetic_temp * (EPS + qvapor) / (EPS * (1.0 + qvapor))
            del qvapor

            rho = total_pressure / RA / virtual_temp
            del total_pressure
            del virtual_temp

            pip = (
                np.squeeze(
                    self.sip.variables["DNI_CON"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNI_IMM"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNI_NUC"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNS_CCR"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                ).filled()
                * rho
                * 10**-3
            )  # L-1s-1

            pip[pip == 0] = np.nan
            PIP_mean_list_sip.append(np.nanmean(np.nanmean(pip, axis=2), axis=2))
            PIP_sum_list_sip.append(np.nansum(np.nansum(pip, axis=2), axis=2))

            sip_total = np.zeros(pip.shape)

            hm = (
                np.squeeze(
                    self.sip.variables["DNI_HM"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                ).filled()
                * rho
                * 10**-3
            )  # L-1s-1

            sip_total = sip_total + hm
            hm[hm == 0] = np.nan
            HM_mean_list_sip.append(np.nanmean(np.nanmean(hm, axis=2), axis=2))
            HM_sum_list_sip.append(np.nansum(np.nansum(hm, axis=2), axis=2))

            del hm

            ds = (
                np.squeeze(
                    self.sip.variables["DNI_DS1"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNI_DS2"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNS_BF1"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNG_BF1"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                ).filled()
                * rho
                * 10**-3
            )  # L-1s-1

            sip_total = sip_total + ds
            ds[ds == 0] = np.nan
            DS_mean_list_sip.append(np.nanmean(np.nanmean(ds, axis=2), axis=2))
            DS_sum_list_sip.append(np.nansum(np.nansum(ds, axis=2), axis=2))

            del ds

            sb = (
                np.squeeze(
                    self.sip.variables["DNI_SBS"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                    + self.sip.variables["DNI_SBG"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                ).filled()
                * rho
                * 10**-3
            )  # L-1s-1

            sip_total = sip_total + sb
            sb[sb == 0] = np.nan
            SUBBR_mean_list_sip.append(np.nanmean(np.nanmean(sb, axis=2), axis=2))
            SUBBR_sum_list_sip.append(np.nansum(np.nansum(sb, axis=2), axis=2))

            del sb

            br = (
                np.squeeze(
                    self.sip.variables["DNI_BR"][
                        start_idx : end_idx + 1,
                        :,
                        self.sn_start : self.sn_end + 1,
                        self.we_start : self.we_end + 1,
                    ]
                ).filled()
                * rho
                * 10**-3
            )  # L-1s-1

            sip_total = sip_total + br
            br[br == 0] = np.nan
            BR_mean_list_sip.append(np.nanmean(np.nanmean(br, axis=2), axis=2))
            BR_sum_list_sip.append(np.nansum(np.nansum(br, axis=2), axis=2))

            del br

            sip_total[sip_total == 0] = np.nan
            mask = ~np.isnan(pip) & ~np.isnan(sip_total)
            if np.all(~mask):
                # pip_v_sip_0_10_list.append(np.array([[0, 0]]))
                # pip_v_sip_10_20_list.append(np.array([[0, 0]]))
                # pip_v_sip_20_30_list.append(np.array([[0, 0]]))
                # pip_v_sip_30_38_list.append(np.array([[0, 0]]))
                pip_v_sip_30_40_list.append(np.array([[0, 0]]))
                del kinetic_temp
                del pip
                del sip_total
                del mask
                start_idx = end_idx
                continue

            # points = np.where((kinetic_temp < 0) & (kinetic_temp >= -10) & mask)
            # points_pip_gt_sip = np.sum(pip[points] > sip_total[points])
            # pip_v_sip_0_10_list.append(np.array([[points_pip_gt_sip, points[0].shape[0]]]))

            # points = np.where((kinetic_temp < -10) & (kinetic_temp >= -20) & mask)
            # points_pip_gt_sip = np.sum(pip[points] > sip_total[points])
            # pip_v_sip_10_20_list.append(np.array([[points_pip_gt_sip, points[0].shape[0]]]))

            # points = np.where((kinetic_temp < -20) & (kinetic_temp >= -30) & mask)
            # points_pip_gt_sip = np.sum(pip[points] > sip_total[points])
            # pip_v_sip_20_30_list.append(np.array([[points_pip_gt_sip, points[0].shape[0]]]))

            # points = np.where((kinetic_temp < -30) & (kinetic_temp >= -38) & mask)
            # points_pip_gt_sip = np.sum(pip[points] > sip_total[points])
            # pip_v_sip_30_38_list.append(np.array([[points_pip_gt_sip, points[0].shape[0]]]))

            points = np.where((kinetic_temp < -30) & (kinetic_temp >= -40) & mask)
            points_pip_gt_sip = np.sum(pip[points] > sip_total[points])
            pip_v_sip_30_40_list.append(np.array([[points_pip_gt_sip, points[0].shape[0]]]))

            del kinetic_temp
            del pip
            del sip_total
            del points
            del mask
            del points_pip_gt_sip
            start_idx = end_idx

        # PIP_mean_sip = np.concatenate(PIP_mean_list_sip, axis=0)
        del PIP_mean_list_sip
        # PIP_sum_sip = np.concatenate(PIP_sum_list_sip, axis=0)
        del PIP_sum_list_sip
        # HM_mean_sip = np.concatenate(HM_mean_list_sip, axis=0)
        del HM_mean_list_sip
        # HM_sum_sip = np.concatenate(HM_sum_list_sip, axis=0)
        del HM_sum_list_sip
        # DS_mean_sip = np.concatenate(DS_mean_list_sip, axis=0)
        del DS_mean_list_sip
        # DS_sum_sip = np.concatenate(DS_sum_list_sip, axis=0)
        del DS_sum_list_sip
        # SUBBR_mean_sip = np.concatenate(SUBBR_mean_list_sip, axis=0)
        del SUBBR_mean_list_sip
        # SUBBR_sum_sip = np.concatenate(SUBBR_sum_list_sip, axis=0)
        del SUBBR_sum_list_sip
        # BR_mean_sip = np.concatenate(BR_mean_list_sip, axis=0)
        del BR_mean_list_sip
        # BR_sum_sip = np.concatenate(BR_sum_list_sip, axis=0)
        del BR_sum_list_sip

        # pip_v_sip_0_10 = np.concatenate(pip_v_sip_0_10_list, axis=0)
        del pip_v_sip_0_10_list
        # pip_v_sip_10_20 = np.concatenate(pip_v_sip_10_20_list, axis=0)
        del pip_v_sip_10_20_list
        # pip_v_sip_20_30 = np.concatenate(pip_v_sip_20_30_list, axis=0)
        del pip_v_sip_20_30_list
        # pip_v_sip_30_38 = np.concatenate(pip_v_sip_30_38_list, axis=0)
        del pip_v_sip_30_38_list

        pip_v_sip_30_40 = np.concatenate(pip_v_sip_30_40_list, axis=0)
        del pip_v_sip_30_40_list

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_SIP_PIP.npy"),
        #     np.stack([PIP_mean_sip, PIP_sum_sip]),
        # )
        # del PIP_mean_sip
        # del PIP_sum_sip

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_SIP_HM.npy"),
        #     np.stack([HM_mean_sip, HM_sum_sip]),
        # )
        # del HM_mean_sip
        # del HM_sum_sip

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_SIP_DS.npy"),
        #     np.stack([DS_mean_sip, DS_sum_sip]),
        # )
        # del DS_mean_sip
        # del DS_sum_sip

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_SIP_SUBBR.npy"),
        #     np.stack([SUBBR_mean_sip, SUBBR_sum_sip]),
        # )
        # del SUBBR_mean_sip
        # del SUBBR_sum_sip

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_SIP_BR.npy"),
        #     np.stack([BR_mean_sip, BR_sum_sip]),
        # )
        # del BR_mean_sip
        # del BR_sum_sip

        # np.save(
        #     Path(self.save_path, f"{self.save_name}_PIP_gt_SIP_0_10.npy"),
        #     pip_v_sip_0_10,
        # )
        # del pip_v_sip_0_10
        # np.save(
        #     Path(self.save_path, f"{self.save_name}_PIP_gt_SIP_10_20.npy"),
        #     pip_v_sip_10_20,
        # )
        # del pip_v_sip_10_20
        # np.save(
        #     Path(self.save_path, f"{self.save_name}_PIP_gt_SIP_20_30.npy"),
        #     pip_v_sip_20_30,
        # )
        # del pip_v_sip_20_30
        # np.save(
        #     Path(self.save_path, f"{self.save_name}_PIP_gt_SIP_30_38.npy"),
        #     pip_v_sip_30_38,
        # )
        # del pip_v_sip_30_38
        np.save(
            Path(self.save_path, f"{self.save_name}_PIP_gt_SIP_30_40.npy"),
            pip_v_sip_30_40,
        )
        del pip_v_sip_30_40





print("CHOPIN_dec13-16_CONTROL")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_dec13-16_CONTROL/wrfout_CHPN_d03_2024-12-13_12:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_dec13-16_SIP/wrfout_CHPN_d03_2024-12-13_12:00:00.nc"),
    "CHOPIN_dec13-16",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_dec19-22")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_dec19-22_CONTROL/wrfout_CHPN_d03_2024-12-19_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_dec19-22_SIP/wrfout_CHPN_d03_2024-12-19_00:00:00.nc"),
    "CHOPIN_dec19-22",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_dec21-23")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_dec21-23_CONTROL/wrfout_CHPN_d03_2024-12-21_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_dec21-23_SIP/wrfout_CHPN_d03_2024-12-21_00:00:00.nc"),
    "CHOPIN_dec21-23",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_dec22-25")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_dec22-25_CONTROL/wrfout_CHPN_d03_2024-12-22_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_dec22-25_SIP/wrfout_CHPN_d03_2024-12-22_00:00:00.nc"),
    "CHOPIN_dec22-25",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_dec3-6")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_dec3-6_CONTROL/wrfout_CHPN_d03_2024-12-03_18:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_dec3-6_SIP/wrfout_CHPN_d03_2024-12-03_18:00:00.nc"),
    "CHOPIN_dec3-6",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_0")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_0_CONTROL/wrfout_CHPN_d03_2024-11-23_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_0_SIP/wrfout_CHPN_d03_2024-11-23_00:00:00.nc"),
    "CHOPIN_gd_0",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_1")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_1_CONTROL/wrfout_CHPN_d03_2024-10-21_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_1_SIP/wrfout_CHPN_d03_2024-10-21_00:00:00.nc"),
    "CHOPIN_gd_1",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_1_2")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_1_CONTROL_2/wrfout_CHPN_d03_2024-10-27_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_1_SIP_2/wrfout_CHPN_d03_2024-10-27_00:00:00.nc"),
    "CHOPIN_gd_1_2",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_2")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_2_CONTROL/wrfout_CHPN_d03_2024-12-18_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_2_SIP/wrfout_CHPN_d03_2024-12-18_00:00:00.nc"),
    "CHOPIN_gd_2",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_3")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_3_CONTROL/wrfout_CHPN_d03_2024-10-17_18:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_envelop_oct17-20_MYNN/wrfout_d03_2024-10-17_full_domain.nc"),
    "CHOPIN_gd_3",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_4")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_4_CONTROL/wrfout_CHPN_d03_2024-12-07_12:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_4_SIP/wrfout_CHPN_d03_2024-12-07_12:00:00.nc"),
    "CHOPIN_gd_4",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_5_2")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_5_CONTROL/wrfout_CHPN_d03_2024-11-10_12:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_5_SIP/wrfout_CHPN_d03_2024-11-10_12:00:00.nc"),
    "CHOPIN_gd_5",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_gd_5")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_gd_5_CONTROL_2/wrfout_CHPN_d03_2024-11-29_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_gd_5_SIP_2/wrfout_CHPN_d03_2024-11-29_00:00:00.nc"),
    "CHOPIN_gd_5_2",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_nov28-dec1")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_nov28-dec1_CONTROL/wrfout_CHPN_d03_2024-11-28_06:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_nov28-dec1_SIP/wrfout_CHPN_d03_2024-11-28_06:00:00.nc"),
    "CHOPIN_nov28-dec1",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

print("CHOPIN_nov30-dec3")
period = d03SIPPeriod(
    Path("/scratch/waseem/CHOPIN_nov30-dec3_CONTROL/wrfout_CHPN_d03_2024-11-30_00:00:00.nc"),
    Path("/scratch/waseem/CHOPIN_nov30-dec3_SIP/wrfout_CHPN_d03_2024-11-30_00:00:00.nc"),
    "CHOPIN_nov30-dec3",
)
period.save_tendency_statistics()
# period.save_statistics()
del period

# # print("CHOPIN_poi_1")
# # period = d03SIPPeriod(Path("/scratch/waseem/CHOPIN_poi_1_CONTROL/wrfout_CHPN_d03_2024-11-10_00:00:00.nc"),
# # Path("/scratch/waseem/CHOPIN_poi_1_SIP/wrfout_CHPN_d03_2024-11-10_00:00:00.nc"), )
# # period.save_tendency_statistics()
# # # period.save_statistics()
# # del period

# # period = d03SIPPeriod(Path("/scratch/waseem/CHOPIN_poi_3_CONTROL/wrfout_CHPN_d03_2024-11-14_06:00:00.nc"),
# # Path("/scratch/waseem/CHOPIN_poi_3_SIP/wrfout_CHPN_d03_2024-11-14_06:00:00.nc"), )
# # period.save_tendency_statistics()
# # # period.save_statistics()
# # del period


# %%
