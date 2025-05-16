import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("./")
from period_analysis import SIPPeriod
from WRFMultiDataset import MIRAMultiDataset, MIRAMultiDatasetFactory, WRFDataset

if __name__ == "__main__":
    # Golden day Thick Cluster
    thick_cluster = (
        pd.Timestamp(year=2024, month=11, day=11, hour=12),
        pd.Timestamp(year=2024, month=11, day=13, hour=12),
    )
    null_cluster = (
        pd.Timestamp(year=2024, month=11, day=24, hour=0),
        pd.Timestamp(year=2024, month=11, day=25, hour=0),
    )
    low_cluster = (
        pd.Timestamp(year=2024, month=10, day=22, hour=0),
        pd.Timestamp(year=2024, month=10, day=24, hour=0),
    )
    high_sparse_cluster_2 = (
        pd.Timestamp(year=2024, month=12, day=19, hour=0),
        pd.Timestamp(year=2024, month=12, day=20, hour=0),
    )
    high_cluster = (
        pd.Timestamp(year=2024, month=10, day=18, hour=18),
        pd.Timestamp(year=2024, month=10, day=20, hour=18),
    )
    # high_cluster_2 = (
    #     pd.Timestamp(year=2024, month=12, day=24, hour=12),
    #     pd.Timestamp(year=2024, month=12, day=25, hour=12),
    # )

    med_high_refl_cluster = (
        pd.Timestamp(year=2024, month=12, day=8, hour=12),
        pd.Timestamp(year=2024, month=12, day=10, hour=0),
    )
    # med_high_refl_cluster_2 = (
    #     pd.Timestamp(year=2024, month=12, day=21, hour=0),
    #     pd.Timestamp(year=2024, month=12, day=23, hour=0),
    # )
    MIRA_dataset_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))

    gd0 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-11-23_gd0_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-11-23_gd0_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRA_dataset_factory,
        start_time=pd.Timestamp(year=2024, month=11, day=24, hour=0),
        end_time=pd.Timestamp(year=2024, month=11, day=25, hour=0),
    )

    gd1 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-10-27_gd1_2_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-10-27_gd1_2_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRA_dataset_factory,
        start_time=pd.Timestamp(year=2024, month=10, day=28, hour=0),
        end_time=pd.Timestamp(year=2024, month=10, day=30, hour=0),
    )

    gd2 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-12-18_gd2_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-12-18_gd2_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRA_dataset_factory,
        start_time=pd.Timestamp(year=2024, month=12, day=19, hour=0),
        end_time=pd.Timestamp(year=2024, month=12, day=20, hour=0),
    )

    gd3 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-10-17_gd3_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK_MYNN.nc"
        ),
        MIRA_factory=MIRA_dataset_factory,
        start_time=pd.Timestamp(year=2024, month=10, day=18, hour=18),
        end_time=pd.Timestamp(year=2024, month=10, day=20, hour=18),
    )

    gd4 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-12-07_gd4_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-12-07_gd4_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRA_dataset_factory,
        start_time=pd.Timestamp(year=2024, month=12, day=8, hour=12),
        end_time=pd.Timestamp(year=2024, month=12, day=10, hour=0),
    )

    # gd5 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-11-10_gd5_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-11-10_gd5_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRA_dataset_factory,
    #     start_time=pd.Timestamp(year=2024, month=11, day=11, hour=12),
    #     end_time=pd.Timestamp(year=2024, month=11, day=13, hour=12),
    # )

    gd5 = SIPPeriod(
        control_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-11-29_gd5_2_CONTROL_NPRK.nc"
        ),
        sip_ncfile_path=Path(
            "/home/waseem/CHOPIN_analysis/data/golden_days/wrfout/wrfout_CHPN_d03_2024-11-29_gd5_2_SIP_NPRK.nc"
        ),
        MIRA_factory=MIRA_dataset_factory,
        start_time=pd.Timestamp(year=2024, month=11, day=30, hour=0),
        end_time=pd.Timestamp(year=2024, month=12, day=1, hour=0),
    )

    # gd5_3 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/pois/wrfout_CHPN_d03_2024-11-10_poi_1_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/pois/wrfout_CHPN_d03_2024-11-10_poi_1_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRA_dataset_factory,
    #     start_time=pd.Timestamp(year=2024, month=11, day=11, hour=12),
    #     end_time=pd.Timestamp(year=2024, month=11, day=13, hour=12),
    # )

    # gd5_4 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-11_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-11_CONTROL_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRA_dataset_factory,
    #     start_time=pd.Timestamp(year=2024, month=11, day=11, hour=12),
    #     end_time=pd.Timestamp(year=2024, month=11, day=13, hour=12),
    # )

    # test = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-13_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-12-13_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRA_dataset_factory,
    #     start_time=pd.Timestamp(year=2024, month=12, day=15, hour=0),
    #     end_time=pd.Timestamp(year=2024, month=12, day=16, hour=0),
    # )

    # test2 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-28_SIP_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRA_dataset_factory,
    #     start_time=pd.Timestamp(year=2024, month=11, day=30, hour=0),
    #     end_time=pd.Timestamp(year=2024, month=12, day=1, hour=0),
    # )

    # test3 = SIPPeriod(
    #     control_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_CONTROL_NPRK.nc"
    #     ),
    #     sip_ncfile_path=Path(
    #         "/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-30_CONTROL_NPRK.nc"
    #     ),
    #     MIRA_factory=MIRA_dataset_factory,
    #     start_time=pd.Timestamp(year=2024, month=12, day=1, hour=12),
    #     end_time=pd.Timestamp(year=2024, month=12, day=2, hour=12),
    # )

    # plot_refl(self, save_path:Path | None = None, ymax: float = 12, temp_level_increment: int = 10, title :str = ""):
    legend_size = 11
    small_size = 13
    medium_size = 14
    large_size = 14
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=legend_size)  # legend fontsize
    plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
    plt.rc("axes", titlesize=medium_size)
    save_folder = Path("/home/waseem/CHOPIN_analysis/figures/final_plots/golden_days")
    gd0.plot_refl(save_path=save_folder, title="Cluster 1 Golden Period 24h", save_id="Cluster_1", ymax=10)
    gd1.plot_refl(save_path=save_folder, title="Cluster 2 Golden Period 48h", save_id="Cluster_2", ymax=10)
    gd2.plot_refl(save_path=save_folder, title="Cluster 3 Golden Period 24h", save_id="Cluster_3", ymax=10)
    gd3.plot_refl(save_path=save_folder, title="Cluster 4 Golden Period 48h", save_id="Cluster_4", ymax=10)
    gd4.plot_refl(save_path=save_folder, title="Cluster 5 Golden Period 36h", save_id="Cluster_5", ymax=8)
    gd5.plot_refl(save_path=save_folder, title="Cluster 6 Golden Period 24h", save_id="Cluster_6", ymax=10)
    # gd5_2.plot_refl(save_path=save_folder, title="Cluster 5.2")
    # gd5_3.plot_refl(save_path=save_folder, title="poi_rerun")
    # gd5_4.plot_refl(save_path=save_folder, title="late")
    # test.plot_refl(save_folder, title="alt")
    # test2.plot_refl(save_folder, title="alt2")
    # test3.plot_refl(save_folder, title="alt3")

    # test.plot_icnc_lwc(save_folder)
    # gd0.plot_icnc_lwc(save_path=save_folder, title="Cluster 0")
    # gd1.plot_icnc_lwc(save_path=save_folder, title="Cluster 1")
    # gd2.plot_icnc_lwc(save_path=save_folder, title="Cluster 2")
    # gd3.plot_icnc_lwc(save_path=save_folder, title="Cluster 3")
    # gd4.plot_icnc_lwc(save_path=save_folder, title="Cluster 4")
    # gd5.plot_icnc_lwc(save_path=save_folder, title="Cluster 5")
    # gd5_2.plot_icnc_lwc(save_path=save_folder, title="Cluster 5.2")
    # gd5_3.plot_icnc_lwc(save_path=save_folder, title="poi_rerun")
    # gd5_4.plot_icnc_lwc(save_path=save_folder, title="late")
