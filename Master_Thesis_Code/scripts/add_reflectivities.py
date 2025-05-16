import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from netCDF4 import Dataset  # type: ignore
from wrf import getvar


@dataclass
class ExtractionParameters:
    BASTA_results_path: Path
    """This is the path to the radar data output by crsim_runner.py or similar"""
    MIRA_results_path: Path
    """This is the path to the radar data output by crsim_runner.py or similar"""
    wrfout_path: Path
    """
    This is the path to the netCDF file the data was run on and that will
    be updated
    """

def main(
        BASTA_results_path: Path,
        MIRA_results_path:Path, 
        ncfile_path: Path, 
    ) -> None:
    """
    This function is to amalgomate the radar data into one file
    """
    ncfile = Dataset(ncfile_path, mode = "a")
    if "Zhh_BASTA" in ncfile.variables.keys():
        print(f"Zhh_BASTA is already in {ncfile_path}, skipping")
    else:
        Zhh_BASTA = ncfile.createVariable("Zhh_BASTA", "f4", ("Time", "bottom_top",), fill_value=-999.0)
        Zhh_BASTA.units = "dBZ"
        Zhh_BASTA.description = "Reflectivity from CR-SIM for BASTA radar"
        for file in BASTA_results_path.iterdir():
            if re.search("Output", file.stem) is None or file.suffix != ".nc" or re.search("_", file.stem) is not None:
                continue

            radar_ncfile = Dataset(file)
            Zhh = radar_ncfile.variables["Zhh"][:,0,0]
            np.ma.set_fill_value(Zhh, np.nan)
            time_step = re.search(r"Output([0-9]+)", file.stem).groups()[0] # type: ignore
            Zhh_BASTA[int(time_step)-1,:] = Zhh.filled()

    if "Zhh_MIRA" in ncfile.variables.keys():
        print(f"Zhh_MIRA is already in {ncfile_path}, skipping")
    else:
        Zhh_MIRA = ncfile.createVariable("Zhh_MIRA", "f4", ("Time", "bottom_top",), fill_value=-999.0)
        Zhh_MIRA.units = "dBZ"
        Zhh_MIRA.description = "Reflectivity from CR-SIM for MIRA radar"
    
        for file in MIRA_results_path.iterdir():
            if re.search("Output", file.stem) is None or file.suffix != ".nc" or re.search("_", file.stem) is not None:
                continue

            radar_ncfile = Dataset(file)
            Zhh = radar_ncfile.variables["Zhh"][:,0,0]
            np.ma.set_fill_value(Zhh, np.nan)
            time_step = re.search(r"Output([0-9]+)", file.stem).groups()[0] # type: ignore
            Zhh_MIRA[int(time_step)-1,:] = Zhh.filled()

    ncfile.close()

if __name__ == "__main__":
    parameters = tyro.cli(ExtractionParameters)
    main(
        BASTA_results_path=parameters.BASTA_results_path, 
        MIRA_results_path=parameters.MIRA_results_path,
        ncfile_path=parameters.wrfout_path,
    )

