import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from netCDF4 import Dataset  # type: ignore
from wrf import getvar


@dataclass
class ExtractionParameters:
    radar_results_path: Path
    """This is the path to the radar data output by crsim_runner.py or similar"""
    netCDF_path: Path
    """This is the path to the netCDF file that contains the time information"""
    save_path: Path | None = None
    """Where to save the final reflectivities"""
    identifier: str = ""
    """Identifier used for saving"""



def main(
        radar_results_path: Path, 
        netCDF_path: Path, 
        save_path:Path | None, 
        identifier:str
    ) -> None:
    """
    This function is to amalgomate the radar data into one file
    """
    netCDF = Dataset(netCDF_path)
    times_arr = getvar(netCDF, "Times", meta=False, timeidx=None) # type: ignore

    data_dict = {}
    for file in radar_results_path.iterdir():
        if re.search("Output", file.stem) is None or file.suffix != ".nc" or re.search("_", file.stem) is not None:
            continue

        nc_file = Dataset(file)
        Zhh = nc_file.variables["Zhh"][:,0,0]
        np.ma.set_fill_value(Zhh, np.nan)
        time_step = re.search(r"Output([0-9]+)", file.stem).groups()[0] # type: ignore
        data_dict[int(time_step)] = Zhh.filled()
        
    keys = list(data_dict.keys())
    min = np.min(keys)
    max = np.max(keys)

    data_arr = np.zeros((max-min+1, data_dict[min].shape[0]))
    
    idx = 0
    for time_step in range(min, max+1, 1):
        data_arr[idx,:] = data_dict[time_step]
        idx += 1

    
    if save_path is None:
        save_path = radar_results_path.parent
    
    if identifier != "":
        identifier = "_" + identifier
    
    reflectivity_path = Path(save_path, f"REFL{identifier}.npy")
    times_path = Path(save_path, f"REFL_times{identifier}.npy")

    np.save(reflectivity_path, data_arr)
    np.save(times_path, times_arr[min:max+1])

if __name__ == "__main__":
    parameters = tyro.cli(ExtractionParameters)
    main(
        parameters.radar_results_path,
        parameters.netCDF_path,
        parameters.save_path,
        parameters.identifier
    )

