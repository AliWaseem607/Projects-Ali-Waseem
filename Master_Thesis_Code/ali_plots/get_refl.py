#%%
from netCDF4 import Dataset
from pathlib import Path
import re
import numpy as np

#%%

times_path = Path("ncdump_AW.times")
data_path_control = Path("/work/lapi/CRSIM/crsim-3.32/share/crsim/test/out/Helmos_control_AW")
data_path_demott = Path("/work/lapi/CRSIM/crsim-3.32/share/crsim/test/out/Helmos_demott_AW")
data_path_allsip = Path("/work/lapi/CRSIM/crsim-3.32/share/crsim/test/out/Helmos_allsip_AW")

#%%
def gather_radar_data(radar_results:Path, times_path:Path) -> tuple[np.ndarray, np.ndarray]:
    with open(times_path, "r") as f:
        text = f.read()

    pattern_times = r'"([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2}:[0-9]{2}:[0-9]{2})"[,;]{1} *// *Times'

    times_found = re.findall(pattern_times, text)
    times = [date + "T" + time for date, time in times_found]
    times_arr = np.array(times)

    data_dict = {}
    for file in radar_results.iterdir():
        if re.search("_", file.stem) is not None:
            continue
        nc_file = Dataset(file)
        Zvv = nc_file.variables["Zvv"][:,0,0]
        np.ma.set_fill_value(Zvv, np.nan)
        time_step = re.search(r"Output([0-9]+)", file.stem).groups()[0]
        data_dict[int(time_step)] = Zvv.filled()
        
    keys = list(data_dict.keys())
    min = np.min(keys)
    max = np.max(keys)

    data_arr = np.zeros((max-min+1, data_dict[min].shape[0]))
    
    idx = 0
    for time_step in range(min, max+1, 1):
        data_arr[idx,:] = data_dict[time_step]
        idx += 1

    return data_arr, times_arr[min:max+1]


data_control, time_control = gather_radar_data(data_path_control, times_path)
data_demott, time_demott = gather_radar_data(data_path_demott, times_path)
data_allsip, time_allsip = gather_radar_data(data_path_allsip, times_path)


#%%
np.save("ali_radar_data/REFL_control.npy", data_control)
np.save("ali_radar_data/time_REFL_control.npy", time_control)
np.save("ali_radar_data/REFL_demott.npy", data_demott)
np.save("ali_radar_data/time_REFL_demott.npy", time_demott)
np.save("ali_radar_data/REFL_allsip.npy", data_allsip)
np.save("ali_radar_data/time_REFL_allsip.npy", time_allsip)



# %%
