# %%
import numpy as np
import pandas as pd

df = pd.read_csv("/home/waseem/CHOPIN_analysis/data/metadata_radiosondes.csv", parse_dates=["start_time", "end_time"])
# %%

for i, row in df.iterrows():
    radio = pd.read_csv(row.file_path, parse_dates=["time"])
    df.loc[i, "start_time"] = radio.time.min()
    df.loc[i, "end_time"] = radio.time.max()

# %%
