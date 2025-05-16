# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%

df = pd.read_csv("./data/radiosondes/latlon_comparison/2024-11-19-09.csv")
plt.hist(df.wind_dir)
# %%

data_path = Path("./data/radiosondes/latlon_comparison_2/")
data = []
for path in data_path.iterdir():
    df = pd.read_csv(path)

    data.append(
        {
            "name": path.stem,
            "last_lat_error": df.iloc[-1]["lat_calculated"] - df.iloc[-1]["lat"],
            "last_lon_error": df.iloc[-1]["lon_calculated"] - df.iloc[-1]["lon"],
            "mean_wind_dir": np.mean(df.wind_dir),
            "mean_wind_speed": np.mean(df.wind_speed),
            "first_lat": df.iloc[0]["lat"],
            "first_lon": df.iloc[0]["lon"],
            "last_lat": df.iloc[-1]["lat"],
            "last_lon": df.iloc[-1]["lon"],
            "mean_error_lat": np.mean(df["lat_calculated"] - df["lat"]),
            "mean_error_lon": np.mean(df["lon_calculated"] - df["lon"]),
            "mean_error_still_lat": np.mean(df.iloc[0]["lat"] - df["lat"]),
            "mean_error_still_lon": np.mean(df.iloc[0]["lon"] - df["lon"]),
        }
    )

df = pd.DataFrame(data)
plt.scatter(df.mean_error_lat, df.mean_error_lon)
# plt.scatter(df.mean_error_still_lat, df.mean_error_still_lon)
plt.figure()
plt.scatter(df.last_lat_error, df.last_lon_error)
# plt.scatter(df.first_lat - df.last_lat, df.first_lon - df.last_lon)


# %%
df = pd.read_csv("./data/radiosondes/latlon_comparison_2/2024-11-19-09.csv")
plt.scatter(df.lat_calculated / df.lat, df.lon_calculated / df.lon)
