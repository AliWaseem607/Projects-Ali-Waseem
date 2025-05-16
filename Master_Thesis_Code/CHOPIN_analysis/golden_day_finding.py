# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from WRFMultiDataset import WRFMultiDatasetFactory

# %%
mapping_9 = {3: 0, 2: 4, 1: 2, 0: 1}
mapping_10 = {0: 2, 1: 1, 2: 4, 3: 0, 4: 3}
mapping_18 = {3: 0, 2: 4, 1: 1, 0: 2}
mapping_20 = {4: 0, 1: 1, 2: 2, 5: 3, 0: 4, 3: 5}
mapping_21 = {5: 0, 2: 1, 4: 2, 0: 3, 1: 4, 3: 5}

clusters = pd.read_csv("./figures/cluster_analysis_20/data.csv", parse_dates=["start_time", "end_time"])
label_mapping = mapping_20
clusters["label_standardized"] = -1
for key, val in label_mapping.items():
    clusters.loc[clusters.label == key, "label_standardized"] = val

clusters.sort_values("start_time", inplace=True)
clusters.reset_index(inplace=True, drop=True)
clusters.drop("Unnamed: 0", axis=1, inplace=True)

clusters_2 = pd.read_csv("./figures/cluster_analysis_21/data.csv", parse_dates=["start_time", "end_time"])
label_mapping_2 = mapping_21
clusters_2["label_standardized"] = -1
for key, val in label_mapping_2.items():
    clusters_2.loc[clusters_2.label == key, "label_standardized"] = val

clusters_2.sort_values("start_time", inplace=True)
clusters_2.reset_index(inplace=True, drop=True)
clusters_2.drop("Unnamed: 0", axis=1, inplace=True)

# Clusters:
# 0 -> Thick periods
# 1 -> less thick "spikey periods"
# 2 -> low cloud amount/shallow cloud periods
# 3 -> periods with clouds with low reflectivities
# 4 -> periods with thinner high clouds

# Clusters for 20
# 0 -> null
# 1 -> low clouds
# 2 -> sparse high clouds
# 3 -> predominately higher clouds/periods with higher base height
# 4 -> two layer/medium thickness, with high refl low mean base height
# 5 -> thick periods

# %%
plt.plot(clusters.start_time, clusters.label_standardized)
# plt.plot(clusters_2.start_time, clusters_2.label_standardized, color="orange")
# %%
plt.figure(figsize=(10, 4))
plt.scatter(clusters.start_time, clusters.label_standardized, s=0.8)
plt.scatter(clusters_2.start_time, clusters_2.label_standardized + 0.25, s=1)
new_dates = set(clusters_2.start_time)
date_mask = [x in new_dates for x in clusters.start_time]
mask = clusters_2.label_standardized.to_numpy() != clusters.loc[date_mask,"label_standardized"].to_numpy()
clusters_slice = clusters[date_mask].reset_index()
for i in range(len(mask)):
    if mask[i]:
        plt.plot(
            [clusters_slice.loc[i, "start_time"], clusters_2.loc[i, "start_time"]],
            [clusters_slice.loc[i, "label_standardized"], clusters_2.loc[i, "label_standardized"] + 0.25],
            linewidth=0.4,
        )
# %%

clusters_2["label_12h"] = -1
cluster_starts = set(clusters.start_time)
cluster_ends = set(clusters.end_time)
counter = 0
for i, row in clusters_2.iterrows():
    if i != counter:
        print("broke")
        break
    if row.start_time in cluster_starts:
        clusters_2.loc[counter, ["label_12h"]] = clusters.loc[
            clusters.start_time == row.start_time, ["label_standardized"]
        ].values[0][0]
        counter += 1
        continue
    if row.end_time in cluster_ends:
        clusters_2.loc[counter, ["label_12h"]] = clusters.loc[
            clusters.end_time == row.end_time, ["label_standardized"]
        ].values[0][0]
        counter += 1
        continue
    print("missing")

plt.scatter(clusters_2.start_time, clusters_2.label_12h, s=1)
plt.scatter(clusters_2.start_time, clusters_2.label_standardized + 0.25, s=1)


mask = clusters_2.label_standardized != clusters_2.label_12h
for i in range(len(mask)):
    if mask[i]:
        plt.plot(
            [clusters_2.loc[i, "start_time"], clusters_2.loc[i, "start_time"]],
            [clusters_2.loc[i, "label_12h"], clusters_2.loc[i, "label_standardized"] + 0.25],
            linewidth=0.4,
        )

# %%
wrf_dataset_factory = WRFMultiDatasetFactory(Path("./data/metadata.csv"))

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
