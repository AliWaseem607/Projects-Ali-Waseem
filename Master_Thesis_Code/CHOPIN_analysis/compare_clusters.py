# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap

# %%

old_cluster_10 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_10.csv", parse_dates=["start_time", "end_time"]
)  # winner time percent, total percent, mean refl
old_cluster_11 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_11.csv", parse_dates=["start_time", "end_time"]
)  # 2nd place total, mean relf, std refl
old_cluster_2 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_2.csv", parse_dates=["start_time", "end_time"]
)  # stock
old_cluster_3 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_3.csv", parse_dates=["start_time", "end_time"]
)  # max cloud height instead of mean
old_cluster_4 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_4.csv", parse_dates=["start_time", "end_time"]
)  # mean refl
old_cluster_5 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_5.csv", parse_dates=["start_time", "end_time"]
)  # time percent
old_cluster_7 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_7.csv", parse_dates=["start_time", "end_time"]
)  # total percent
old_cluster_8 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_8.csv", parse_dates=["start_time", "end_time"]
)  # time percent, total percent
old_cluster_12 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_12.csv", parse_dates=["start_time", "end_time"]
)  # mean refl, std refl
old_cluster_15 = pd.read_csv(
    "./figures/old_clusters/data_cluster_analysis_15.csv", parse_dates=["start_time", "end_time"]
)  # mean refl, total percent

cluster_10 = pd.read_csv(
    "./figures/cluster_analysis_10/data.csv", parse_dates=["start_time", "end_time"]
)  # winner time percent, total percent, mean refl
cluster_11 = pd.read_csv(
    "./figures/cluster_analysis_11/data.csv", parse_dates=["start_time", "end_time"]
)  # 2nd place total, mean relf, std refl
cluster_2 = pd.read_csv(
    "./figures/cluster_analysis_2/data.csv", parse_dates=["start_time", "end_time"]
)  # stock
cluster_3 = pd.read_csv(
    "./figures/cluster_analysis_3/data.csv", parse_dates=["start_time", "end_time"]
)  # max cloud height instead of mean
cluster_4 = pd.read_csv(
    "./figures/cluster_analysis_4/data.csv", parse_dates=["start_time", "end_time"]
)  # mean refl
cluster_5 = pd.read_csv(
    "./figures/cluster_analysis_5/data.csv", parse_dates=["start_time", "end_time"]
)  # time percent
cluster_7 = pd.read_csv(
    "./figures/cluster_analysis_7/data.csv", parse_dates=["start_time", "end_time"]
)  # total percent
cluster_8 = pd.read_csv(
    "./figures/cluster_analysis_8/data.csv", parse_dates=["start_time", "end_time"]
)  # time percent, total percent
cluster_12 = pd.read_csv(
    "./figures/cluster_analysis_12/data.csv", parse_dates=["start_time", "end_time"]
)  # mean refl, std refl
cluster_15 = pd.read_csv(
    "./figures/cluster_analysis_15/data.csv", parse_dates=["start_time", "end_time"]
)  # mean refl, total percent


old_mapping_2 = {0: 3, 1: 0, 2: 2, 3: 1, 4: 4}
old_mapping_3 = {0: 1, 1: 3, 2: 2, 3: 4, 4: 0}
old_mapping_4 = {0: 3, 1: 1, 2: 2, 3: 0, 4: 4}
old_mapping_5 = {0: 2, 1: 4, 2: 0, 3: 1, 4: 3}
old_mapping_7 = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
old_mapping_8 = {0: 3, 1: 2, 2: 0, 3: 4, 4: 1}
old_mapping_10 = {0: 4, 1: 1, 2: 0, 3: 2, 4: 3}
old_mapping_11 = {0: 4, 1: 3, 2: 2, 3: 0, 4: 1}
old_mapping_12 = {0: 2, 1: 1, 2: 3, 3: 0, 4: 4}
old_mapping_15 = {0: 1, 1: 3, 2: 2, 3: 0, 4: 4}


mapping_2 = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}
mapping_3 = {0: 2, 1: 4, 2: 0, 3: 3, 4: 1}
mapping_4 = {0: 2, 1: 1, 2: 4, 3: 0, 4: 3}
mapping_5 = {0: 2, 1: 0, 2: 4, 3: 3, 4: 1}
mapping_7 = {0: 2, 1: 0, 2: 4, 3: 1, 4: 3}
mapping_8 = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}
mapping_10 = {0: 2, 1: 1, 2: 4, 3: 0, 4: 3}
mapping_11 = {0: 2, 1: 0, 2: 4, 3: 3, 4: 1}
mapping_12 = {0: 2, 1: 3, 2: 4, 3: 1, 4: 0}
mapping_15 = {0: 2, 1: 0, 2: 4, 3: 1, 4: 3}


def clean_and_add_standardized_labels(df: pd.DataFrame, mapping: dict[int, int]):
    df["label_standardized"] = -1
    for key, val in mapping.items():
        df.loc[df.label == key, "label_standardized"] = val

    df.sort_values("start_time", inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop("Unnamed: 0", axis=1, inplace=True)


clean_and_add_standardized_labels(df=old_cluster_10, mapping=old_mapping_10)
clean_and_add_standardized_labels(df=old_cluster_11, mapping=old_mapping_11)
clean_and_add_standardized_labels(df=old_cluster_2, mapping=old_mapping_2)
clean_and_add_standardized_labels(df=old_cluster_3, mapping=old_mapping_3)
clean_and_add_standardized_labels(df=old_cluster_4, mapping=old_mapping_4)
clean_and_add_standardized_labels(df=old_cluster_5, mapping=old_mapping_5)
clean_and_add_standardized_labels(df=old_cluster_7, mapping=old_mapping_7)
clean_and_add_standardized_labels(df=old_cluster_8, mapping=old_mapping_8)
clean_and_add_standardized_labels(df=old_cluster_12, mapping=old_mapping_12)
clean_and_add_standardized_labels(df=old_cluster_15, mapping=old_mapping_15)

clean_and_add_standardized_labels(df=cluster_10, mapping=mapping_10)
clean_and_add_standardized_labels(df=cluster_11, mapping=mapping_11)
clean_and_add_standardized_labels(df=cluster_2, mapping=mapping_2)
clean_and_add_standardized_labels(df=cluster_3, mapping=mapping_3)
clean_and_add_standardized_labels(df=cluster_4, mapping=mapping_4)
clean_and_add_standardized_labels(df=cluster_5, mapping=mapping_5)
clean_and_add_standardized_labels(df=cluster_7, mapping=mapping_7)
clean_and_add_standardized_labels(df=cluster_8, mapping=mapping_8)
clean_and_add_standardized_labels(df=cluster_12, mapping=mapping_12)
clean_and_add_standardized_labels(df=cluster_15, mapping=mapping_15)

# %%


def plot_scatter_fixed_dual(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    for i in range(5):
        mask = df.label == i
        ax[0].scatter(df.loc[mask, "mean_cloud_base_height"], df.loc[mask, "mean_cloud_depth"])
        mask = df.label_standardized == i
        ax[1].scatter(df.loc[mask, "mean_cloud_base_height"], df.loc[mask, "mean_cloud_depth"])


plot_scatter_fixed_dual(cluster_2)
plot_scatter_fixed_dual(cluster_3)
plot_scatter_fixed_dual(cluster_4)
plot_scatter_fixed_dual(cluster_5)
plot_scatter_fixed_dual(cluster_7)
plot_scatter_fixed_dual(cluster_8)
plot_scatter_fixed_dual(cluster_10)
plot_scatter_fixed_dual(cluster_11)
plot_scatter_fixed_dual(cluster_12)
plot_scatter_fixed_dual(cluster_15)
# %%

cluster_all = cluster_2[["start_time", "end_time"]].copy()
### The old assignments
# # cluster_all["2"] = cluster_2.label_standardized  # not the best at putting things in null
# # cluster_all["3"] = cluster_3.label_standardized  # most independant measuremenst
# # cluster_all["4"] = cluster_4.label_standardized  # not the best at putting things in null
# # cluster_all["5"] = cluster_5.label_standardized  # too lenient on thick periods
# # cluster_all["7"] = cluster_7.label_standardized
# # cluster_all["8"] = cluster_8.label_standardized
# # cluster_all["10"] = cluster_10.label_standardized
# # cluster_all["11"] = cluster_11.label_standardized
# # cluster_all["12"] = cluster_12.label_standardized  # not the best at putting things in null
# # cluster_all["15"] = cluster_15.label_standardized  # not the best at putting things in null


# cluster_all["2"] = cluster_2.label_standardized # some weird clustering
# cluster_all["3"] = cluster_3.label_standardized # most independant again
cluster_all["4"] = cluster_4.label_standardized
# cluster_all["5"] = cluster_5.label_standardized # bad at clustering thick periods
cluster_all["7"] = cluster_7.label_standardized
cluster_all["8"] = cluster_8.label_standardized
cluster_all["10"] = cluster_10.label_standardized
cluster_all["11"] = cluster_11.label_standardized  # misclassified one null set
# cluster_all["12"] = cluster_12.label_standardized # also quite independant
# cluster_all["15"] = cluster_15.label_standardized # This is bad at null periods

cluster_all["num_unique"] = cluster_all.apply(lambda x: len(np.unique(x.iloc[2:].to_numpy())), axis=1)

# labels_all = cluster_all.iloc[:,2:].to_numpy()
# all_same = np.zeros(labels_all.shape[0], dtype="bool")
# for i in range(all_same.shape[0]):
#     all_same[i] = len(np.unique(labels_all[i,:])) == 1

# cluster_diff = cluster_all.loc[~all_same]

# %%
plt.figure()
plt.hist(cluster_all.num_unique)


cluster_diff = cluster_all.loc[cluster_all.num_unique > 1].reset_index(drop=True).copy()
row_start = 0
row_end = 300
plt.figure(figsize=(10, 3))
plt.grid(alpha=0.6)
plt.ylim(-1, 5)
plt.yticks([0, 1, 2, 3, 4])
tab10 = get_cmap("tab10")
for i, row in cluster_diff.iterrows():
    # if i < row_start or i > row_end:
    #     continue
    colour = tab10(i % 10)
    labels = row.iloc[2:-1].to_numpy()
    for j in np.unique(labels):
        percent = np.sum(labels == j) / len(labels)
        plt.scatter(i, j, s=percent * 50, color=colour)

        if percent == 1 / len(labels) and j == 0:
            print(row.start_time)
# %%
# labels_arr = cluster_all.iloc[:, 2:-1].to_numpy()

# only_2_different = (cluster_all["num_unique"] == 2).to_numpy()
# only_3_different = (cluster_all["num_unique"] == 3).to_numpy()

percent_diff = np.zeros((len(cluster_all), 3))
# single_outlier = np.ones(len(cluster_all))*-1
single_outliers = []
for i, row in cluster_all.iterrows():
    labels = row.iloc[2:-1].to_numpy()
    idx = 0
    for j in np.unique(labels):
        percent = np.sum(labels == j) / len(labels)
        percent_diff[i, idx] = percent
        if percent == 1 / len(labels):
            key = np.argwhere(labels == j)[0][0] + 2
            single_outliers.append([i, j, key])
        idx += 1
    # if np.any(percent_diff[i, :] == 1/len(labels)):
    #     single_outlier[i] =

single_outliers = np.array(single_outliers)
plt.hist(
    single_outliers[:, 2],
    bins=np.linspace(-0.5, len(cluster_all.columns) + 0.5, len(cluster_all.columns) + 2),
)

# %%
mask = single_outliers[:, 2] == 4
single_outliers[mask]
idx = single_outliers[mask, 0]

# %%

old_mapping_9 = {3: 2, 2: 0, 1: 4, 0: 1}
mapping_9 = {3: 0, 2: 4, 1: 2, 0: 1}

numb = "9"
mapping_old = old_mapping_9
mapping_new = mapping_9


cluster_old = pd.read_csv(
    f"./figures/old_clusters/data_cluster_analysis_{numb}.csv", parse_dates=["start_time", "end_time"]
)

cluster_new = pd.read_csv(
    f"./figures/cluster_analysis_{numb}/data.csv", parse_dates=["start_time", "end_time"]
)

# cluster_other = pd.read_csv(
#     f"./figures/old_clusters/data_cluster_analysis_9.csv", parse_dates=["start_time", "end_time"]
# )

clean_and_add_standardized_labels(df=cluster_old, mapping=mapping_old)
clean_and_add_standardized_labels(df=cluster_new, mapping=mapping_new)
# clean_and_add_standardized_labels(df=cluster_other, mapping=mapping_9)

plt.scatter(cluster_old.start_time, cluster_old.label_standardized, s=1)
plt.scatter(cluster_new.start_time, cluster_new.label_standardized + 0.25, s=1)
# plt.scatter(cluster_other.start_time, cluster_other.label_standardized + 0.125, s=1)

mask = cluster_old.label_standardized != cluster_new.label_standardized
for i in range(len(mask)):
    if mask[i]:
        plt.plot(
            [cluster_old.loc[i, "start_time"], cluster_new.loc[i, "start_time"]],
            [cluster_old.loc[i, "label_standardized"], cluster_new.loc[i, "label_standardized"] + 0.25],
            linewidth=0.4,
        )
print(np.sum(cluster_old.label_standardized != cluster_new.label_standardized))
