# %%
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.append("./")
from WRFMultiDataset import MIRAMultiDataset, MIRAMultiDatasetFactory

# %%
MIRA_factory = MIRAMultiDatasetFactory(Path("./data/metadata_MIRA_trimmed.csv"))
start_time = pd.Timestamp(year=2024, month=10, day=12)
end_time = pd.Timestamp(year=2024, month=12, day=26)
MIRA = MIRA_factory.get_dataset(start_time=start_time, end_time=end_time)

MIRA_features = np.load(
    "/home/waseem/CHOPIN_analysis/data/MIRA_extracted_features/20241012-20241231_4_features.npy"
)
MIRA_feature_times = pd.Series(
    [pd.Timedelta(x, "s") + pd.Timestamp(year=2024, month=10, day=1) for x in MIRA_features[:, 0]]
)
# %%

splits = pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta(12, "h"))
data = []
too_little_cluster = []
elevation = MIRA.variables("elv")
assert MIRA.clean_mask is not None
i = 0
range_mask = MIRA.range < 10000
for i in range(1, len(splits)):
    if (splits[i - 1] >= pd.Timestamp(year=2024, month=10, day=16, hour=9)) & (
        splits[i - 1] < pd.Timestamp(year=2024, month=10, day=17, hour=9)
    ):
        continue
    mask = (MIRA.times >= splits[i - 1]) & (MIRA.times < splits[i])
    refl = MIRA.refl[mask, :].copy()
    clean_mask = MIRA.clean_mask[mask, :]
    if refl.shape[0] == 0:
        continue

    mask_elv = elevation[mask] > 85

    refl[clean_mask == 1] = np.nan
    refl_up = refl[mask_elv]
    refl_up[np.isinf(refl_up)] = np.nan

    if refl_up.shape[0] == 0:
        continue

    # time_cloud_percent = np.sum(np.any(~np.isnan(refl_up), axis=1)) / refl_up.shape[0]
    # total_cloud_percent = np.sum(~np.isnan(refl_up[:, range_mask])) / (
    #     refl_up.shape[0] * refl_up[:, range_mask].shape[1]
    # )
    total_cloud_percent = np.sum(~np.isnan(refl_up)) / (refl_up.shape[0] * refl_up.shape[1])

    mean_refl = np.nanmean(refl_up)
    std_refl = np.nanstd(refl_up)

    if np.all(np.isnan(refl_up)):
        print(f"oddity in {splits[i - 1]} {splits[i]}")
        mean_refl = -65
        std_refl = 0

    # if total_cloud_percent < 0.01:
    #     too_little_cluster.append(
    #         {
    #             "start_time": splits[i - 1],
    #             "end_time": splits[i],
    #             "total_cloud_percent": total_cloud_percent,
    #         }
    #     )
    #     continue
    mask_feautres = (MIRA_feature_times >= splits[i - 1]) & (MIRA_feature_times < splits[i])

    period_features = MIRA_features[mask_feautres, :]
    if period_features.shape[0] > 0:
        time_cloud_percent = period_features.shape[0] / refl_up.shape[0]
        mean_cloud_base_height = np.mean(period_features[:, 1])
        std_cloud_base_height = np.std(period_features[:, 1])
        max_cloud_top_height = np.max(period_features[:, 2])
        mean_cloud_top_height = np.mean(period_features[:, 2])
        std_cloud_top_height = np.std(period_features[:, 2])
        mean_cloud_depth = np.mean(period_features[:, 3])
        std_cloud_depth = np.std(period_features[:, 3])
        mean_base_refl = np.mean(period_features[:, 4])
        std_base_refl = np.mean(period_features[:, 4])
    else:
        time_cloud_percent = 0
        mean_cloud_base_height = 0
        std_cloud_base_height = 0
        max_cloud_top_height = 0
        mean_cloud_top_height = 0
        std_cloud_top_height = 0
        mean_cloud_depth = 0
        std_cloud_depth = 0
        mean_base_refl = -65
        std_base_refl = 0

    data.append(
        {
            "start_time": splits[i - 1],
            "end_time": splits[i],
            "time_cloud_percent": time_cloud_percent,
            "total_cloud_percent": total_cloud_percent,
            "mean_refl": mean_refl,
            "std_refl": std_refl,
            "mean_cloud_base_height": mean_cloud_base_height,
            "std_cloud_base_height": std_cloud_base_height,
            "max_cloud_top_height": max_cloud_top_height,
            "mean_cloud_top_height": mean_cloud_top_height,
            "std_cloud_top_height": std_cloud_top_height,
            "mean_cloud_depth": mean_cloud_depth,
            "std_cloud_depth": std_cloud_depth,
            "mean_base_refl": mean_base_refl,
            "std_base_refl": std_base_refl,
        }
    )
df = pd.DataFrame(data)
# small_cluster = pd.DataFrame(too_little_cluster)
# plt.hist(np.log10(df.total_cloud_percent*100),bins = np.linspace(-1.625, 2.125, 16))


def plot_cluster(
    row,
    MIRA_dataset: MIRAMultiDataset,
    cluster_name: str,
    analysis_number: int,
    features: np.ndarray | None = None,
    feature_times: np.ndarray | None = None,
):
    tick_locs = mdates.drange(row.start_time, row.end_time + pd.Timedelta(1, "h"), pd.Timedelta(3, "hours"))
    tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

    plt.figure(figsize=(8, 5))
    plt.title(
        f"{row.start_time.strftime(r'%Y%m%d')}-{row.end_time.strftime(r'%Y%m%d')}, Cluster: {cluster_name}"
    )
    elevation = MIRA.variables("elv")
    mask = (MIRA_dataset.times >= row.start_time) & (MIRA_dataset.times < row.end_time) & (elevation > 85)
    assert MIRA.clean_mask is not None
    im0 = plt.pcolormesh(
        MIRA.times[mask][::10],
        MIRA.range / 1000,
        MIRA.refl[mask][::10, :].T,
        vmin=-60,
        vmax=35,
        cmap="jet",
    )
    cbar = plt.colorbar(im0)
    cbar.set_label("Reflectivity [dBZ]")
    plt.ylim(0, 12)
    plt.ylabel("Altitude [km]")
    plt.title("MIRA")
    plt.xticks(tick_locs, tick_labels)
    plt.savefig(
        f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/cluster_{cluster_name}_{row.start_time.strftime(r'%Y%m%d%H')}-{row.end_time.strftime(r'%Y%m%d%H')}"
    )

    plt.close()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dn = dendrogram(linkage_matrix, **kwargs)
    dn["color_list"] = [
        "dimgrey",
        "mediumturqiose",
        "sandybrown",
        "orchid",
        "lightpink",
        "greenyellow",
        "sienna",
    ]


# %%

# # mask_1 = df.mean_cloud_base_height > df.mean_cloud_top_height
# # df_slice = df.loc[mask_1]
# # idx = 0

# # tick_locs = mdates.drange(
# #     df_slice.iloc[idx].start_time,
# #     df_slice.iloc[idx].end_time + pd.Timedelta(1, "h"),
# #     pd.Timedelta(6, "hours"),
# # )
# # tick_labels = [mdates.num2date(t).strftime("%d/%m" + "\n" + "%H:%M") for t in tick_locs]

# # plt.figure()

# # mask = (MIRA.times >= df_slice.iloc[idx].start_time) & (MIRA.times < df_slice.iloc[idx].end_time)
# # features_mask = (MIRA_feature_times >= df_slice.iloc[idx].start_time) & (
# #     MIRA_feature_times < df_slice.iloc[idx].end_time
# # )

# # im0 = plt.pcolormesh(
# #     MIRA.times[mask][::10],
# #     MIRA.range / 1000,
# #     MIRA.refl[mask][::10, :].T,
# #     vmin=-60,
# #     vmax=35,
# #     cmap="jet",
# # )
# # cbar = plt.colorbar(im0)
# # cbar.set_label("Reflectivity [dBZ]")
# # plt.ylim(0, 12)
# # plt.ylabel("Altitude [km]")
# # plt.title("MIRA")
# # plt.xticks(tick_locs, tick_labels)
# # plt.plot(MIRA_feature_times[features_mask], MIRA_features[features_mask, 1] / 1000)
# # plt.plot(MIRA_feature_times[features_mask], MIRA_features[features_mask, 2] / 1000)

# # plt.axhline(df_slice.iloc[idx]["mean_cloud_base_height"] / 1000, 0, 1, color="k")
# # plt.axhline(df_slice.iloc[idx]["mean_cloud_top_height"] / 1000, 0, 1, color="r")

# # # plt.savefig(
# # #     f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/cluster_{cluster_name}_{row.start_time.strftime(r'%Y%m%d%H')}-{row.end_time.strftime(r'%Y%m%d%H')}"
# # # )
# %%

columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 6
analysis_number = 21

features = df.loc[:, columns_to_use].to_numpy()
features_mean = np.mean(features, axis=0)
features_std = np.std(features, axis=0)
X = (features - features_mean) / features_std
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(X)

cmap = plt.get_cmap("turbo", 8)
link_colors = []
for i in [5, 4, 0, 1, 2, 3]:
    link_colors.append(mcolors.rgb2hex(cmap(i)))
set_link_color_palette(link_colors)
plt.figure(figsize=(7, 3))
plot_dendrogram(clustering, truncate_mode="level", p=10, color_threshold=8)
plt.ylabel("Distance", fontsize=12)
plt.xticks([60, 300, 500, 800, 1145, 1300], ["6", "5", "1", "2", "3", "4"], rotation=0, fontsize=10)
plt.xlabel("Cluster ID", fontsize=12)
# plt.savefig(
#     f"/home/waseem/CHOPIN_analysis/figures/final_plots/dendrogram_coloured.png", bbox_inches="tight", dpi=200
# )
# plt.close()

# %%
clustering_final = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
df["label"] = clustering_final.labels_
mapping_21 = {5: 0, 2: 1, 4: 2, 0: 3, 1: 4, 3: 5}
label_mapping = mapping_21
df["label_standardized"] = -1
for key, val in label_mapping.items():
    df.loc[df.label == key, "label_standardized"] = val
label_arr = df.label_standardized.to_numpy()
plt.figure()
for i in range(n_clusters):
    mask = label_arr == i
    plt.scatter(
        df.loc[mask, "mean_cloud_base_height"],
        df.loc[mask, "mean_cloud_depth"],
        label=f"Cluster {i+1}",
        color=cmap(i),
    )
plt.ylabel("Cloud Depth")
plt.xlabel("Cloud Base Height")

# %%

# for i in range(n_clusters):
#     df_slice = df.loc[df.label == i]
#     for _, row in df_slice.iterrows():
#         plot_cluster(row, MIRA_dataset=MIRA, cluster_name=f"{i}")

# for _, row in small_cluster.iterrows():
#     plot_cluster(row, MIRA_dataset=MIRA, cluster_name=f"5")

# %%
# PCA Analysis
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

cov_matrix = np.zeros((len(columns_to_use), n_components))
cov_p_value = np.zeros((len(columns_to_use), n_components))
for i in range(cov_matrix.shape[0]):
    for j in range(cov_matrix.shape[1]):
        corr, pval = pearsonr(X[:, i], X_pca[:, j])
        cov_matrix[i, j] = corr
        cov_p_value[i, j] = pval

# %%
legend_size = 10
small_size = 11
medium_size = 12
large_size = 13
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=legend_size)  # legend fontsize
plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
plt.rc("axes", titlesize=medium_size)

fig = plt.figure(figsize=(3.5, 6))
annot_labels = []
for i in range(cov_matrix.shape[0]):
    sub_list = []
    for j in range(cov_matrix.shape[1]):
        if cov_p_value[i, j] <= 0.01:
            sub_list.append(f"{cov_matrix[i,j]:.2f}**")
        elif cov_p_value[i, j] <= 0.05:
            sub_list.append(f"{cov_matrix[i,j]:.2f}* ")
        else:
            sub_list.append(f"{cov_matrix[i,j]:.2f}  ")
    annot_labels.append(sub_list)
ax = sns.heatmap(
    cov_matrix,
    xticklabels=[f"PC {x+1}" for x in range(n_components)],
    yticklabels=columns_to_use,
    annot=np.array(annot_labels),
    fmt="",
    linewidth=0.5,
    cmap="RdBu",
    annot_kws={"fontsize": 12},
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "Correlation"},
)
ax.figure.axes[-1].yaxis.label.set_size(12)
plt.yticks(
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    [
        "Time Cloud\nPercent",
        "Total Cloud\nPercent",
        "Mean Reflectivity",
        "Mean Cloud\nBase Height",
        "Mean Cloud\nTop Height",
        "Mean Cloud\nDepth",
        "Mean Base\nReflecitivity",
    ],
)
plt.title("PCA correlations")
plt.savefig(
    f"/home/waseem/CHOPIN_analysis/figures/final_plots/PCA_correlations.png", bbox_inches="tight", dpi=200
)
# plt.close()

fig = plt.figure(figsize=(4.75, 5.5))

label_arr = df.label_standardized.to_numpy()
for i in range(n_clusters):
    mask = label_arr == i
    plt.scatter(X_pca[mask, 1], X_pca[mask, 0], label=f"Cluster {i+1}", color=cmap(i))

plt.ylabel(f"PC 1 (Explained Var. {100*pca.explained_variance_ratio_[0]:.2f}%)")
plt.xlabel(f"PC 2 (Explained Var. {100*pca.explained_variance_ratio_[1]:.2f}%)")
plt.legend()
plt.savefig(
    f"/home/waseem/CHOPIN_analysis/figures/final_plots/cluster_PCA_scatter.png", bbox_inches="tight", dpi=200
)

print(fig.get_size_inches())


def cluster_analysis(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    MIRA_dataset: MIRAMultiDataset,
    analysis_number: int,
    n_clusters: int,
):
    save_path = Path(f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}")
    if not save_path.exists():
        save_path.mkdir(parents=True)
    _df = feature_df.copy()
    features = _df.loc[:, feature_columns].to_numpy()
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    X = (features - features_mean) / features_std
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(X)

    plot_dendrogram(clustering, truncate_mode="level", p=100)
    plt.title(f"Dendrogram: Cluster Analysis {analysis_number}")
    plt.savefig(f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/dendrogram.png")
    plt.close()
    clustering_final = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    _df["label"] = clustering_final.labels_

    MIRA.refl[MIRA.clean_mask] = np.nan

    for i in range(n_clusters):
        df_slice = _df.loc[_df.label == i]
        for _, row in df_slice.iterrows():
            plot_cluster(row, MIRA_dataset=MIRA_dataset, cluster_name=f"{i}", analysis_number=analysis_number)

    # PCA Analysis
    n_components = 3
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    cov_matrix = np.zeros((len(feature_columns), n_components))
    cov_p_value = np.zeros((len(feature_columns), n_components))
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            corr, pval = pearsonr(X[:, i], X_pca[:, j])
            cov_matrix[i, j] = corr
            cov_p_value[i, j] = pval

    sns.heatmap(
        cov_matrix,
        xticklabels=[f"PCA {x+1}" for x in range(n_components)],
        yticklabels=feature_columns,
        annot=cov_p_value,
        linewidth=0.5,
        cmap="PuOr_r",
        vmin=-1,
        vmax=1,
    )
    plt.title("PCA correlations")
    plt.savefig(
        f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/PCA_correlations.png",
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()

    label_arr = _df.label.to_numpy()
    for i in range(n_clusters):
        mask = label_arr == i
        plt.scatter(X_pca[mask, 1], X_pca[mask, 0], label=f"Cluster {i}")

    plt.ylabel(f"PCA 1 (Exp. Var. {pca.explained_variance_ratio_[0]})")
    plt.xlabel(f"PCA 2 (Exp. Var. {pca.explained_variance_ratio_[1]})")

    plt.savefig(
        f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/Cluster_scatter.png"
    )
    plt.close()

    plt.figure()
    for i in range(n_clusters):
        mask = label_arr == i
        plt.scatter(
            _df.loc[mask, "mean_cloud_base_height"],
            _df.loc[mask, "mean_cloud_depth"],
            label=f"Cluster {i}",
        )
    plt.ylabel("Cloud Top Height")
    plt.xlabel("Cloud Base Height")
    plt.savefig(
        f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/Cluster_scatter_fixed.png"
    )
    plt.close()

    return_columns = ["start_time", "end_time", "label"]
    return_columns.extend(feature_columns)
    df_slice = _df.loc[:, return_columns]
    df_slice.to_csv(f"/home/waseem/CHOPIN_analysis/figures/cluster_analysis_{analysis_number}/data.csv")


# %%
columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    "max_cloud_top_height",
    # "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]

features = df.loc[:, columns_to_use].to_numpy()
features_mean = np.mean(features, axis=0)
features_std = np.std(features, axis=0)
X = (features - features_mean) / features_std
sil_scores = []
for i in range(2, 14):
    kmean_cluster = KMeans(n_clusters=i, random_state=10).fit(X)
    sil_score = silhouette_score(X, kmean_cluster.labels_)
    sil_scores.append({"ncluster": i, "score": sil_score})

temp_df = pd.DataFrame(sil_scores)
plt.plot(temp_df.ncluster, temp_df.score)

# %%
# Run multiple analyses
df = pd.DataFrame(data)


columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 3
analysis_number = 1
print("Performing Analysis 1")

# # cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 2
print("Performing Analysis 2")

# # cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    "max_cloud_top_height",
    # "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 3
print("Performing Analysis 3")

# # cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 4
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    # "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 5
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 4
analysis_number = 6
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 7
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 8
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 4
analysis_number = 9
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 10
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    # "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 11
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    "mean_refl",
    "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 12
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 4
analysis_number = 13
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    # "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 4
analysis_number = 14
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

columns_to_use = [
    # "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 15
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent_2",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 5
analysis_number = 16
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 4
analysis_number = 17
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 4
analysis_number = 18
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    # "time_cloud_percent",
    # "total_cloud_percent",
    # "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 7
analysis_number = 19
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 6
analysis_number = 20
print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

# Max data 12-26
columns_to_use = [
    "time_cloud_percent",
    "total_cloud_percent",
    "mean_refl",
    # "std_refl",
    "mean_cloud_base_height",
    # "std_cloud_base_height",
    # "max_cloud_top_height",
    "mean_cloud_top_height",
    # "std_cloud_top_height",
    "mean_cloud_depth",
    # "std_cloud_depth",
    "mean_base_refl",
    # "std_base_refl",
]
n_clusters = 6
analysis_number = 21
print(f"Performing Analysis {analysis_number}")

cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

# # Max data 12-26
# columns_to_use = [
#     "time_cloud_percent",
#     "total_cloud_percent",
#     "mean_refl",
#     # "std_refl",
#     "mean_cloud_base_height",
#     # "std_cloud_base_height",
#     # "max_cloud_top_height",
#     "mean_cloud_top_height",
#     # "std_cloud_top_height",
#     "mean_cloud_depth",
#     # "std_cloud_depth",
#     "mean_base_refl",
#     # "std_base_refl",
# ]
# n_clusters = 7
# analysis_number = 22
# print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)

# # Max data 12-26
# columns_to_use = [
#     "time_cloud_percent",
#     "total_cloud_percent",
#     "mean_refl",
#     # "std_refl",
#     "mean_cloud_base_height",
#     # "std_cloud_base_height",
#     # "max_cloud_top_height",
#     "mean_cloud_top_height",
#     # "std_cloud_top_height",
#     "mean_cloud_depth",
#     # "std_cloud_depth",
#     "mean_base_refl",
#     # "std_base_refl",
# ]
# n_clusters = 8
# analysis_number = 23
# print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)


# # Max data 12-26
# columns_to_use = [
#     "time_cloud_percent",
#     "total_cloud_percent",
#     "mean_refl",
#     # "std_refl",
#     "mean_cloud_base_height",
#     # "std_cloud_base_height",
#     # "max_cloud_top_height",
#     "mean_cloud_top_height",
#     # "std_cloud_top_height",
#     "mean_cloud_depth",
#     # "std_cloud_depth",
#     "mean_base_refl",
#     # "std_base_refl",
# ]
# n_clusters = 9
# analysis_number = 24
# print(f"Performing Analysis {analysis_number}")

# cluster_analysis(df, columns_to_use, MIRA, analysis_number, n_clusters)
