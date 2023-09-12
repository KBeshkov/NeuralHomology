import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-8] + "src")
from Algorithms import *
from metrics import *
import seaborn as sns
import matplotlib

plt.style.use("default")  # ../misc/report.mplstyle')
cust_cmap = sns.color_palette("flare_r", as_cmap=True)
cust_cmap2 = matplotlib.cm.get_cmap("Set2")  # |.colors
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6
cm = 1 / 2.54
#%%Initialize analysis hyperparameters
data_folder = "/Users/constb/Data/PachitariuData/"
subfolders = next(os.walk(data_folder))[1]

save_path = "/Users/constb/Data/NeuralHomology/Sampling/"

sampling_type = "cells"

stim_bins = np.linspace(0, 2 * np.pi + 2 * np.pi / 90, 92)[
    :-1
]  # makes bins of 4 degree differences
stim_bins_degrees = 180 * stim_bins / np.pi
dim_red = 0  # determines whether to do dimensionality reduction
n_perms = 100
pvalue = 0.1 / 29  # Bonferonni corrected pvalue in percent

Homologizer = Persistent_Homology()
BA = Barcode_Analyzer()
geod_metric = geodesic_knn(k=4, adaptive=True).fit
euclid_metric = pairwise_distances

#%%Loop over all datasets
cells_n = []
for stim_type in subfolders:
    files = next(os.walk(data_folder + stim_type))[2]
    for f_name in files:
        # Loading and binning data
        mouse_name = f_name[-20:-17]
        resp = np.load(data_folder + stim_type + "/" + f_name, allow_pickle=True).item()
        print("Loaded stimulus " + stim_type + " for mouse " + mouse_name)
        responses = resp["sresp"]
        indices = resp["istim"]
        responses = responses / np.max(responses)  # normalize responses
        cells_n.append(len(responses))

        if sampling_type == "stim":
            bin_step = 2
            binnings = np.arange(8, 128, bin_step)
            bin_stimuli = lambda n: np.linspace(0, 2 * np.pi + 2 * np.pi / (n - 2), n)[
                :-1
            ]
            avg_responses = [np.zeros([i - 2, len(responses)]) for i in binnings]
            stim_bins = []
            for count, n in enumerate(binnings):
                stim_bins.append(bin_stimuli(n))
                for c in range(len(stim_bins[-1]) - 1):
                    bin_indx = np.where(
                        np.logical_and(
                            stim_bins[-1][c] < indices, indices < stim_bins[-1][c + 1]
                        )
                    )[0]
                    avg_responses[count][c, :] = np.nanmean(responses[:, bin_indx], 1)
        elif sampling_type == "cells":
            bin_num = 50
            binnings = np.logspace(1, 4, bin_num).astype(
                int
            )  # np.arange(5,10000,bin_step)
            stim_bins = np.linspace(0, 2 * np.pi + 2 * np.pi / 90, 92)[:-1]
            avg_binned_responses = np.zeros([len(stim_bins) - 1, len(responses)])
            avg_responses = [np.zeros([len(stim_bins) - 1, i]) for i in binnings]
            for c in range(len(stim_bins) - 1):
                bin_indx = np.where(
                    np.logical_and(stim_bins[c] < indices, indices < stim_bins[c + 1])
                )[0]
                avg_binned_responses[c, :] = np.nanmean(responses[:, bin_indx], 1)
            for count, n in enumerate(binnings):
                rand_idx = np.random.choice(
                    np.arange(0, len(avg_binned_responses.T)), n, replace=False
                )
                avg_responses[count] = avg_binned_responses[:, rand_idx]

        hom_groups_euclid = []
        cycle_lengths_euclid = []
        hom_groups_geod = []
        cycle_lengths_geod = []
        for i in range(len(binnings)):
            hom_groups_euclid.append(
                Homologizer(
                    avg_responses[i], euclid_metric, True, dim_red, [1, None], coeff=2
                )
            )
            cycle_lengths_euclid.append(
                np.sort(
                    hom_groups_euclid[-1][1][1][:, 1]
                    - hom_groups_euclid[-1][1][1][:, 0]
                )
            )
            hom_groups_geod.append(
                Homologizer(
                    avg_responses[i], geod_metric, True, dim_red, [1, None], coeff=2
                )
            )
            cycle_lengths_geod.append(
                np.sort(
                    hom_groups_geod[-1][1][1][:, 1] - hom_groups_geod[-1][1][1][:, 0]
                )
            )

        # extract the largest three (co)cyclces
        top_n = 3
        euclid_cycles = np.zeros([len(binnings), top_n])
        geod_cycles = np.zeros([len(binnings), top_n])
        for i in range(len(binnings)):
            for j in range(top_n):
                try:
                    euclid_cycles[i, j] = cycle_lengths_euclid[i][-j - 1]
                    geod_cycles[i, j] = cycle_lengths_geod[i][-j - 1]
                except:
                    euclid_cycles[i, j] = 0
                    geod_cycles[i, j] = 0
        np.save(
            save_path + stim_type + "_" + mouse_name + sampling_type + "_euclid.npy",
            euclid_cycles,
        )
        np.save(
            save_path + stim_type + "_" + mouse_name + sampling_type + "_geod.npy",
            geod_cycles,
        )


#%% Load extracted curves and plot a baseline generated by a circle
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"
from matplotlib.ticker import FuncFormatter

if sampling_type == "cells":
    S1s = [Manifold_Generator().S1(90, 1).T for i in range(len(binnings))]
else:
    S1s = [Manifold_Generator().S1(binnings[i], 1).T for i in range(len(binnings))]
S1_baseline = [
    Homologizer(S1s[i], geod_metric, True, dim_red, [1, None])
    for i in range(len(binnings))
]
S1_cycles = [
    np.sort(S1_baseline[i][1][1][:, 1] - S1_baseline[i][1][1][:, 0])[0]
    for i in range(len(binnings))
]

# load all 30 datasets
loadfiles = next(os.walk(save_path))
all_euclid_cycles = []
all_geod_cycles = []
for datfile in loadfiles[2]:
    if "euclid" in datfile and sampling_type in datfile:
        all_euclid_cycles.append(np.load(save_path + datfile))
    elif "geod" in datfile and sampling_type in datfile:
        all_geod_cycles.append(np.load(save_path + datfile))
all_euclid_cycles = np.dstack(all_euclid_cycles)
all_geod_cycles = np.dstack(all_geod_cycles)
mean_euclid_cycles = np.mean(all_euclid_cycles, 2)
mean_geod_cycles = np.mean(all_geod_cycles, 2)
std_euclid_cycles = np.std(all_euclid_cycles, 2) / 2
std_geod_cycles = np.std(all_geod_cycles, 2) / 2

fig, ax = plt.subplots(
    figsize=(8 * cm, 6 * cm),
)
if sampling_type=="cells":
    binnings = np.log10(binnings)
plt.plot(binnings, S1_cycles, "k")
for i in range(top_n):
    plt.plot(binnings, mean_euclid_cycles[:, i], color=[0.75, i * 0.3, 0.25, 1])
    plt.fill_between(
        binnings,
        mean_euclid_cycles[:, i] - std_euclid_cycles[:, i],
        mean_euclid_cycles[:, i] + std_euclid_cycles[:, i],
        color=[0.75, i * 0.3, 0.25, 1],
        alpha=0.1,
    )
for i in range(top_n):
    plt.plot(binnings, mean_geod_cycles[:, i], color=[i * 0.3, 0.75, 0.25, 0.5])
    plt.fill_between(
        binnings,
        mean_geod_cycles[:, i] - std_geod_cycles[:, i],
        mean_geod_cycles[:, i] + std_geod_cycles[:, i],
        color=[i * 0.3, 0.75, 0.25, 0.5],
        alpha=0.1,
    )
ax.set_ylim(0, 1)
ax.grid("on")
if sampling_type == "cells":
    ax.set_xticks([2, 3.5])
    ax.set_xticklabels(["{}\u00b2".format(10), "{}\u00b3".format(10)])

    # ax.set_xticks(ax.get_xticks(),)

    # plt.xlabel("#cells")
# else:
    # plt.xlabel("#stimulus bins")
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/" + sampling_type + "_sampling.png",
    bbox_inches="tight",
    transparent=True,
    dpi=500,)

# legend = plt.legend(['$\\beta^1_1(S^1)$',
#             '$\\beta^1_1(X_{Euclid})$','$\\beta^2_1(X_{Euclid})$','$\\beta^3_1(X_{Euclid})$',
#             '$\\beta^1_1(X_{geod})$','$\\beta^2_1(X_{geod})$','$\\beta^3_1(X_{geod})$'],loc='upper left',fontsize=8,
#            framealpha=1, frameon=False,bbox_to_anchor=(1, 1))
# fig  = legend.figure
# fig.canvas.draw()
# bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('/Users/constb/Figures/NeuralHomology/legend.png',bbox_inches=bbox,transparent=True,dpi=500,figsize=(8*cm,6*cm))

#%%Make plots of manifolds

# reducer = PCA(n_components=2)

if sampling_type == "cells":
    indx_samp = 5
    stim_bins = [np.linspace(0, 2 * np.pi + 2 * np.pi / 90, 92)[:-1]] * (indx_samp + 1)
    # low_sampling_mfld = reducer.fit_transform(avg_responses[indx_samp])
    # high_sampling_mfld = reducer.fit_transform(avg_responses[-indx_samp])
    cutoff_euclid = (
        np.sort(hom_groups_euclid[-1][1][1][:, 1] - hom_groups_euclid[-1][1][1][:, 0])[
            -1
        ]
        / 2
    )
    cutoff_geod = (
        np.sort(hom_groups_geod[-1][1][1][:, 1] - hom_groups_geod[-1][1][1][:, 0])[-1]
        / 2
    )
else:
    indx_samp = 5
    # low_sampling_mfld = reducer.fit_transform(avg_responses[indx_samp])
    # high_sampling_mfld = reducer.fit_transform(avg_responses[-indx_samp])
    cutoff_euclid = (
        np.sort(hom_groups_euclid[-1][1][1][:, 1] - hom_groups_euclid[-1][1][1][:, 0])[
            -1
        ]
        / 2
    )
    cutoff_geod = (
        np.sort(hom_groups_geod[-1][1][1][:, 1] - hom_groups_geod[-1][1][1][:, 0])[-1]
        / 2
    )


pctl = 50

low_cutoff = np.percentile(hom_groups_geod[indx_samp][0].flatten(),pctl)
plt.figure(figsize=(10.8 * cm, 10.8 * cm))
BA.plotCocycle2D(
    hom_groups_geod[indx_samp][0],
    avg_responses[indx_samp][:, 0],
    low_cutoff,
    node_cmap="seismic",
    n_labels=0
)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/" + sampling_type + "_low_conn.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight")


mid_indx = 14
mid_cutoff = np.percentile(hom_groups_geod[mid_indx][0].flatten(),pctl)
plt.figure(figsize=(10.8 * cm, 10.8 * cm))
BA.plotCocycle2D(
    hom_groups_geod[mid_indx][0],
    avg_responses[mid_indx][:, 0],
    mid_cutoff,
    node_cmap="seismic",
    n_labels=0
)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/" + sampling_type + "_mid_conn.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight")


upmid_indx = 30
upmid_cutoff = np.percentile(hom_groups_geod[upmid_indx][0].flatten(),pctl)
plt.figure(figsize=(10.8 * cm, 10.8 * cm))
BA.plotCocycle2D(
    hom_groups_geod[upmid_indx][0],
    avg_responses[upmid_indx][:, 0],
    upmid_cutoff,
    node_cmap="seismic",
    n_labels=0
)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/" + sampling_type + "_upmid_conn.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight")

high_cutoff = np.percentile(hom_groups_geod[-indx_samp][0].flatten(),pctl)
plt.figure(figsize=(10.8 * cm, 10.8 * cm))
BA.plotCocycle2D(
    hom_groups_geod[-indx_samp][0],
    avg_responses[-indx_samp][:, 0],
    high_cutoff,
    node_cmap="seismic",
    n_labels=0
)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/" + sampling_type + "_high_conn.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight")
