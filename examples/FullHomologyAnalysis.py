#%%Run homology analysis for all datasets
import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-8] + "src")
from Algorithms import *
from metrics import *
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 
plt.style.use("default")  # ../misc/report.mplstyle')
cust_cmap = sns.color_palette("flare_r", as_cmap=True)
cust_cmap2 = cm.get_cmap("Set2").colors
cm = 1 / 2.54
plt.rcParams["font.family"] = "Arial"
#plt.rcParams.update({'figure.max_open_warning': 0})
# plt.ioff()
#%%Initialize analysis hyperparameters
data_folder = "/Users/constb/Data/PachitariuData/"
subfolders = next(os.walk(data_folder))[1]

metric_type = "GeodesicKNN"  # determines the metric that will be used throughout the homology calculations
kn = 4#''#
if metric_type == "Euclidean":
    metric = pairwise_distances
elif metric_type == "Geodesic":
    metric = geodesic
elif metric_type == "GeodesicKNN":
    metric = geodesic_knn(k=kn, adaptive=True).fit
elif metric_type == "Implicit":
    metric = implicit_metric_circle

half_circle = "half"
save_path = "/Users/constb/Figures/NeuralHomology/" + metric_type + str(kn) + "/"


stim_res = 126
stim_bins = np.linspace(0, 2 * np.pi + 2 * np.pi / (stim_res + 4), stim_res + 2)[
    :-1
]  # makes bins of degree differences
stim_bins_degrees = 180 * stim_bins / np.pi
N_sub = 100 #determines the number of subset neurons to choose from each class
dim_red = 0  # determines whether to do dimensionality reduction
dmult = 0.5  # determines where to take a cross-section
n_perms = 100 # determines how many permutation to do in the statistical test
pvalue = 0.1 / 29  # Bonferonni corrected pvalue in percent
reducer = PCA(
    n_components=2
)  # umap.UMAP(n_neighbors=50,min_dist=0,n_components=2) #method for dimensinoality reduction plots


Homologizer = Persistent_Homology()
BA = Barcode_Analyzer()

significant_manifolds = {}
significant_uni = {}
significant_bi = {}
significant_nonlin = {}
significant_noise = {}

decoding_360 = {'DO': [],'O': [],'D': [],'untuned': []}
decoding_180 = {'DO': [],'O': [],'D': [],'untuned': []}
decoding_difference = {'DO': [],'O': [],'D': [],'untuned': []}
#%%Loop over all datasets
for stim_type in subfolders:
    files = next(os.walk(data_folder + stim_type))[2]
    for f_name in files:
        # Loading and binning data
        mouse_name = f_name[-20:-17]
        resp = np.load(data_folder + stim_type + "/" + f_name, allow_pickle=True).item()
        print("Loaded stimulus " + stim_type + " for mouse " + mouse_name)
        responses = resp["sresp"]
        indices = resp["istim"]
        if len(responses.T)>len(indices):
            responses = responses[:,:len(indices)]
        avg_responses = np.zeros([len(stim_bins) - 1, len(responses)])
        for c in range(len(stim_bins) - 1):
            bin_indx = np.where(
                np.logical_and(stim_bins[c] < indices, indices < stim_bins[c + 1])
            )[0]
            avg_responses[c] = np.nanmean(responses[:, bin_indx], 1)
        # Calculate the cohomology groups and perform significance testing
        hom_data = Homologizer(avg_responses, metric, False, dim_red, [1, None])
        pvalue_hom = Homologizer.perm_test(
            avg_responses, metric, n_perms=100, pval=100 - pvalue
        )
        significant_manifolds[stim_type + "_" + mouse_name] = sum(
            pvalue_hom[1] > pvalue_hom[0]
        )

        # Choose a crossection for plots
        try:
            cycle_lengths = hom_data[1][1][:, 1] - hom_data[1][1][:, 0]
            max_cycle = np.argsort(cycle_lengths).astype(int)[-1]
            hom_crossection = hom_data[1][1][max_cycle][0] + dmult * (
                hom_data[1][1][max_cycle][1] - hom_data[1][1][max_cycle][0]
            )
            Homologizer.barcode_plot(hom_data[1], pval=pvalue_hom[0],figsize=(4.2*cm,3.3*cm))
            plt.xticks([])
            plt.suptitle(stim_type+' '+mouse_name,fontsize=10)
            plt.savefig(
                save_path
                + stim_type
                + "_"
                + mouse_name
                + "_barcode_"
                + metric_type
                + str(kn)
                + ".png",
                dpi=500, transparent=True
            )
        except:
            cycle_lengths = hom_data[1][0][:, 1] - hom_data[1][0][:, 0]
            max_cycle = np.argsort(cycle_lengths).astype(int)[-1]
            hom_crossection = hom_data[1][0][max_cycle][0] + dmult * (
                hom_data[1][0][max_cycle][1] - hom_data[1][0][max_cycle][0]
            )
            Homologizer.barcode_plot(hom_data[1], 1, pval=pvalue_hom[0],figsize=(4.2*cm,3.3*cm))
            plt.savefig(
                save_path
                + stim_type
                + "_"
                + mouse_name
                + "_barcode_"
                + metric_type
                + str(kn)
                + ".png", transparent=True,
                dpi=500,
            )

        # Visualize the manifold with dimensionality reduction
        ori_mfld = reducer.fit_transform(avg_responses)
        fig = plt.figure()
        plt.subplot(121)
        plt.scatter(ori_mfld[:, 0], ori_mfld[:, 1], cmap="seismic", c=stim_bins[:-1])
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(hom_data[0])
        plt.savefig(
            save_path
            + stim_type
            + "_"
            + mouse_name
            + "reduced"
            + metric_type
            + str(kn)
            + ".png",
            dpi=1000,
        )

        # Visualize the manifold with a polar graph and thresholded edges
        edge_dist = np.copy(hom_data[0])
        edge_dist[edge_dist > hom_crossection] = 0
        plt.figure(dpi=200)
        plt.subplot(121)
        BA.plotCocycle2D(
            hom_data[0],
            ori_mfld[:, :2],
            hom_crossection,
            labels=stim_bins_degrees,
            node_cmap="seismic",
        )
        plt.subplot(122)
        plt.imshow(edge_dist, extent=[0, 360, 360, 0], cmap=cust_cmap)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(
            save_path
            + stim_type
            + "_"
            + mouse_name
            + "polar"
            + metric_type
            + str(kn)
            + ".png",
            dpi=1000,
        )

        if stim_type != "StaticSin":
            # Cluster neurons by their different selectivity properties
            osi = orientation_selectivity_index(avg_responses, stim_bins[:-1])
            dsi = direction_selectivity_index(avg_responses, stim_bins[:-1])

            osi_centered = osi - 0.5
            dsi_centered = dsi - 0.5
            si_2d = np.vstack([osi_centered, dsi_centered]).T
            xy_boundaries = (0.3, 0.3) 

            if N_sub==0:
                unimodal_neurons = np.where(
                    np.logical_and(
                        osi_centered > xy_boundaries[0], dsi_centered > xy_boundaries[1]
                    )
                )[0]  # clust_labels==0)[0]#
                bimodal_neurons = np.where(
                    np.logical_and(
                        osi_centered > xy_boundaries[0], dsi_centered < -xy_boundaries[1]
                    )
                )[0]  # clust_labels==1)[0]#
                nonlinear_neurons = np.where(
                    np.logical_and(
                        osi_centered < -xy_boundaries[0], dsi_centered > xy_boundaries[1]
                    )
                )[0]  # clust_labels==2)[0]#
                noise_neurons = np.where(
                    np.logical_and(
                        osi_centered < -xy_boundaries[0], dsi_centered < -xy_boundaries[1]
                    )
                )[0]  # clust_labels==3)[0]#
            
            elif N_sub>0:
                xy_boundaries = (0, 0) #do not do osi/dsi thresholding
                unimodal_neurons = unimodal_neurons[np.argsort(np.linalg.norm(si_2d[unimodal_neurons],ord=1,axis=1))[-N_sub:]]
                bimodal_neurons = bimodal_neurons[np.argsort(np.linalg.norm(si_2d[bimodal_neurons],ord=1,axis=1))[-N_sub:]]
                nonlinear_neurons = nonlinear_neurons[np.argsort(np.linalg.norm(si_2d[nonlinear_neurons],ord=1,axis=1))[-N_sub:]]
                noise_neurons = noise_neurons[np.argsort(np.linalg.norm(si_2d[noise_neurons],ord=1,axis=1))[-N_sub:]]

            color_labels = np.zeros(len(osi_centered))
            color_labels[unimodal_neurons] = 1
            color_labels[bimodal_neurons] = 2
            color_labels[nonlinear_neurons] = 3
            color_labels[noise_neurons] = 4

            uni_responses = avg_responses[:, unimodal_neurons]
            bi_responses = avg_responses[:, bimodal_neurons]
            nonlinear_responses = avg_responses[:, nonlinear_neurons]
            noise_responses = avg_responses[:, noise_neurons]

            if half_circle == "half":
                uni_responses = uni_responses[
                    :, np.argsort(np.argmax(uni_responses, 0))
                ][: int(stim_res / 2)]
                bi_responses = bi_responses[:, np.argsort(np.argmax(bi_responses, 0))][
                    : int(stim_res / 2)
                ]
                nonlinear_responses = nonlinear_responses[
                    :, np.argsort(np.argmax(nonlinear_responses, 0))
                ][: int(stim_res / 2)]
                noise_responses = noise_responses[
                    :, np.argsort(np.argmax(noise_responses, 0))
                ][: int(stim_res / 2)]

            #plot some random tunning curves from the two populations
            fig = plt.figure(dpi=200,figsize=(6,6))
            for i in range(3):
                fig.add_subplot(4,3,i+1,projection='polar')
                plt.polar(stim_bins[:-1],uni_responses[:,np.random.randint(len(uni_responses.T))],color=cust_cmap2[2],linewidth=1)
                plt.xticks([0,np.pi/2,np.pi,(3/2)*np.pi])
            for i in range(3):
                fig.add_subplot(4,3,i+4,projection='polar')
                plt.polar(stim_bins[:-1],bi_responses[:,np.random.randint(len(bi_responses.T))],color=cust_cmap2[4],linewidth=1)
                plt.xticks([0,np.pi/2,np.pi,(3/2)*np.pi])
            for i in range(3):
                fig.add_subplot(4,3,i+7,projection='polar')
                plt.polar(stim_bins[:-1],nonlinear_responses[:,np.random.randint(len(nonlinear_responses.T))],color=cust_cmap2[6],linewidth=1)
                plt.xticks([0,np.pi/2,np.pi,(3/2)*np.pi])
            for i in range(3):
                fig.add_subplot(4,3,i+10,projection='polar')
                plt.polar(stim_bins[:-1],noise_responses[:,np.random.randint(len(noise_responses.T))],color=cust_cmap2[7],linewidth=1)
                plt.xticks([0,np.pi/2,np.pi,(3/2)*np.pi])
            plt.tight_layout()
            plt.savefig(save_path+stim_type+'_'+mouse_name+'tunning'+metric_type+str(kn)+'.png',dpi=1000)

            # Compute the cohomology groups on the cell subsets
            hom_uni = Homologizer(uni_responses, metric, False, dim_red, [1, None])
            hom_bi = Homologizer(bi_responses, metric, False, dim_red, [1, None])
            hom_nonlin = Homologizer(nonlinear_responses, metric, False, dim_red, [1, None])
            hom_noise = Homologizer(noise_responses, metric, False, dim_red, [1, None])

            pvalue_hom_uni = Homologizer.perm_test(
                uni_responses, metric, n_perms=n_perms, pval=100 - pvalue
            )
            significant_uni[stim_type + "_" + mouse_name] = sum(
                pvalue_hom_uni[1] > pvalue_hom_uni[0]
            )
            pvalue_hom_bi = Homologizer.perm_test(
                bi_responses, metric, n_perms=n_perms, pval=100 - pvalue
            )
            significant_bi[stim_type + "_" + mouse_name] = sum(
                pvalue_hom_bi[1] > pvalue_hom_bi[0]
            )
            pvalue_hom_nonlin = Homologizer.perm_test(
                nonlinear_responses, metric, n_perms=n_perms, pval=100 - pvalue
            )
            significant_nonlin[stim_type + "_" + mouse_name] = sum(
                pvalue_hom_nonlin[1] > pvalue_hom_nonlin[0]
            )
            pvalue_hom_noise = Homologizer.perm_test(
                noise_responses, metric, n_perms=n_perms, pval=100 - pvalue
            )
            significant_noise[stim_type + "_" + mouse_name] = sum(
                pvalue_hom_noise[1] > pvalue_hom_noise[0]
            )
            
            try:
                uni_cycle = np.argsort(hom_uni[1][1][:, 1] - hom_uni[1][1][:, 0]).astype(int)[-1]
            except:
                uni_cycle = 0
            try:
                bi_cycle = np.argsort(hom_bi[1][1][:, 1] - hom_bi[1][1][:, 0]).astype(int)[-1]
            except:
                bi_cycle = 0
            try:
                nonlin_cycle = np.argsort(hom_nonlin[1][1][:, 1] - hom_nonlin[1][1][:, 0]).astype(int)[-1]
            except:
                nonlin_cycle = 0
            try:
                noise_cycle = np.argsort(hom_noise[1][1][:, 1] - hom_noise[1][1][:, 0]).astype(int)[-1]
            except:
                noise_cycle = 0

            uni_crossection = hom_uni[1][1][uni_cycle][0] + dmult * (
                hom_uni[1][1][uni_cycle][1] - hom_uni[1][1][uni_cycle][0]
            )
            bi_crossection = hom_bi[1][1][bi_cycle][0] + dmult * (
                hom_bi[1][1][bi_cycle][1] - hom_bi[1][1][bi_cycle][0]
            )
            nonlin_crossection = hom_nonlin[1][1][nonlin_cycle][0] + dmult * (
                hom_nonlin[1][1][nonlin_cycle][1] - hom_nonlin[1][1][nonlin_cycle][0]
            )
            noise_crossection = hom_noise[1][1][noise_cycle][0] + dmult * (
                hom_noise[1][1][noise_cycle][1] - hom_noise[1][1][noise_cycle][0]
            )

            Homologizer.barcode_plot(hom_uni[1],2,pval=pvalue_hom_uni[0])
            plt.savefig(save_path+stim_type+'_'+mouse_name+'uni_barcode'+metric_type+str(kn)+'.png',dpi=1000)

            plt.figure()
            BA.plotCocycle2D(hom_uni[0],uni_responses[:,:2],uni_crossection,labels=stim_bins_degrees,node_cmap='seismic')
            plt.axis('off')
            plt.savefig(save_path+stim_type+'_'+mouse_name+'uni_hom'+metric_type+str(kn)+'.png',dpi=1000)

            Homologizer.barcode_plot(hom_bi[1],2,pval=pvalue_hom_bi[0])
            plt.savefig(save_path+stim_type+'_'+mouse_name+'bi_barcode'+metric_type+str(kn)+'.png',dpi=1000)
            plt.figure()
            BA.plotCocycle2D(hom_bi[0],bi_responses[:,:2],bi_crossection,labels=stim_bins_degrees,node_cmap='seismic')
            plt.axis('off')
            plt.savefig(save_path+stim_type+'_'+mouse_name+'bi_hom'+metric_type+str(kn)+'.png',dpi=1000)

            Homologizer.barcode_plot(hom_nonlin[1],2,pval=pvalue_hom_nonlin[0])
            plt.savefig(save_path+stim_type+'_'+mouse_name+'nonlin_barcode'+metric_type+str(kn)+'.png',dpi=1000)
            plt.figure()
            BA.plotCocycle2D(hom_nonlin[0],nonlinear_responses[:,:2],nonlin_crossection,labels=stim_bins_degrees,node_cmap='seismic')
            plt.axis('off')
            plt.savefig(save_path+stim_type+'_'+mouse_name+'nonlin_hom'+metric_type+str(kn)+'.png',dpi=1000)

            Homologizer.barcode_plot(hom_noise[1],2,pval=pvalue_hom_noise[0])
            plt.savefig(save_path+stim_type+'_'+mouse_name+'noise_barcode'+metric_type+str(kn)+'.png',dpi=1000)
            plt.figure()
            BA.plotCocycle2D(hom_noise[0],noise_responses[:,:2],noise_crossection,labels=stim_bins_degrees,node_cmap='seismic')
            plt.axis('off')
            plt.savefig(save_path+stim_type+'_'+mouse_name+'noise_hom'+metric_type+str(kn)+'.png',dpi=1000)

            with open('/Users/constb/Data/NeuralHomology/significant_manifolds'+metric_type+'.pkl', 'wb') as f:
                pickle.dump(significant_manifolds, f)

            #Compare decoding accuracy from each subpopulation
            X_train, X_test, y_train, y_test = train_test_split(
                responses.T, indices, test_size=0.2,random_state=0)
            X_train_180, X_test_180, y_train_180, y_test_180 = train_test_split(
                responses.T, indices%np.pi, test_size=0.2,random_state=0)
            DO_reg = LinearRegression().fit(X_train[:,unimodal_neurons],y_train)
            O_reg = LinearRegression().fit(X_train[:,bimodal_neurons],y_train)
            D_reg = LinearRegression().fit(X_train[:,nonlinear_neurons],y_train)
            untuned_reg = LinearRegression().fit(X_train[:,noise_neurons],y_train)
            
            DO_reg_180 = LinearRegression().fit(X_train_180[:,unimodal_neurons],y_train_180)
            O_reg_180 = LinearRegression().fit(X_train_180[:,bimodal_neurons],y_train_180)
            D_reg_180 = LinearRegression().fit(X_train_180[:,nonlinear_neurons],y_train_180)
            untuned_reg_180 = LinearRegression().fit(X_train_180[:,noise_neurons],y_train_180)
            
            decoding_360['DO'].append(DO_reg.score(X_test[:,unimodal_neurons],y_test))
            decoding_360['O'].append(O_reg.score(X_test[:,bimodal_neurons],y_test))
            decoding_360['D'].append(D_reg.score(X_test[:,nonlinear_neurons],y_test))
            decoding_360['untuned'].append(untuned_reg.score(X_test[:,noise_neurons],y_test))
            
            decoding_180['DO'].append(DO_reg_180.score(X_test_180[:,unimodal_neurons],y_test_180))
            decoding_180['O'].append(O_reg_180.score(X_test_180[:,bimodal_neurons],y_test_180))
            decoding_180['D'].append(D_reg_180.score(X_test_180[:,nonlinear_neurons],y_test_180))
            decoding_180['untuned'].append(untuned_reg_180.score(X_test_180[:,noise_neurons],y_test_180))
            
            decoding_difference['DO'].append(DO_reg.score(X_test[:,unimodal_neurons],y_test)-DO_reg_180.score(X_test_180[:,unimodal_neurons],y_test_180))
            decoding_difference['O'].append(O_reg.score(X_test[:,bimodal_neurons],y_test)-O_reg_180.score(X_test_180[:,bimodal_neurons],y_test_180))
            decoding_difference['D'].append(D_reg.score(X_test[:,nonlinear_neurons],y_test)-D_reg_180.score(X_test_180[:,nonlinear_neurons],y_test_180))
            decoding_difference['untuned'].append(untuned_reg.score(X_test[:,noise_neurons],y_test)-untuned_reg_180.score(X_test_180[:,noise_neurons],y_test_180))

        with open("/Users/constb/Data/NeuralHomology/decoding_360_"+str(N_sub)+".pkl","wb") as f:
            pickle.dump(decoding_360, f)
            
        with open("/Users/constb/Data/NeuralHomology/decoding_180_"+str(N_sub)+".pkl","wb") as f:
            pickle.dump(decoding_180, f)
            
        with open("/Users/constb/Data/NeuralHomology/decoding_difference_"+str(N_sub)+".pkl","wb") as f:
            pickle.dump(decoding_difference, f)
            
        with open(
            "/Users/constb/Data/NeuralHomology/significant_unimanifolds_"
            + half_circle
            + metric_type
            + str(N_sub)+".pkl",
            "wb",
        ) as f:
            pickle.dump(significant_uni, f)
        with open(
            "/Users/constb/Data/NeuralHomology/significant_bimanifolds_"
            + half_circle
            + metric_type
            + str(N_sub)+".pkl",
            "wb",
        ) as f:
            pickle.dump(significant_bi, f)
        with open(
            "/Users/constb/Data/NeuralHomology/significant_nonlinmanifolds_"
            + half_circle
            + metric_type
            + str(N_sub)+".pkl",
            "wb",
        ) as f:
            pickle.dump(significant_nonlin, f)
        with open(
            "/Users/constb/Data/NeuralHomology/significant_noisemanifolds_"
            + half_circle
            + metric_type
            +str(N_sub)+ ".pkl",
            "wb",
        ) as f:
            pickle.dump(significant_noise, f)
for i in range(1000):
    plt.close()
plt.ion()
#%%
plt.rc("axes", axisbelow=True)

with open(
    "/Users/constb/Data/NeuralHomology/significant_manifolds" + metric_type + ".pkl",
    "rb",
) as f:
    significant_manifolds = pickle.load(f)

plt.figure(figsize=(10 * cm, 6 * cm))
# plt.grid('on')
if metric_type == "GeodesicKNN":
    plt.hist(
        significant_manifolds.values(),
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        color="forestgreen",
        edgecolor="black",
        linewidth=3,
    )  # maroon for Euclidean, teal for Geodesic
else:
    significant_manifolds_array = np.array(list(significant_manifolds.values()))
    significant_manifolds_array[
        np.array(list(significant_manifolds.values())) > 3
    ] = 3  # compress all manifolds with more than three circles into one bin
    plt.hist(
        significant_manifolds_array,
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        color="maroon",
        edgecolor="black",
        linewidth=3,
    )  # maroon for Euclidean, teal for Geodesic
plt.xticks([])
plt.ylim(0, 30)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/" + metric_type + "cohom_features.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight"
)


#%%
sorted_responses = avg_responses[:, np.argsort(np.argmax(avg_responses, 0))]

plt.figure(figsize=(8 * cm, 6.5 * cm))
plt.imshow(sorted_responses.T, aspect="auto", vmax=100, cmap="seismic")
plt.xlabel("$\\theta$")
plt.ylabel("$r_i$",rotation=0)
plt.xticks(
    [0, 32, 64, 96],
    [
        0,
        90,
        180,
        270,
    ],
)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/angle_sorted_responses.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight"
)

#%%
from scipy.ndimage.filters import gaussian_filter1d

n_traces = 4
rand_ids = np.random.randint(0,len(sorted_responses.T),[n_traces])
for i in range(n_traces):
    smoothed_signal = gaussian_filter1d(sorted_responses[:,rand_ids[i]],2)
    plt.figure(figsize=(3 * cm, 0.5 * cm))
    plt.plot(stim_bins_degrees[:-1],smoothed_signal,linewidth=1.25,color='black')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        "/Users/constb/Figures/NeuralHomology/angle_sorted_trace"+str(i+1)+".png",
        transparent=True,
        dpi=500,
        bbox_inches="tight"
    )
#%%

with open(
    "/Users/constb/Data/NeuralHomology/significant_unimanifolds_"
    + half_circle
    + metric_type
    + str(N_sub) + ".pkl",
    "rb",
) as f:
    significant_uni = list(pickle.load(f).values())
with open(
    "/Users/constb/Data/NeuralHomology/significant_bimanifolds_"
    + half_circle
    + metric_type
    + str(N_sub) + ".pkl",
    "rb",
) as f:
    significant_bi = list(pickle.load(f).values())
with open(
    "/Users/constb/Data/NeuralHomology/significant_nonlinmanifolds_"
    + half_circle
    + metric_type
    + str(N_sub) + ".pkl",
    "rb",
) as f:
    significant_nonlin = list(pickle.load(f).values())
with open(
    "/Users/constb/Data/NeuralHomology/significant_noisemanifolds_"
    + half_circle
    + metric_type
    + str(N_sub) + ".pkl",
    "rb",
) as f:
    significant_noise = list(pickle.load(f).values())


plt.figure(figsize=(8 * cm, 6 * cm))
plt.hist(
    [significant_uni, significant_bi, significant_nonlin, significant_noise],
    [-0.5, 0.5, 1.5, 2.5, 3.5],
    color=cust_cmap2[:4],
    linewidth=3,
)
plt.xticks([])
plt.ylim(0, 27)
plt.savefig(
    "/Users/constb/Figures/NeuralHomology/"
    + half_circle
    + metric_type
    + str(N_sub)
    + "subset_cohom_features.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight",
)  # ,figsize=(8*cm,6*cm))

#%%
legend = plt.legend(
    ["DO cells", "O cells", "D cells", "Untuned cells"],
    loc="upper left",
    fontsize=8,
    framealpha=1,
    frameon=False,
    bbox_to_anchor=(1, 1),
)
fig = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(
    "/Users/constb/Figures/NeuralHomology/legend_cells.png",
    bbox_inches=bbox,
    transparent=True,
    dpi=500,
)  # ,figsize=(8*cm,6*cm))


#%% Decoding plots
N_sub = 100
with open("/Users/constb/Data/NeuralHomology/decoding_360_"+str(N_sub)+".pkl", "rb") as f:
# with open("/Users/constb/Data/NeuralHomology/decoding_360.pkl", "rb") as f:
    decoding_360 = list(pickle.load(f).values())
    for i in range(len(decoding_360)):
        inval_indx = np.where(np.array(decoding_360[i])<0)[0]
        correct_array = np.array(decoding_360[i])
        correct_array[inval_indx] = 0
        decoding_360[i] = correct_array

# with open("/Users/constb/Data/NeuralHomology/decoding_180_"+str(N_sub)+".pkl", "rb") as f:
with open("/Users/constb/Data/NeuralHomology/decoding_180.pkl", "rb") as f:
    decoding_180 = list(pickle.load(f).values())
    for i in range(len(decoding_180)):
        inval_indx = np.where(np.array(decoding_180[i])<0)[0]
        correct_array = np.array(decoding_180[i])
        correct_array[inval_indx] = 0
        decoding_180[i] = correct_array    
        
decoding_difference = [decoding_360[i]-decoding_180[i] for i in range(len(decoding_360))]
    

signif_360 = np.zeros([len(decoding_360),len(decoding_360)])
signif_180 = np.zeros([len(decoding_180),len(decoding_180)])
signif_diff = np.zeros([len(decoding_difference),len(decoding_difference)])
bf_correct = lambda x: x/18 #Bonferroni correction for the 18 test that we do
for i in range(len(decoding_360)):
    for j in range(len(decoding_360)):
        if i<j:
            signif_360[i,j] = ranksums(decoding_360[i],decoding_360[j])[1]
            signif_180[i,j] = ranksums(decoding_180[i],decoding_180[j])[1]   
            signif_diff[i,j] = ranksums(decoding_difference[i],decoding_difference[j])[1]


xpos = np.arange(1.5,4.5)
plt.figure(figsize=(6.67*cm,6.67*cm))
violin_plot = plt.violinplot(decoding_360,showmeans=True,)
for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor(cust_cmap2[i])
    pc.set_edgecolor('black')
violin_plot['cbars'].set_edgecolor('black')
violin_plot['cmeans'].set_edgecolor('black')
violin_plot['cmaxes'].set_edgecolor('black')
violin_plot['cmins'].set_edgecolor('black')
for i in range(len(decoding_360)):
    for j in range(len(decoding_360)):
        if i<j:
            lineheight = 0.925-i*0.075
            plt.plot([i+1,j+1],[lineheight,lineheight],'-|',color=cust_cmap2[i])
            if signif_360[i,j]<bf_correct(0.001):
                plt.text(xpos[j-1],lineheight,'***',ha='center',fontsize=12,color=cust_cmap2[j])
            elif signif_360[i,j]<bf_correct(0.01):
                plt.text(xpos[j-1],lineheight,'**',ha='center',fontsize=12,color=cust_cmap2[j])
            elif signif_360[i,j]<bf_correct(0.05):
                plt.text(xpos[j-1],lineheight,'*',ha='center',fontsize=12,color=cust_cmap2[j])
            else:
                plt.text(xpos[j-1],lineheight+0.02,'n.s.',ha='center',fontsize=10,color=cust_cmap2[j])
# plt.xticks(np.arange(1,5),['DO','O','D','untuned'])
plt.xticks([])
plt.ylim(0,1)
plt.grid('on')
plt.savefig("/Users/constb/Figures/NeuralHomology/decoding_360"+str(N_sub)+".png",
    bbox_inches='tight',
    transparent=True,
    dpi=500)

plt.figure(figsize=(6.67*cm,6.67*cm))
violin_plot = plt.violinplot(decoding_180,showmeans=True,)
for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor(cust_cmap2[i])
    pc.set_edgecolor('black')
violin_plot['cbars'].set_edgecolor('black')
violin_plot['cmeans'].set_edgecolor('black')
violin_plot['cmaxes'].set_edgecolor('black')
violin_plot['cmins'].set_edgecolor('black')
for i in range(len(decoding_360)):
    for j in range(len(decoding_360)):
        if i<j:
            lineheight = 0.925-i*0.075
            plt.plot([i+1,j+1],[lineheight,lineheight],'-|',color=cust_cmap2[i])
            if signif_180[i,j]<bf_correct(0.001):
                plt.text(xpos[j-1],lineheight,'***',ha='center',fontsize=12,color=cust_cmap2[j])
            elif signif_180[i,j]<bf_correct(0.01):
                plt.text(xpos[j-1],lineheight,'**',ha='center',fontsize=12,color=cust_cmap2[j])
            elif signif_180[i,j]<bf_correct(0.05):
                plt.text(xpos[j-1],lineheight,'*',ha='center',fontsize=12,color=cust_cmap2[j])
            else:
                plt.text(xpos[j-1],lineheight+0.02,'n.s.',ha='center',fontsize=10,color=cust_cmap2[j])

# plt.xticks(np.arange(1,5),['DO','O','D','untuned'])
plt.xticks([])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['','','','','',''])
plt.ylim(0,1)
plt.grid('on')
plt.savefig("/Users/constb/Figures/NeuralHomology/decoding_180"+str(N_sub)+".png",
    bbox_inches='tight',
    transparent=True,
    dpi=500)

plt.figure(figsize=(6.67*cm,6.67*cm))
violin_plot = plt.violinplot(decoding_difference,showmeans=True,)
for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor(cust_cmap2[i])
    pc.set_edgecolor('black')
violin_plot['cbars'].set_edgecolor('black')
violin_plot['cmeans'].set_edgecolor('black')
violin_plot['cmaxes'].set_edgecolor('black')
violin_plot['cmins'].set_edgecolor('black')
for i in range(len(decoding_difference)):
    for j in range(len(decoding_difference)):
        if i<j:
            lineheight = 0.475-i*0.075
            plt.plot([i+1,j+1],[lineheight,lineheight],'-|',color=cust_cmap2[i])
            if signif_diff[i,j]<bf_correct(0.001):
                plt.text(xpos[j-1],lineheight,'***',ha='center',fontsize=12,color=cust_cmap2[j])
            elif signif_diff[i,j]<bf_correct(0.01):
                plt.text(xpos[j-1],lineheight,'**',ha='center',fontsize=12,color=cust_cmap2[j])
            elif signif_diff[i,j]<bf_correct(0.05):
                plt.text(xpos[j-1],lineheight,'*',ha='center',fontsize=12,color=cust_cmap2[j])
            else:
                plt.text(xpos[j-1],lineheight+0.02,'n.s.',ha='center',fontsize=10,color=cust_cmap2[j])

# plt.xticks(np.arange(1,5),['DO','O','D','untuned'])
plt.xticks([])
plt.ylim(-0.55,0.55)
plt.grid('on')
plt.savefig("/Users/constb/Figures/NeuralHomology/decoding_difference"+str(N_sub)+".png",
    bbox_inches='tight',
    transparent=True,
    dpi=500)