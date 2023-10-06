#%%Fit Von Mises functions
import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-8] + "src")
from Algorithms import *
from metrics import *
import pickle
import time
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 
plt.style.use("default")  # ../misc/report.mplstyle')
cust_cmap = sns.color_palette("flare_r", as_cmap=True)
cust_cmap2 = cm.get_cmap("Set2").colors
cm = 1 / 2.54
plt.rcParams["font.family"] = "Arial"
#plt.rcParams.update({'figure.max_open_warning': 0})

#%%Initialize analysis hyperparameters
data_folder = "/Users/constb/Data/PachitariuData/"
data_vm_folder = "/Users/constb/Data/NeuralHomology/"
subfolders = next(os.walk(data_folder))[1]

save_path = "/Users/constb/Figures/NeuralHomology/"

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
Homologizer = Persistent_Homology()
dim_red = 0  # determines whether to do dimensionality reduction

half_circle = 'full'

stim_res = 126
stim_bins = np.linspace(0, 2 * np.pi + 2 * np.pi / (stim_res + 4), stim_res + 2)[
    :-1
]  # makes bins of degree differences
stim_bins_degrees = 180 * stim_bins / np.pi


n_perms = 100 # determines how many permutation to do in the statistical test
pvalue = 0.1 / 29  # Bonferonni corrected pvalue in percent


r2_perc = 75 #percentile threshold selecting the best predicted cells
angle_spacing = np.pi/16 #determines how many clusters to consider
min_spacing = np.pi/2
min_peak_ratio = 0.70 #determines the ratio between peaks needed for a tuning curve to be considered bimodal
 

significant_manifolds = {}
#%%Loop over all datasets
for stim_type in subfolders:
    files = next(os.walk(data_folder + stim_type))[2]
    for f_name in files:     
        
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
        
        
        if 'von_Mises_'+f_name+'.pkl' in list(os.walk(data_vm_folder))[0][2]:
            f =  open(data_vm_folder+'von_Mises_'+f_name+'.pkl','rb')
            von_mises_tuning_curves, von_mises_1st, von_mises_2nd, tuning_parameters, r2_scores = pickle.load(f)
            
            r2_thresh = 0.6#np.percentile(r2_scores,r2_perc)
            selected_tun_curves = avg_responses[:,r2_scores>r2_thresh]
            selected_fit_curves = von_mises_tuning_curves[:,r2_scores>r2_thresh]
            selected_parameters = tuning_parameters[r2_scores>r2_thresh,:]
            vm1_selected = von_mises_1st[:,r2_scores>r2_thresh] - np.mean(von_mises_1st[:,r2_scores>r2_thresh],0)
            vm1_selected[vm1_selected<0] = 0
            vm2_selected = von_mises_2nd[:,r2_scores>r2_thresh] - np.mean(von_mises_2nd[:,r2_scores>r2_thresh],0)
            vm2_selected[vm2_selected<0] = 0
            
            peak_ratio = np.max(vm1_selected,0)/np.max(vm2_selected,0)
            peak_ratio_inv = 1/peak_ratio
            peak_ratio = np.min(np.vstack([peak_ratio, peak_ratio_inv]),0)
            
            angle_difference = np.arccos(np.cos(selected_parameters[:,0]-selected_parameters[:,1]))

            edges_centers = np.arange(min_spacing,np.pi+angle_spacing,2*angle_spacing)#[:-1]
            unimod = np.where(np.logical_or(peak_ratio<min_peak_ratio, angle_difference<min_spacing-angle_spacing))[0]
            bimod = np.where(np.logical_and(peak_ratio>min_peak_ratio, np.logical_and(angle_difference>edges_centers[0], angle_difference<edges_centers[-1]-angle_spacing)))[0]
            lastmod = np.where(np.logical_and(peak_ratio>min_peak_ratio, np.logical_and(angle_difference>edges_centers[-1]-angle_spacing, angle_difference<edges_centers[-1]+angle_spacing)))[0]
            cell_labels_list = [unimod, bimod, lastmod]
            clust_responses = [selected_tun_curves[:,unimod], selected_tun_curves[:,bimod], selected_tun_curves[:,lastmod]]
            clust_responses_fitted = [selected_fit_curves[:,unimod], selected_fit_curves[:,bimod], selected_fit_curves[:,lastmod]]
                    
            for i, resps in enumerate(clust_responses):
                if half_circle == "half":
                    resps = resps[:int(stim_res / 2)+1]
                try:
                    pvalue_hom = Homologizer.perm_test(resps,metric,n_perms=n_perms,pval=pvalue)
    
                    significant_manifolds[stim_type + "_" + mouse_name+'pop_'+str(i+1)] = sum(
                        pvalue_hom[1] > pvalue_hom[0])
                except:
                    significant_manifolds[stim_type + "_" + mouse_name+'pop_'+str(i+1)] = 0
                print(len(resps.T))
                
            with open(data_vm_folder+'/significant_manifolds_von_Mises_'+metric_type+'_'+half_circle+'.pkl', 'wb') as f:
                pickle.dump(significant_manifolds, f)

        else:
            # Compute von Mises fits
            start = time.time()
            von_mises_tuning_curves = np.zeros(np.shape(avg_responses))
            von_mises_1st = np.zeros(np.shape(avg_responses))
            von_mises_2nd = np.zeros(np.shape(avg_responses))
            tuning_parameters = np.zeros([len(avg_responses.T),7])
            r2_scores = np.zeros(len(avg_responses.T))
            for n in range(len(avg_responses.T)):
                f, f1, f2, params = fit_von_mises(stim_bins[:-1],avg_responses[:,n]/np.linalg.norm(avg_responses[:,n]))
                von_mises_tuning_curves[:,n] = f*np.linalg.norm(avg_responses[:,n])
                von_mises_1st[:,n] = f1
                von_mises_2nd[:,n] = f2
                tuning_parameters[n] = params
                r2_scores[n] = r2_score(avg_responses[:,n],von_mises_tuning_curves[:,n])
                if n%500==0:
                    print(n)
            vm_list = [von_mises_tuning_curves, von_mises_1st, von_mises_2nd, tuning_parameters, r2_scores]
            plt.figure(figsize=(4*cm,4*cm))
            plt.hist(r2_scores,100)
            
            with open(
                data_vm_folder + "von_Mises_"
                + f_name
                + ".pkl",
                "wb"
            ) as f:
                    pickle.dump(vm_list, f)
            print(round(time.time()-start,1))


#%%
hom_subpop =  pickle.load(open(data_vm_folder+'/significant_manifolds_von_Mises_'+metric_type+'_'+half_circle+'.pkl','rb'))
signif_names = [i for i in hom_subpop.keys() if i[-1]=='1']
n_subpops = int(len(hom_subpop)/29)
hist_spacing = 1/n_subpops

significant_mflds = [[] for i in range(n_subpops)]
for hom in hom_subpop:
    significant_mflds[int(hom[-1])-1].append(hom_subpop[hom])
significant_mflds = [significant_mflds[0], 
                     significant_mflds[2], 
                     significant_mflds[1]]

stim_names = ['Static_','Short','Local','Minnie','Drifting','Noisy','LowContrast']
stim_cmap = ['#873d1bff','#d66634ff','#3f8d2fff','#76cb65ff','#315992ff','#4c7dc3ff','#7b9fd2ff']
prev_hist_data = [[] for i in range(n_subpops)]
plt.figure(figsize=(8 * cm, 6 * cm))
for i, s in enumerate(stim_names):
    stim_indices = [s in cur_stim for cur_stim in signif_names]
    
    curr_hist_data = [np.concatenate([np.array(significant_mflds[i])[stim_indices],prev_hist_data[i]])
                      for i in range(n_subpops)]
    for j, dat in enumerate(curr_hist_data):
        dat[dat>3] = 3 #compress cases with more than 3 features to last bin
        dat = np.histogram(dat,[0,1,2,3,4])[0]
        plt.bar(
            [j*hist_spacing, 1+hist_spacing+j*hist_spacing, 
             2*(1+hist_spacing)+j*hist_spacing, 
             3*(1+hist_spacing)+j*hist_spacing],
            dat,
            color=[stim_cmap[i]],
            zorder=20-i,
            width=1.25*hist_spacing/2
        )

        if i==len(stim_names)-1:
            plt.bar(
                [j*hist_spacing, 1+hist_spacing+j*hist_spacing, 
                 2*(1+hist_spacing)+j*hist_spacing, 
                 3*(1+hist_spacing)+j*hist_spacing],
                dat,
                color=(0,0,0,0),
                zorder=100,
                width=1.25*hist_spacing/2,
                edgecolor=cust_cmap2[j],
                linewidth=1
            )

    prev_hist_data = curr_hist_data
plt.xticks([])
plt.ylim(0, 27)
plt.savefig(
    save_path
    + "von_Mises"
    + half_circle
    + metric_type
    + "subset_cohom_features.png",
    transparent=True,
    dpi=500,
    bbox_inches="tight",
)  # ,figsize=(8*cm,6*cm))

#%%
legend = plt.legend(
    ["D cells", "O cells", "DD cells"],
    loc="upper left",
    fontsize=8,
    framealpha=0,
    frameon=False,
    bbox_to_anchor=(1, 1),
)
for i in range(3):
    legend.legendHandles[i].set_color([0,0,0,0])
    legend.legendHandles[i].set_edgecolor(cust_cmap2[i])
    legend.legendHandles[i].set_linewidth(2.5)
fig = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(
    "/Users/constb/Figures/NeuralHomology/legend_cells_vM.png",
    bbox_inches=bbox,
    transparent=True,
    dpi=500,
)  # ,figsize=(8*cm,6*cm))

    
#%% Explore R2 distributions

r2_distrib = {}
for stim_type in subfolders:
    files = next(os.walk(data_folder + stim_type))[2]
    for f_name in files:     
        
        mouse_name = f_name[-20:-17]
        print("Loaded stimulus " + stim_type + " for mouse " + mouse_name)
        
        if 'von_Mises_'+f_name+'.pkl' in list(os.walk(data_vm_folder))[0][2]:
            f =  open(data_vm_folder+'von_Mises_'+f_name+'.pkl','rb')
            von_mises_tuning_curves, von_mises_1st, von_mises_2nd, tuning_parameters, r2_scores = pickle.load(f)
            r2_distrib[f_name] = r2_scores
    
