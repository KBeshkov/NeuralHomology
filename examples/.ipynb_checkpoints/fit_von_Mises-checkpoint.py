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

half_circle = 'half'

stim_res = 126
stim_bins = np.linspace(0, 2 * np.pi + 2 * np.pi / (stim_res + 4), stim_res + 2)[
    :-1
]  # makes bins of degree differences
stim_bins_degrees = 180 * stim_bins / np.pi


n_perms = 100 # determines how many permutation to do in the statistical test
pvalue = 0.1 / 29  # Bonferonni corrected pvalue in percent


r2_perc = 50 #percentile threshold selecting the best predicted cells
angle_spacing = np.pi/8 #determines how many clusters to consider
min_spacing = np.pi/2
min_peak_ratio = 0.85 #determines the ratio between peaks needed for a tuning curve to be considered bimodal
 

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
            
            r2_thresh = np.percentile(r2_scores,r2_perc)
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
            cell_labels_list = [unimod]
            clust_responses = [selected_tun_curves[:,unimod]]
            clust_responses_fitted = [selected_fit_curves[:,unimod]]
            for i in range(len(edges_centers)):
                cell_labels_list.append(np.where(np.logical_and(peak_ratio>min_peak_ratio, np.logical_and(angle_difference>edges_centers[i]-angle_spacing, angle_difference<edges_centers[i]+angle_spacing)))[0])
                clust_responses.append(selected_tun_curves[:,cell_labels_list[-1]])
                clust_responses_fitted.append(selected_fit_curves[:,cell_labels_list[-1]])
                
            for i, resps in enumerate(clust_responses):
                if half_circle == "half":
                    resps = resps[:int(stim_res / 2)+1]
                try:
                    pvalue_hom = Homologizer.perm_test(resps,metric,n_perms=n_perms,pval=pvalue)
    
                    significant_manifolds[stim_type + "_" + mouse_name+'pop_'+str(i+1)] = sum(
                        pvalue_hom[1] > pvalue_hom[0])
                except:
                    significant_manifolds[stim_type + "_" + mouse_name+'pop_'+str(i+1)] = 0
                
            with open(data_vm_folder+'/significant_manifolds_von_Mises'+metric_type+'.pkl', 'wb') as f:
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
hom_subpop =  pickle.load(open(data_vm_folder+'/significant_manifolds_von_Mises'+metric_type+'.pkl','rb'))

significant_0 = [hom_subpop[i] for i in hom_subpop if i[-1]=='1']
significant_90 = [hom_subpop[i] for i in hom_subpop if i[-1]=='2']
significant_135 = [hom_subpop[i] for i in hom_subpop if i[-1]=='3']
significant_180 = [hom_subpop[i] for i in hom_subpop if i[-1]=='4']

plt.figure(figsize=(8 * cm, 6 * cm))
plt.hist(
    [significant_0, significant_180, significant_90, significant_135],
    [-0.5, 0.5, 1.5, 2.5, 3.5],
    color=cust_cmap2[:4],
    linewidth=3,
)
plt.xticks([])
#plt.ylim(0, 27)
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

        