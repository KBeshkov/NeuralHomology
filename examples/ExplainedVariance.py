import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-8] + "src")
from Algorithms import *
from metrics import *
import seaborn as sns
import matplotlib

plt.style.use("default")  # ../misc/report.mplstyle')
cust_cmap = sns.color_palette("flare_r", as_cmap=True)
cust_cmap2 = matplotlib.cm.get_cmap("Set2")  # .colors
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
cm = 1 / 2.54
plt.ioff()
#%%Initialize analysis hyperparameters
data_folder = "/Users/constb/Data/PachitariuData/"
subfolders = next(os.walk(data_folder))[1]

save_path = "/Users/constb/Figures/NeuralHomology/"


stim_res = 126
stim_bins = np.linspace(0, 2 * np.pi + 2 * np.pi / (stim_res + 4), stim_res + 2)[
    :-1
]  # makes bins of degree differences
stim_bins_degrees = 180 * stim_bins / np.pi

reducer = PCA() 
isomap_red = Isomap(n_components=2)
tsne_red = TSNE(n_components=2)
umap_red = umap.UMAP(n_components=2)

explained_variance_2 = []
explained_variance_10 = []
n_intgs = np.arange(1,len(stim_bins))
spectrum = 1/n_intgs
spectrum = spectrum/np.sum(spectrum)
exp_vars = []
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
        avg_responses = (avg_responses-np.mean(avg_responses))/np.std(avg_responses)
        red_mfld = reducer.fit(avg_responses)
        explained_var_dat = red_mfld.explained_variance_ratio_
        exp_vars.append(explained_var_dat)
        
        
        explained_variance_2.append(np.sum(explained_var_dat[:2]))
        explained_variance_10.append(np.sum(explained_var_dat[:10]))
        
        isomap_mfld = isomap_red.fit_transform(avg_responses).T
        tsne_mfld = tsne_red.fit_transform(avg_responses).T
        umap_mfld = umap_red.fit_transform(avg_responses).T
        
        plt.figure(figsize=(5*cm,5*cm))
        plt.scatter(isomap_mfld[0],isomap_mfld[1],cmap='seismic',c=stim_bins[:-1])
        plt.axis('off')
        plt.savefig(save_path+stim_type+mouse_name+'isomap.png',transparent=True,dpi=500)

        
        plt.figure(figsize=(5*cm,5*cm))
        plt.scatter(tsne_mfld[0],tsne_mfld[1],cmap='seismic',c=stim_bins[:-1])
        plt.axis('off')
        plt.savefig(save_path+stim_type+mouse_name+'tsne.png',transparent=True,dpi=500)

        
        plt.figure(figsize=(5*cm,5*cm))
        plt.scatter(umap_mfld[0],umap_mfld[1],cmap='seismic',c=stim_bins[:-1])
        plt.axis('off')
        plt.savefig(save_path+stim_type+mouse_name+'umap.png',transparent=True,dpi=500)

#%%   
exp_vars = np.array(exp_vars)
exp_mean = np.mean(exp_vars,0)
exp_std = np.std(exp_vars,0)
        
plt.figure(figsize=(5*cm,5*cm))
#plt.plot(n_intgs,spectrum,'k')
plt.plot(n_intgs,exp_mean,'k')
plt.fill_between(n_intgs,exp_mean+exp_std,exp_mean-exp_std,color='gray')
plt.ylim(1e-4,1)
plt.yscale('log')
plt.xscale('log')
plt.grid('on')
plt.tight_layout()
plt.savefig(save_path+'exp_var.png',transparent=True,dpi=500)
        
#%%
print('Explained variance by top 2 components '+str(round(np.mean(explained_variance_2),2))+' $\pm$ ' + str(round(np.std(explained_variance_2),2)))
print('Explained variance by top 2 components '+str(round(np.mean(explained_variance_10),2))+' $\pm$ ' + str(round(np.std(explained_variance_10),2)))
