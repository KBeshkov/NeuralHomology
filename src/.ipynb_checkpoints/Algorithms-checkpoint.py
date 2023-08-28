# implementation of algorithms
import math
import numpy as np
import time as time
from cmath import sqrt as isqrt
import warnings


from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import seaborn as sns

plt.style.use("default")  # ../misc/report.mplstyle')

import numpy.matlib
import numpy.polynomial as npoly
import random

from scipy import stats
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, jensenshannon
from scipy.special import i0, iv, hyp1f1
from scipy.optimize import linear_sum_assignment, curve_fit
from scipy.stats import (
    rankdata,
    ortho_group,
    percentileofscore,
    spearmanr,
    special_ortho_group,
    ranksums
)
from scipy.signal import find_peaks, argrelextrema
from scipy.spatial.transform import Rotation
from scipy.interpolate import splrep, splev

from sklearn.metrics import pairwise_distances, r2_score, silhouette_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

import networkx as nx
import persim
import persim.landscapes
from ripser import ripser as tda
from persim import plot_diagrams
import umap.umap_ as umap
from perlin_noise import PerlinNoise

import torch
from torch import nn
import torch.optim as optim

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class Manifold_Generator:
    """
    Manifold generator class.
    """

    def __init__(self):
        return None

    def __call__(self, amount_of_points, manifold_type, *args):
        """


        Parameters
        ----------
        amount_of_points : int
            The number of points to be sampled from the manifold.
        manifold_type : string
            String specifying the type of manifold to be sampled from.
        *args : list
            List of args specific to each particualr manifold.

        Returns
        -------
        call
            calls the specified method.

        """
        return getattr(self, manifold_type)(amount_of_points, args)

    def R3(self, amount_of_points, *args):
        """
        Samples equidistant points from $\mathbb{R}^3$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Lower and upper xyz coordinates of the bounding box.

        Returns
        -------
        R : numpy array
            array of xyz coordinates.

        """
        x, y, z = np.meshgrid(
            np.linspace(-args[0], args[0], amount_of_points),
            np.linspace(-args[1], args[1], amount_of_points),
            np.linspace(-args[2], args[2], amount_of_points),
        )
        R = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        return R

    def S1(self, amount_of_points, *args):
        """
        Samples equidistant points from $S^1$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Radius.

        Returns
        -------
        numpy array
            array of xy circle coordinates.

        """
        r, theta = args[0], np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False)
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def S2(self, amount_of_points, *args):
        """
        Samples equidistant points from $S^2$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Radius.

        Returns
        -------
        numpy array
            array of xyz sphere coordinates.

        """
        r, phi, theta = (
            args[0],
            np.linspace(0, np.pi, amount_of_points, endpoint=False),
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
        )
        Phi, Theta = np.meshgrid(phi, theta)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array(
            [
                r * np.cos(Theta) * np.sin(Phi),
                r * np.sin(Theta) * np.sin(Phi),
                r * np.cos(Phi),
            ]
        )

    def T2(self, amount_of_points, *args):
        """
        Samples equidistant points from $T^2$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Small and large torus radius.

        Returns
        -------
        numpy array
            array of xyz torus coordinates.

        """
        R, r, phi, theta = (
            args[0],
            args[1],
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
        )
        Phi, Theta = np.meshgrid(theta, phi)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array(
            [
                (R + r * np.cos(Theta)) * np.cos(Phi),
                (R + r * np.cos(Theta)) * np.sin(Phi),
                r * np.sin(Theta),
            ]
        )

    def Sn(self, amount_of_points, *args):
        """
        Samples random points from an $S^n$ sphere.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Dimension of the sphere.

        Returns
        -------
        numpy array
            array of $[x_1,x_2,...,x_N]$ sphere coordinates.

        """
        x = np.random.randn(amount_of_points, args[0])
        x = x.T / np.linalg.norm(x, axis=1)
        return x

    def KB(self, amount_of_points, immersion, *args):
        """


        Samples points from the Klein bottle.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        immersion: bool
        Whether to use the 3 dimensional immersion or the 4 dimensional embedding.
        *args : list
            R, P and epsilon parameters for the 4 dimensional embedding.

        Returns
        -------
        numpy array
            array of Klein bottle coordinates.

        """
        if immersion == True:
            phi = np.linspace(0, np.pi, amount_of_points)
            theta = np.linspace(0, 2 * np.pi, amount_of_points)
            Phi, Theta = np.meshgrid(phi, theta)
            Phi_KB, Theta_KB = Phi.flatten(), Theta.flatten()
            x = (
                (-2 / 15)
                * np.cos(Phi_KB)
                * (
                    3 * np.cos(Theta_KB)
                    - 30 * np.sin(Phi_KB)
                    + 90 * np.sin(Phi_KB) * np.cos(Phi_KB) ** 2
                    - 60 * np.sin(Phi_KB) * np.cos(Phi_KB) ** 6
                    + 5 * np.cos(Phi_KB) * np.cos(Theta_KB) * np.sin(Phi_KB)
                )
            )
            y = (
                (-1 / 15)
                * np.sin(Phi_KB)
                * (
                    3 * np.cos(Theta_KB)
                    - 3 * np.cos(Theta_KB) * np.cos(Phi_KB) ** 2
                    - 48 * np.cos(Theta_KB) * np.cos(Phi_KB) ** 2
                    + 48 * np.cos(Theta_KB) * np.cos(Phi_KB) ** 6
                    - 60 * np.sin(Phi_KB)
                    + 5 * np.cos(Phi_KB) * np.cos(Theta_KB) * np.sin(Phi_KB)
                    - 5 * np.cos(Theta_KB) * np.sin(Phi_KB) * np.cos(Phi_KB) ** 3
                    - 80 * np.cos(Theta_KB) * np.sin(Phi_KB) * np.cos(Phi_KB) ** 5
                    + 80 * np.cos(Theta_KB) * np.sin(Phi_KB) * np.cos(Phi_KB) ** 7
                )
            )
            z = (2 / 15) * (3 + 5 * np.cos(Phi_KB) * np.sin(Phi_KB)) * np.sin(Theta_KB)
            KB = np.array([x, y, z])
        else:
            theta = np.linspace(0, 2 * np.pi, amount_of_points)
            v = np.linspace(0, 2 * np.pi, amount_of_points)
            Theta, V = np.meshgrid(theta, v)
            Theta, V = Theta.flatten(), V.flatten()
            R, P, eps = args[0], args[1], args[2]
            x = R * (np.cos(Theta / 2) * np.cos(V) + np.sin(Theta / 2) * np.sin(2 * V))
            y = R * (np.sin(Theta / 2) * np.cos(V) + np.cos(Theta / 2) * np.sin(2 * V))
            z = P * np.cos(Theta) * (1 + eps * np.sin(V))
            w = P * np.sin(Theta) * (1 + eps * np.sin(V))
            KB = np.array([x, y, z, w])
        return KB


class Persistent_Homology:
    def __init__(self):
        None

    def __call__(self, manifold, metric, normalized, dimred=0, *args, coeff=2):
        distance_matrix, birth_death_diagram, cocycles = self.homology_analysis(
            manifold, metric, dimred, *args, coeff=coeff
        )
        return [
            distance_matrix,
            self.normalize(birth_death_diagram) if normalized else birth_death_diagram,
            cocycles,
        ]

    def homology_analysis(self, manifold, metric, dimred, *args, coeff=2):
        """
        Computes persistent homology
        -----------------------
        Outputs the distance_matrix and birth_death_diagram for the given manifold and metric.
        """
        if dimred > 0:
            manifold = PCA(n_components=dimred).fit_transform(manifold)
        distance_matrix = metric(manifold)
        birth_death_diagram = tda(
            distance_matrix,
            distance_matrix=True,
            maxdim=args[0][0],
            n_perm=args[0][1],
            do_cocycles=True,
            coeff=coeff,
        )
        diagram = birth_death_diagram["dgms"]
        cocycles = birth_death_diagram["cocycles"]
        return distance_matrix, diagram, cocycles

    def normalize(self, birth_death_diagram):
        """
        Normalized birth/death distance
        -----------------------
        Outputs the persistence of a feature normalized between 0 and 1.
        """
        birth_death_diagram_copy = np.copy(birth_death_diagram)
        a = np.concatenate(birth_death_diagram_copy).flatten()
        finite_dgm = a[np.isfinite(a)]
        ax_min, ax_max = np.min(finite_dgm), np.max(finite_dgm)
        x_r = ax_max - ax_min

        buffer = x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
        yr = y_up - y_down
        b_inf = y_down + yr * 0.95
        norm_pers = []
        for i in range(len(birth_death_diagram)):
            norm_pers.append((birth_death_diagram[i]) / b_inf)
        return norm_pers

    def perm_test(self, manifold, metric, n_perms, dimred=0, dim=1, pval=99,nperms=None):
        """
        Permutation test to estimate a p value for a manifold appearing by chance
        ----------------------
        Outputs p values at different confidence thresholds.
        """
        hom = self.normalize(
            self.homology_analysis(manifold, metric, dimred, [dim, nperms])[1]
        )
        cycle_lengths = hom[dim][:, 1] - hom[dim][:, 0]
        percentiles = []
        for n in range(n_perms):
            permute = np.vstack(
                [np.random.permutation(len(manifold)) for i in range(len(manifold.T))]
            ).T
            temp_manifold = np.vstack(
                [manifold[permute[:, i], i] for i in range(len(manifold.T))]
            ).T
            temp_hom = self.normalize(
                self.homology_analysis(temp_manifold, metric, dimred, [dim, nperms])[1]
            )
            if len(temp_hom[dim]) > 0:
                persistances = temp_hom[dim][:, 1] - temp_hom[dim][:, 0]
                percentiles.append(np.max(persistances))
            else:
                percentiles.append(0)
        pvalue = np.percentile(percentiles, pval)

        print(
            str(sum(cycle_lengths > pvalue)) + " significant " + str(dim) + "-cocycles"
        )
        return pvalue, cycle_lengths, percentiles

    def barcode_plot(
        self, diagram, dims=2, cutoff_ax=0, pval=0, normalize=True, figsize=(14, 6)
    ):
        if normalize:
            diagram = self.normalize(diagram)
        results = {}
        if cutoff_ax == 0:
            largest_pers = 0
            for d in range(dims):
                results["h" + str(d)] = diagram[d]
                if np.max(diagram[d][np.isfinite(diagram[d])]) > largest_pers:
                    largest_pers = np.max(diagram[d][np.isfinite(diagram[d])])
        elif cutoff_ax != 0:
            largest_pers = cutoff_ax
        clrs = cm.get_cmap(
            "Set2"
        ).colors  # ['tab:blue','tab:orange','tab:green','tab:red']#['b','r','g','m','c']
        diagram[0][~np.isfinite(diagram[0])] = largest_pers + 0.1 * largest_pers
        plot_prcnt = 0 * np.ones(dims)
        to_plot = []
        for curr_h, cutoff in zip(diagram, plot_prcnt):
            bar_lens = curr_h[:, 1] - curr_h[:, 0]
            plot_h = curr_h[bar_lens >= np.percentile(bar_lens, cutoff)]
            to_plot.append(plot_h)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(dims, 4)
        for curr_betti, curr_bar in enumerate(to_plot):
            ax = fig.add_subplot(gs[curr_betti, :])
            for i, interval in enumerate(reversed(curr_bar)):
                plt.plot(
                    [interval[0], interval[1]], [i, i], color=clrs[curr_betti], lw=1.5
                )
                if pval > 0 and curr_betti==1:
                    plt.plot(
                        [interval[0], interval[0] + pval],
                        [i, i],
                        color="k",
                        alpha=0.05,
                        lw=2,
                    )
            if curr_betti == dims - 1:
                ax.set_xlim([0, largest_pers + 0.01])
                ax.set_ylim([-1, len(curr_bar)])
                ax.set_yticks([])
            else:
                ax.set_xlim([0, largest_pers + 0.01])
                ax.set_xticks([])
                ax.set_ylim([-1, len(curr_bar)])
                ax.set_yticks([])


class Barcode_Analyzer:
    def __init__(self):
        None

    def histograms(self, barcode, binslst=[20, 15, 10]):
        """
        Betti number persistance histograms
        -----------------------
        Outputs a list of histograms for persistance/birth/death for all specified Betti numbers.
        """
        histograms = []
        persistant_values = []
        for i in range(len(barcode)):
            persistant_values.append(barcode[i][:, 1] - barcode[i][:, 0])
            b_i = np.histogram(
                barcode[i][:, 1] - barcode[i][:, 0],
                np.linspace(0, 0.1, binslst[i]),
                density=True,
            )
            histograms.append(b_i)
        return [persistant_values, histograms]

    def curves(self, barcode, number_of_curves=3, epsilon=0.001, duration=1000):
        """
        Betti curves
        -----------------------
        Outputs a list of Betti curves, indicating how many n-holes there are for a given epsilon value.
        """
        curves = []
        finite_dgm = barcode[0][np.isfinite(barcode[0])]
        x_maxs = []
        for i in range(len(barcode)):
            if len(barcode[i]) > 0 and i != 0:
                x_maxs.append(np.max(barcode[i]))
            else:
                x_maxs.append(np.max(finite_dgm))
        e = np.copy(epsilon)
        for i in range(len(barcode)):
            Bn = np.zeros(duration)
            for j in range(duration):
                count = 0
                for k in range(len(barcode[i][:, 0])):
                    if barcode[i][k, 0] < epsilon and barcode[i][k, 1] > epsilon:
                        count = count + 1
                Bn[j] = count
                epsilon = epsilon + e
            epsilon = np.copy(e)
            curves.append(Bn)
        return [np.linspace(0, epsilon * duration, duration), curves]

    def plotCocycle2D(
        self,
        D,
        X,
        thresh,
        labels=None,
        node_cmap="Greys",
        edge_color="black",
        n_labels=12,
    ):
        """
        Given a 2D point cloud X, display the graph
        at a given threshold "thresh"
        """
        N = X.shape[0]
        c = plt.get_cmap(node_cmap)
        C = c(np.array(np.round(np.linspace(0, 255, N)), dtype=np.int32))
        D_thresh = np.copy(D)
        for i in range(N):
            for j in range(N):
                if D[i, j] > thresh:
                    D_thresh[i, j] = 0
                # plt.plot([X[i,0],X[j,0]],[X[i,1],X[j,1]],color=edge_color,linewidth=1,alpha=0.1)
        G = nx.from_numpy_array(D_thresh / np.linalg.norm(D_thresh))
        circ_layout = nx.circular_layout(G)
        lbl_offset = 1.25
        nx.draw_circular(
            G,
            node_color=np.linspace(0, 1, len(G)),
            cmap=node_cmap,
            node_size=160,
            alpha=0.75,
            edge_color=edge_color,
            linewidths=0.001,
        )
        # Plot vertex labels
        # plt.scatter(X[:, 0], X[:, 1],color='green',edgecolor='black',s=8,linewidths=0.25)
        if n_labels != 0:
            txt_labels = int(N / n_labels)
            for i in range(n_labels):
                plt.text(
                    lbl_offset * circ_layout[i * txt_labels][0],
                    lbl_offset * circ_layout[i * txt_labels][1],
                    str(int(round(labels[i * txt_labels]))),
                    color="black",
                    ha="center",
                    va="center",
                )  # C[i*txt_labels])
        plt.axis("equal")


class Neural_Network:
    """
    Custom neural network class which allows to easily study how manifolds propagate through a neural network.
    """

    def __init__(self, n_neurons, weight_std=1, bias_std=0):
        """
        Parameters
        ----------
        n_neurons : list
            A list of the number of neurons in each layer. [input neurons, hidden neurons, ..., output neurons]
        weight_std : list or float
            The standard deviation of the weights between layers. If it is a list it has to have a size of #layers - 1.
            If it is a scalar the weights between all the layers will have the same standard deviation.
            The default is 1, which corresponds to an edge of chaos regime.
        bias_std: list or float
            The standard deviation of the biases in each layer. If it is a list it has to have a size of #layers-1.
            If it is a sclar the biases of all layers will have the same standard deviation.
            The default is 0, which corresponds to no bias.

        Returns
        -------
        None.
        """

        self.n_neurons = n_neurons
        self.weight_std = weight_std
        self.weights = []
        self.biases = []
        for l in range(len(n_neurons) - 1):
            if type(weight_std) == list:
                current_wstd = weight_std[l]**2 / n_neurons[l + 1]
                current_bias = bias_std[l]
            else:
                current_wstd = weight_std
                current_bias = bias_std
            self.weights.append(
                current_wstd * np.random.randn(self.n_neurons[l], self.n_neurons[l + 1])
            )
            self.biases.append(current_bias * np.random.randn(self.n_neurons[l + 1]))

    def forward(self, x):
        per_layer_state = []
        for i in range(len(self.n_neurons) - 2):
            biases = np.ones([len(x.T), len(self.weights[i].T)]) * self.biases[i]
            x = self.weights[i].T @ x + biases.T
            per_layer_state.append(x)            
            x = np.tanh(x)
        x = self.weights[-1].T @ x
        per_layer_state.append(x)
        return x, per_layer_state

    def inverse(self, y):
        y = np.linalg.pinv(self.weights[-1].T) @ y
        for i in range(2, len(self.n_neurons)):
            biases = np.ones([len(y.T), len(self.weights[-i].T)]) * self.biases[-i]
            y = np.linalg.pinv(self.weights[-i].T) @ (np.arctanh(y) - biases.T)
        return y


class Inverse_Autoencoder(nn.Module):
    def __init__(self, in_dim, n_neurons, out_dim, std_weights, n_layers=1):
        super(Inverse_Autoencoder, self).__init__()
        self.in_dim = in_dim
        self.n_neurons = n_neurons
        self.std_weights = std_weights
        self.n_layers = n_layers
        self.out_dim = out_dim

        self.layers = nn.ModuleList()
        self.inp_layer = nn.Linear(in_dim, n_neurons, bias=True)
        self.act_fun = nn.Tanh()
        for i in range(n_layers):
            out_layer = nn.Linear(n_neurons, n_neurons, bias=True)
            torch.nn.init.normal_(
                out_layer.weight, mean=0.0, std=std_weights / np.sqrt(self.n_neurons)
            )
            self.layers.append(out_layer.double())
        self.out_layer = nn.Linear(n_neurons, out_dim)

    def forward(self, x):
        out = self.act_fun(self.inp_layer(x.double().T))
        for layer in self.layers:
            out = self.act_fun(layer(out))
        out = self.out_layer(out)
        return out

    def train(self, train_dat, optimizer, epochs=100, batch_sz=20):
        running_loss = 0.0
        loss_curve = []
        inputs = torch.tensor(train_dat[0])
        inputs.requires_grad = True
        targets = torch.tensor(train_dat[1]).double()
        criterion = nn.MSELoss()
        for epoch in range(epochs):  # loop over the dataset multiple times
            batch_ind = torch.arange(
                0, len(train_dat[1].T), batch_sz
            )  # torch.tensor(np.random.choice(np.arange(0,len(train_dat[1].T)),batch_sz,replace=False))#
            for i in batch_ind:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs[i : i + batch_sz].T)
                loss = criterion(outputs, targets[:, i : i + batch_sz].T)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss = loss.item()
                loss_curve.append(running_loss)
            if epoch % 50 == 0 or epoch == 0:
                print(str(running_loss) + " MSE loss")
        return loss_curve
