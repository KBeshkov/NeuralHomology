from Algorithms import *


def geodesic(X, r=-1, r_step=0.1, eps=0.2, count=1):
    r_step_ = r_step
    Xn = np.copy(X)
    if r > 0:
        N = len(Xn)
        d = pairwise_distances(Xn)
        d_geod = (10**10) * np.ones([N, N])
        neighbors = []
        for i in range(N):
            neighbors = np.where(d[i, :] < r)[0]
            d_geod[i, neighbors] = d[i, neighbors]
        d_geod = shortest_path(d_geod)
        if np.sum(d_geod >= 10**10) > 0:
            count += 1
            dn = geodesic(Xn, r=r + eps, eps=eps, count=count)
            return dn
        else:
            #            print('finished in ' + str(count) + ' recursions')
            return d_geod
    else:
        N = len(Xn)
        d = pairwise_distances(Xn)
        hom = tda(d, distance_matrix=True, maxdim=0)["dgms"][0]
        r = hom[-2, 1] + eps * hom[-2, 1]
        d_geod = (10**10) * np.ones([N, N])
        neighbors = []
        for i in range(N):
            neighbors = np.where(d[i, :] < r)[0]
            d_geod[i, neighbors] = d[i, neighbors]
        d_geod = shortest_path(d_geod)
        if np.sum(d_geod >= 10**10) > 0:
            count += 1
            dn = geodesic(Xn, r=r + r_step_ * r, eps=r_step_ * r, count=count)
            return dn
        # else:
        # print('finished in ' + str(count) + ' recursions')

        return d_geod


class geodesic_knn:
    def __init__(self, k=2, adaptive=False):
        self.k = k
        self.adaptive = adaptive
        
    def symmetrize(self, d):
        d_stack = np.stack([d,d.T])
        d_stack[d_stack==0] = np.inf
        d = np.min(d_stack,0)
        return d

    def fit(self, X):
        d = self.symmetrize(kneighbors_graph(X, self.k, mode="distance").toarray())
        d_geod = shortest_path(d)
        if self.adaptive:
            store_dmats = []
            adaptive_k = self.k
            store_dmats.append(d_geod)
            while np.any(np.isinf(d_geod)):
                adaptive_k += 1
                d = self.symmetrize(kneighbors_graph(X, adaptive_k, mode="distance").toarray())
                d[d == 0] = np.inf
                d_geod = shortest_path(d)
                store_dmats.append(d_geod)
            d_geod = store_dmats[0]
            for i in range(len(store_dmats)):
                d_current = store_dmats[i]
                d_geod[
                    np.logical_and(~np.isinf(d_current), np.isinf(d_geod))
                ] = d_current[np.logical_and(~np.isinf(d_current), np.isinf(d_geod))]
        return d_geod


def implicit_metric_circle(X,dt=0.05):
    d = np.zeros([len(X),len(X)])
    for i in range(len(X)):
        X_shifted = np.concatenate([X[i:],X[:i]])
        count = 0
        for j in range(i+1,len(X)):
            if count<len(X)/2:
                d[i,j] = dt*np.linalg.norm(X_shifted[0:count]-X_shifted[1:count+1])
            else:
                d[i,j] = dt*np.linalg.norm(X_shifted[count+1:-1]-X_shifted[count:-2])
            count += 1
                    
    return d+d.T

def extract_cycles(dmat, cocycle, threshold, all_generators=False):
    S = []
    strt = cocycle[0]
    fin = cocycle[1]
    adj_mat = np.copy(dmat)
    adj_mat[dmat >= threshold] = 0
    adj_mat[dmat <= threshold] = 1
    adj_mat[strt, fin] = 0
    adj_mat[fin, strt] = 0
    a = shortest_path(
        adj_mat,
        directed=False,
        return_predecessors=True,
        unweighted=True,
        indices=[strt, fin],
    )[1]
    c = a[0, fin]
    S.append(fin)
    S.append(c)
    while c != strt:
        S.append(a[0, c])
        c = a[0, c]
    if all_generators == True:
        S.append(np.unique(np.where(dmat[S, :] <= threshold)[1]))
        return np.unique(S[-1])
    else:
        return S


def EM_dist(X, Y, L1=0, L2=0, D=0, norm="euclidean"):
    """Earth mover's distance as a linear assignment problem:

    Input:
        X: an array of inputs (eg spikes)
        Y: an array of inputs
        L1 = a list of cells that are being used in X
        L2 = a list of cells that are being used in Y
        D: a precomputed disance matrix

    Output: the distance between clusters
    """
    if norm == "euclidean":
        d = cdist(X, Y)
        assignment = linear_sum_assignment(d)
        em_dist = d[assignment].sum()
        return em_dist
    elif norm == "spiking":
        L = np.ix_(L1, L2)
        d = D[L]
        em_dist = np.mean(d[linear_sum_assignment(d)])
        return em_dist


def pairwise_EM_dist(X):
    D = np.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            if i > j:
                D[i, j] = EM_dist(X[i], X[j])
    return D + D.T


def order_complex(D):
    N = len(D[:, 0])
    ord_mat = np.triu(D)
    np.fill_diagonal(ord_mat, 0)
    Ord = rankdata(ord_mat.flatten(), method="dense").reshape(np.shape(D))
    #    inv_ranks = np.sum(Ord==1)
    Ord = np.triu(Ord) + np.triu(Ord).T
    Ord = Ord  # - inv_ranks
    np.fill_diagonal(Ord, 0)
    return Ord / np.max(Ord)


def rips_plot(
    pcloud, radius, graph=False, dmat=None, polygons=False, shading=True, kn=0
):
    """Draws circles around the points of a point cloud, first dimension contains the number of points"""

    plt.plot(pcloud[:, 0], pcloud[:, 1], "b.")
    fig = plt.gcf()
    ax = fig.gca()
    for i in range(len(pcloud)):
        if shading == True:
            circle = plt.Circle(
                (pcloud[i, 0], pcloud[i, 1]), radius, color="r", alpha=0.025
            )
            ax.add_artist(circle)
        if graph == True:
            for j in range(len(pcloud)):
                if dmat[i, j] <= radius:
                    if i < j:
                        ax.plot(
                            [pcloud[i, 0], pcloud[j, 0]],
                            [pcloud[i, 1], pcloud[j, 1]],
                            "k",
                            alpha=0.25,
                            markersize=12,
                        )
            if polygons == True:
                for k in range(len(pcloud)):
                    if (
                        dmat[i, j] <= radius
                        and dmat[i, k] <= radius
                        and dmat[j, k] <= radius
                    ):
                        polygon = Polygon(pcloud[[i, j, k], :])
                        p = PatchCollection([polygon], alpha=0.5)
                        p.set_array(np.array([5, 50, 100]))
                        ax.add_collection(p)
        if type(shading) == float:
            neighbors_i = np.argsort(dmat[i])[1 : 1 + kn]
            for l in neighbors_i:
                v = pcloud[l] - pcloud[i]
                if np.all(pcloud[i] + v == pcloud[l]):
                    u = pcloud[i] + radius * v
                else:
                    u = pcloud[i] + radius * v
                ax.fill_between(
                    np.array([pcloud[i, 0], u[0]]),
                    np.array([pcloud[i, 1], u[1]]) + shading,
                    np.array([pcloud[i, 1], u[1]]) - shading,
                    color="green",
                    alpha=0.05,
                )
    return fig, ax


def direct_graph_plot(pcloud, radius, graph=False, dmat=None, polygons=False):
    fig = plt.gcf()
    ax = fig.gca()
    for i in range(len(pcloud)):
        if graph == True:
            for j in range(len(pcloud)):
                if dmat[i, j] < radius:
                    plt.plot(
                        [pcloud[i, 0], pcloud[j, 0]],
                        [pcloud[i, 1], pcloud[j, 1]],
                        "k",
                        alpha=0.05,
                    )
                    plt.arrow(
                        pcloud[i, 0],
                        pcloud[i, 1],
                        0.7 * (pcloud[j, 0] - pcloud[i, 0]),
                        0.7 * (pcloud[j, 1] - pcloud[i, 1]),
                        color="b",
                        head_width=0.1,
                        lw=0.01,
                    )
                if polygons == True:
                    for k in range(len(pcloud)):
                        if (
                            dmat[i, j] < radius
                            and dmat[i, k] < radius
                            and dmat[j, k] < radius
                        ):
                            polygon = Polygon(pcloud[[i, j, k], :])
                            p = PatchCollection([polygon], alpha=0.5)
                            p.set_array(np.array([5, 50, 100]))
                            ax.add_collection(p)


"""Finds the most persistent component of a given dimension > 0"""


def max_pers(pd, dim=1):
    if len(pd[dim]) > 0:
        pers = pd[dim][:, 1] - pd[dim][:, 0]
        max_persistence = np.max(pers)
        return max_persistence
    else:
        return 0


def recurrence_plot(X, thresh=0.1, func=lambda x: pairwise_distances(x)):
    D = func(X)
    D[D >= thresh] = 0
    D[np.logical_and(D < thresh, D != 0)] = 1
    np.fill_diagonal(D, 1)
    return D


def gen_weight_mat(N, rank, g=1, svd=True, eigenvals=[], zero=False):
    P = np.zeros([N, N])
    if svd == False:
        for r in range(rank):
            m = np.random.randn(N)
            n = np.random.randn(N)
            P = P + np.outer(m, n) / (1 + r)
        gX = ((g**2) / N) * np.random.randn(N, N)
        J = gX + P
        np.fill_diagonal(J, 0)
        return J
    elif svd == "eigdecomp":
        U = ortho_group.rvs(N)
        if eigenvals != []:
            D = np.diag(np.concatenate([eigenvals, np.zeros(N - rank)]))
        else:
            D = np.diag(
                np.concatenate([2 * np.random.rand(rank) - 1, np.zeros(N - rank)])
            )
        V = np.linalg.inv(U)
        P = 5 * rank * (np.matmul(U, np.matmul(D, V))) / N * rank
        gX = ((g**2) / N) * np.random.randn(N, N)
        J = gX + P
        np.fill_diagonal(J, 0)
        return J, [U, D, V]
    elif svd == "qr_decomp":
        # A = 0.01*np.random.randn(N,N)
        A = ortho_group.rvs(N)
        # np.fill_diagonal(A,0)
        U = np.linalg.qr(A)
        for i in range(N):
            if i < rank:
                U[1][i, i] = eigenvals[i]
            else:
                U[1][i, i] = 0
        P = U[0] @ U[1] @ U[0].T
        gX = ((g**2) / N) * np.random.randn(N, N)
        J = gX + P
        if zero:
            np.fill_diagonal(J, 0)
        return J, U
    else:
        U = ortho_group.rvs(N)
        if eigenvals != []:
            D = np.diag(np.concatenate([eigenvals, np.zeros(N - rank)]))
        else:
            D = np.diag(
                np.concatenate([np.sort(np.random.randn(rank)), np.zeros(N - rank)])
            )
        V = ortho_group.rvs(N)
        P = rank * (np.matmul(U, np.matmul(D, V.T))) / N * rank
        gX = ((g**2) / N) * np.random.randn(N, N)
        J = gX + P
        # np.fill_diagonal(J,0)
        return J, [U, D, V]


def low_rank_rnn(
    N, T, I=0, P=0, rank=1, mu=0.05, init_x=[0], g=1, svd=False, act_fun=np.tanh
):
    if P is None:
        P = gen_weight_mat(N, rank, g, svd)[0]
    x = np.zeros([N, T])
    if len(init_x) == 0:
        x[:, 0] = np.random.rand(N)
    else:
        x[:, 0] = init_x
    for t in range(T - 1):
        dx = -x[:, t] + np.dot(P, act_fun(x[:, t])) + I[:, t]
        x[:, t + 1] = x[:, t] + mu * dx
    return x


def annotate_imshow(D, round_val=2, txt_size=6):
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.imshow(D, aspect="auto")
    for (j, i), label in np.ndenumerate(D):
        if label != 0:
            ax.text(
                i,
                j,
                round(label, round_val),
                ha="center",
                va="center",
                fontsize=txt_size,
            )


def transient_nets(N, T, W, I, mu=0.05):
    X = np.zeros([N, T])
    for t in range(T - 1):
        dX = 10 * (-X[:, t] + np.dot(W, X[:, t]) + I[:, t])
        X[:, t + 1] = X[:, t] + mu * dX
    return X


def plot_circl_eig(r):
    rd = np.abs(r)
    r_perc = np.percentile(rd, 95)
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(1 * np.cos(theta), 1 * np.sin(theta), "k")
    plt.plot(r_perc * np.cos(theta), r_perc * np.sin(theta), color="g")
    return r_perc


def get_boundary(x):
    hull = ConvexHull(x)
    points = hull.points
    vertices = hull.vertices
    return np.squeeze(np.array([[points[vertices, 0], points[vertices, 1]]]))


"""Takes in a list of persistence diagrams and calculates the bottleneck distances between them"""


def bottleneck_dmat(pdiag1, pdiag2, dim=1):
    D = np.zeros([len(pdiag1), len(pdiag2)])
    for i in range(len(pdiag1)):
        for j in range(len(pdiag2)):
            if i > j:
                D[i, j] = persim.bottleneck(pdiag1[i][dim], pdiag2[j][dim])
    D = D + D.T
    return D


def bottleneck_time(pdiags, dim=1, features=1, plot=False):
    pers_vals = []
    matchings = []
    for t in range(len(pdiags) - 1):
        feat1, feat2 = (
            np.argsort(pdiags[t][dim][:, 1] - pdiags[t][dim][:, 0])[-features:],
            np.argsort(pdiags[t + 1][dim][:, 1] - pdiags[t + 1][dim][:, 0])[-features:],
        )
        pers_vals.append([pdiags[t][dim][feat1, :], pdiags[t + 1][dim][feat2, :]])
        _, matching = persim.bottleneck(
            np.array(pers_vals[-1][0]), np.array(pers_vals[-1][1]), matching=True
        )
        matchings.append(matching)

    if plot == True:
        for t in range(len(pdiags) - 1):
            persim.bottleneck_matching(pers_vals[t][0], pers_vals[t][1], matchings[t])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot([0, 1], [0, 1], "k")
    return pers_vals, matchings


def visualize_functions(points, functs):
    for i in functs:
        plt.figure(dpi=200)
        plt.plot(i(points))


def plot_grid(layer_list, fft=False):
    if fft:
        for i in range(len(layer_list)):
            plt.figure(dpi=200)
            n_figs = int(np.sqrt(len(layer_list[i])))
            N = int(np.sqrt(len(layer_list[i].T)))
            for j in range(n_figs**2):
                rshaped_img = np.reshape(layer_list[i][j], [N, N])
                rshaped_img = np.fft.ifftshift(rshaped_img)
                rshaped_img = np.fft.fft2(rshaped_img)
                rshaped_img = np.fft.fftshift(rshaped_img)
                plt.subplot(n_figs, n_figs, j + 1)
                # plt.contourf(x,y,np.abs(rshaped_img),vmax=100000,levels=10)
                plt.imshow(np.abs(rshaped_img), vmax=1)
                plt.axis("off")
            plt.tight_layout()
        return
    for i in range(len(layer_list)):
        plt.figure(dpi=200)
        n_figs = int(np.sqrt(len(layer_list[i])))
        N = int(np.sqrt(len(layer_list[i].T)))
        for j in range(n_figs**2):
            rshaped_img = np.reshape(layer_list[i][j], [N, N])
            plt.subplot(n_figs, n_figs, j + 1)
            plt.imshow(rshaped_img, vmin=-1, vmax=1)
            plt.axis("off")
        plt.tight_layout()


def compute_kernel(model, data):
    X = np.zeros([len(data), len(data)])
    model_out = model(data.T)[1]
    for i in range(len(data)):
        for j in range(len(data)):
            if i >= j:
                X[i, j] = model_out[i] @ model_out[j].T
    X = X + X.T
    return X / np.linalg.norm(X)


def eval_expressivity(dat, iters=10, deg=20, plot=False, compositional=0):
    scores = []
    data = np.copy(dat)
    for i in range(iters):
        if compositional > 0:
            indcs = np.random.choice(
                np.arange(0, len(dat.T)), size=compositional, replace=False
            )
            data = dat[:, indcs]
        train_id = np.sort(
            np.random.choice(
                np.arange(0, len(dat)), size=int(0.7 * len(dat)), replace=False
            )
        )
        test_id = np.setdiff1d(np.arange(0, len(dat)), train_id)
        data_train = data[train_id]
        data_test = data[test_id]
        rand_poly = npoly.chebyshev.chebval(
            np.linspace(-1, 1, len(data)), 0.1 * np.random.randn(deg)
        )
        poly_fit = LinearRegression().fit(data_train, rand_poly[train_id])
        scores.append(poly_fit.score(data_test, rand_poly[test_id]))
    if plot:
        plt.plot(rand_poly)
        plt.plot(poly_fit.predict(data))
    return scores


def persistence_landscape_distance(pdiag1, pdiag2, d=1, plot=False):
    pers_lands1 = persim.landscapes.PersLandscapeApprox(dgms=pdiag1, hom_deg=d)
    pers_lands2 = persim.landscapes.PersLandscapeApprox(dgms=pdiag2, hom_deg=d)
    [pland1_snapped, pland2_snapped] = persim.landscapes.snap_pl(
        [pers_lands1, pers_lands2]
    )
    true_diff_pl = pland1_snapped - pland2_snapped
    significance = true_diff_pl.sup_norm()
    print(f"The threshold for significance is {significance}.")
    if plot:
        plt.figure()
        persim.landscapes.plot_landscape_simple(true_diff_pl)
        plt.legend([])
    return pers_lands1, pers_lands2, true_diff_pl


def binary_hierarchical_labeling(nclasses, npoints):
    labels = []
    for l in range(len(nclasses)):
        labels.append(np.zeros(npoints))
        for k in range(nclasses[l]):
            interval = int(npoints / nclasses[l])
            labels[l][k * interval : (k + 1) * interval] = k
    return labels


def hierarchical_labeling_from_data(labels, pairings):
    lbls_new = []
    for l in labels:
        lbls_new.append(np.where(l == pairings)[0][0])
    return np.asarray(lbls_new)


def generate_gratings(res, angle, sfq):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    k = np.array([1, 0])
    k_r = sfq * R @ k
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x)
    grating = np.cos(k_r[0] * Y + k_r[1] * X)
    return grating / np.linalg.norm(grating)


def generate_gabors(res, angle, sfq, width, aspectr=1):
    X, Y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    xp = X * np.cos(angle) + Y * np.sin(angle)
    yp = -X * np.sin(angle) + Y * np.cos(angle)
    g_filt = np.exp(-(xp**2 + (yp * aspectr) ** 2) / 2 * width**2) * np.cos(
        2 * np.pi * sfq * (xp)
    )
    return g_filt / np.linalg.norm(g_filt)


def orientation_selectivity_index(X, bins):
    N = len(X.T)
    # X = X/np.linalg.norm(X,axis=0)
    pref_tuning = np.argmax(X, 0)
    orth_tuning = (pref_tuning + int(len(bins) / 4)) % len(bins)
    #orth_tuning2 = (pref_tuning - int(len(bins) / 4)) % len(bins)
    R_pref = np.zeros(N)
    R_orth = np.zeros(N)
    for i in range(N):
        R_pref[i] = X[pref_tuning[i], i]
        R_orth[i] = X[orth_tuning[i], i] #max(X[orth_tuning1[i], i], X[orth_tuning2[i],i])
    return (R_pref - R_orth) / (R_pref + R_orth)


def direction_selectivity_index(X, bins):
    N = len(X.T)
    pref_tuning = np.argmax(X, 0)
    opp_tuning = (pref_tuning + int(len(bins) / 2)) % len(bins)
    R_pref = np.zeros(N)
    R_opp = np.zeros(N)
    for i in range(N):
        R_pref[i] = X[pref_tuning[i], i]
        R_opp[i] = X[opp_tuning[i], i]
    return (R_pref - R_opp) / (R_pref + R_opp)


def angle_selectivity_index(X, angl_int):
    N = len(X.T)
    bins = len(X) - 1
    # X = X/np.linalg.norm(X,axis=0)
    pref_tunning = np.argmax(X, 0)
    orth_tunning = (pref_tunning + int(bins / angl_int)) % bins
    R_pref = np.zeros(N)
    R_orth = np.zeros(N)
    for i in range(N):
        R_pref[i] = X[pref_tunning[i], i]
        R_orth[i] = X[orth_tunning[i], i]
    return (R_pref - R_orth) / (R_pref + R_orth)

def fit_von_mises(x,y):
    von_mises = lambda x, mu, kappa, m: m*np.exp(kappa*np.cos(x-mu))
    von_mises_mix = lambda x, mu1, mu2, kappa1, kappa2, m1, m2, b: von_mises(x ,mu1, kappa1, m1)+von_mises(x, mu2, kappa2, m2)+b
    param_bounds = np.array([[-np.pi,np.pi],[-np.pi,np.pi],[0,np.inf],[0,np.inf],[0,np.inf],[0,np.inf],[0,np.inf]])
    init_params = [0,0,1,1,1,1,np.mean(y)]
    try:
        params = curve_fit(von_mises_mix, x, y, p0=init_params, bounds=param_bounds.T, method='trf', maxfev=10000)[0]
    except:
        params = [0,0,1,1,0,0,np.mean(y)]
    f_x = von_mises_mix(x,*params)
    f_x1 = von_mises(x, params[0], params[2], params[4])+params[6]
    f_x2 = von_mises(x, params[1], params[3], params[5])+params[6]
    return f_x, f_x1, f_x2, params

def reassign_kmeans_labels(labels, clust_centers):
    nlabels = np.copy(labels)
    new_order = [
        np.argmax(np.sum(clust_centers, 0)),
        np.argmax(clust_centers[:, 0] - clust_centers[:, 1]),
        np.argmax(clust_centers[:, 1] - clust_centers[:, 0]),
        np.argmin(np.sum(clust_centers, 0)),
    ]
    nlabels[labels == new_order[0]] = 0
    nlabels[labels == new_order[1]] = 1
    nlabels[labels == new_order[2]] = 2
    nlabels[labels == new_order[3]] = 3
    return nlabels


def vonMisses_Fisher_distribution(x, mean, kappa):
    D = len(x)
    I = iv((D / 2) - 1, kappa)
    C = kappa ** ((D / 2) - 1) / ((2 * np.pi) ** (D / 2) * I)
    return C * np.exp(kappa * mean @ x)


def Bingham_distribution(x, covariance):
    D = len(x)
    Z, M = np.linalg.eig(covariance)
    Z = np.diag(Z)
    F = hyp1f1(1 / 2, D / 2, Z)
    return np.linalg.inv(F) * np.diag(np.exp(x.T @ M @ Z @ M.T @ x))


def get_peaks(X, horizontal_distance, height_range=[0.6, 1]):
    X_norm = X / np.max(X, 0)
    peaks_numbers = []
    for i in range(len(X_norm.T)):
        temp_peaks = find_peaks(
            X_norm[:, i], distance=horizontal_distance, height=height_range
        )[0]
        peaks_numbers.append(len(temp_peaks))
    return peaks_numbers


def shift_triag_diag(x):
    x = x / np.linalg.norm(x, 2, 1)
    return np.vstack(
        [
            np.hstack([x[i:, i], np.nan * np.zeros([len(x[:i, i])])])
            for i in range(len(x))
        ]
    )


def angle_invar_clustering(x, k=4):
    x = x / np.linalg.norm(x, 2, 0)
    x_shifted = np.zeros(np.shape(x))
    for i in range(len(x)):
        max_resp = np.argmax(x[i])
        x_shifted[i, :] = np.concatenate([x[i, max_resp:], x[i, :max_resp]])
    if type(k) == list:
        silhouettes = []
        for i in k:
            cluster_i = KMeans(n_clusters=i).fit(x_shifted)
            clust_i_labels = cluster_i.labels_
            silhouettes.append(silhouette_score(x_shifted, clust_i_labels))
        print("The best choice of k is " + str(k[np.argmax(silhouettes)]))
        clusters = KMeans(n_clusters=k[np.argmax(silhouettes)]).fit(x_shifted)
    else:
        clusters = KMeans(n_clusters=k).fit(x_shifted)
    return x_shifted, clusters.labels_


def si_clustering(x, n_angles=[2, 3, 4], k=4):
    angle_features = np.zeros([len(x.T), len(n_angles)])
    n_bins = len(x)
    for c, i in enumerate(n_angles):
        angle_features[:, c] = angle_selectivity_index(x, i)
    clusters = KMeans(n_clusters=k).fit(angle_features)
    return clusters.labels_


def extract_cell_RFs(dat):
    X = dat["stat"]
    centers = np.zeros([len(X), 2])
    radii = np.zeros(len(X))
    for i, cell in enumerate(X):
        radii[i] = cell["radius"]
        centers[i] = cell["med"]
    return centers, radii


def linear_interpolator(x, y, points, rand=True):
    if rand:
        t = np.random.rand(points, 1) @ np.ones([1, len(x)])
    else:
        t = np.linspace(0, 1, points)[:, None] @ np.ones([1, len(x)])
    return t * x + (1 - t) * y


def interpolation_distance(f_t):
    d = []
    for i in range(len(f_t) - 1):
        d.append(np.linalg.norm(f_t[i] - f_t[i + 1], 2))
    return sum(d)


def interpolation_distance_matrix(X, func=lambda x: x, samples=100):
    d = np.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            if i > j:
                f_t = func(linear_interpolator(X[i], X[j], samples, rand=False))
                d[i, j] = interpolation_distance(f_t)
    return d + d.T


class Random_Projector:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        randM = np.random.randn(len(X.T), len(X.T))
        Proj = np.linalg.svd(randM)
        D = Proj[1]
        D[self.n_components :] = 0
        Proj = Proj[0][: self.n_components, :] @ np.diag(D) @ Proj[2]
        return X @ Proj


def point_cloud_to_image(X, grid_res=8):
    '''
    Generates a 2 dimensional projection of a 3 dimensional point cloud

    Parameters
    ----------
    X : numpy array
        3 dimensional point cloud.
    grid_res : numpy array
        Specifies the size of the 2 dimensional grid. The default is 8.

    Returns
    -------
    X_new : numpy array
        A resXres projection of the point cloud.

    '''
    x_min, x_max, y_min, y_max = (
        np.min(X[:, 0]),
        np.max(X[:, 0]),
        np.min(X[:, 1]),
        np.max(X[:, 1]),
    )
    grid_x, grid_y = np.linspace(x_min, x_max, grid_res), np.linspace(
        y_min, y_max, grid_res
    )
    grid = np.array([grid_x, grid_y])
    X_new = np.zeros([grid_res, grid_res])

    for i in range(grid_res - 1):
        for j in range(grid_res - 1):
            box = np.array([[grid_x[i], grid_y[j]], [grid_x[i + 1], grid_y[j + 1]]])
            temp_points = np.all(
                (box[0] <= X[:, [0, 1]]) & (X[:, [0, 1]] <= box[1]), axis=1) 
            try:
                X_new[i, j] = np.max(X[temp_points, 2])
            except:
                pass

    return X_new


def Perlin_noise_grid(octaves=[1, 10, 100], octave_weights=[1, 0.5, 0.25], res=10):
    pic = np.zeros([res, res])
    noises = []
    for o in range(len(octaves)):
        noises.append(PerlinNoise(octaves=octaves[o]))
    for i in range(res):
        for j in range(res):
            noise_val = octave_weights[0] * noises[0]([i / res, j / res])
            for n in range(1, len(noises)):
                noise_val += octave_weights[n] * noises[n]([i / res, j / res])
            pic[i, j] = noise_val
    return pic


def Perlin_noise_gen(dim, octaves=[1, 5, 10], octave_weights=[1, 0.5, 0.25], res=10):
    pic = np.zeros([dim, res, res, res])
    for d in range(dim):
        noises = []
        for o in range(len(octaves)):
            noises.append(PerlinNoise(octaves=octaves[o]))
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    noise_val = octave_weights[0] * noises[0](
                        [i / res, j / res, k / res]
                    )
                    for n in range(1, len(noises)):
                        noise_val += octave_weights[n] * noises[n](
                            [i / res, j / res, k / res]
                        )
                    pic[d, i, j, k] = noise_val
    pic = np.reshape(pic, (dim, res * res * res))
    return pic
