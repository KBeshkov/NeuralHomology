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
    """Class for computing geodesic distances based on a k-nearest neighbors graph"""
    def __init__(self, k=2, adaptive=False):
        """
        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors. The default is 2.
        adaptive : bool, optional
            Whether to increase k until the graph is becomes fully connected.
            The default is False.
        """
        self.k = k
        self.adaptive = adaptive
        
    def symmetrize(self, d):
        """
        Given that k-nearest neighbor graphs need not be symmetric. This function
        makes the distance-weighted graph undirected, which is necessary for 
        the notion of distance to make sense.

        Parameters
        ----------
        d : numpy array
            A non-symmetric distance matrix.

        Returns
        -------
        d : numpy array
            A symmetric distance matrix.

        """
        d_stack = np.stack([d,d.T])
        d_stack[d_stack==0] = np.inf
        d = np.min(d_stack,0)
        return d

    def fit(self, X):
        """
        Fits a k-nearest neighbor graph to a point cloud.

        Parameters
        ----------
        X : numpy array
            A point cloud.

        Returns
        -------
        d_geod : numpy array
            A distance matrix of geodesic distances.

        """
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


def max_pers(pd, dim=1):
    """Finds the most persistent component of a given dimension > 0"""
    if len(pd[dim]) > 0:
        pers = pd[dim][:, 1] - pd[dim][:, 0]
        max_persistence = np.max(pers)
        return max_persistence
    else:
        return 0


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


def generate_gratings(res, angle, sfq):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    k = np.array([1, 0])
    k_r = sfq * R @ k
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x)
    grating = np.cos(k_r[0] * Y + k_r[1] * X + np.pi/2)
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
    R_pref = np.zeros(N)
    R_orth = np.zeros(N)
    for i in range(N):
        R_pref[i] = X[pref_tuning[i], i]
        R_orth[i] = X[orth_tuning[i], i] 
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
