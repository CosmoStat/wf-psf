import numpy as np


class GraphBuilder(object):
    r"""GraphBuilder class.

    This class computes the necessary quantities for RCA's graph constraint.

    Parameters
    ----------
    obs_data: numpy.ndarray
        Observed data.
    obs_pos: numpy.ndarray
        Corresponding positions.
    obs_weights: numpy.ndarray
        Corresponding per-pixel weights.
    n_comp: int
        Number of RCA components.
    n_eigenvects: int
        Maximum number of eigenvectors to consider per :math:`(e,a)` couple.
        Default is ``None``;
        if not provided, *all* eigenvectors will be considered,
        which can lead to a poor selection of graphs, especially when data
        is undersampled.
        Ignored if ``VT`` and ``alpha`` are provided.
    n_iter: int
        How many alternations should there be when optimizing over
        :math:`e` and :math:`a`. Default is 3.
    ea_gridsize: int
        How fine should the logscale grid of :math:`(e,a)` values be.
        Default is 10.
    distances: numpy.ndarray
        Pairwise distances for all positions. Default is ``None``;
        if not provided, will be computed from given positions.
    auto_run: bool
        Whether to immediately build the graph quantities.
        Default is ``True``.
    """

    def __init__(
        self,
        obs_data,
        obs_pos,
        obs_weights,
        n_comp,
        n_eigenvects=None,
        n_iter=3,
        ea_gridsize=10,
        distances=None,
        auto_run=True,
        verbose=2,
    ):
        r"""Initialize class attributes."""
        self.obs_data = obs_data
        shap = self.obs_data.shape
        self.obs_pos = obs_pos
        self.obs_weights = obs_weights
        # change to same format as that we will use for
        # residual matrix R later on
        self.obs_weights = np.transpose(
            self.obs_weights.reshape((shap[0] * shap[1], shap[2]))
        )
        self.n_comp = n_comp
        if n_eigenvects is None:
            self.n_eigenvects = self.obs_data.shape[2]
        else:
            self.n_eigenvects = n_eigenvects
        self.n_iter = n_iter
        self.ea_gridsize = ea_gridsize
        if verbose > 1:
            self.verbose = True
        else:
            self.verbose = False

        if distances is None:
            self.distances = pairwise_distances(self.obs_pos)
        else:
            self.distances = distances
        if auto_run:
            self._build_graphs()

    def _build_graphs(self):
        r"""Compute graph-constraint related values.

        Notes
        -----
        See RCA paper (Ngole et al.) sections 5.2 and (especially) 5.5.3.
        """
        shap = self.obs_data.shape
        e_max = self.pick_emax()
        if self.verbose:
            print(" > power max = ", e_max)

        # [TL] Modif min from 0.01 to 0.001
        a_range = np.geomspace(0.001, 1.99, self.ea_gridsize)
        e_range = np.geomspace(0.01, e_max, self.ea_gridsize)

        # initialize R matrix with observations
        R = np.copy(np.transpose(self.obs_data.reshape((shap[0] * shap[1], shap[2]))))

        self.sel_a = []
        self.sel_e = []
        idx = []
        list_eigenvects = []
        for _ in range(self.n_comp):
            e, a, j, best_VT = self.select_params(R, e_range, a_range)
            self.sel_e += [e]
            self.sel_a += [a]
            idx += [j]
            list_eigenvects += [best_VT]
            vect = best_VT[j].reshape(1, -1)
            R -= vect.T.dot(vect.dot(R))
            if self.verbose:
                print(
                    " > selected e: {}\tselected a:".format(e)
                    + "{}\t chosen index: {}/{}".format(a, j, self.n_eigenvects)
                )
        self.VT = np.vstack((eigenvect for eigenvect in list_eigenvects))
        self.alpha = np.zeros((self.n_comp, self.VT.shape[0]))
        for i in range(self.n_comp):
            self.alpha[i, i * self.n_eigenvects + idx[i]] = 1

    def pick_emax(self, epsilon=1e-15):
        r"""Pick maximum value for ``e`` parameter.

        From now, we fix the maximum :math:`e` to 1 and ignore the old
        procedure that was giving values that were too big.

        Old procedure:
        Select maximum value of :math:`e` for the greedy search over set of
        :math:`(e,a)` couples, so that the graph is still fully connected.
        """
        # nodiag = np.copy(self.distances)
        # nodiag[nodiag==0] = 1e20
        # dist_ratios = np.min(nodiag,axis=1) / np.max(self.distances, axis=1)
        # r_med = np.min(dist_ratios**2)
        # return np.log(epsilon)/np.log(r_med)

        return 1.0

    def select_params(self, R, e_range, a_range):
        r"""Select best graph parameters.

        Select :math:`(e,a)` parameters and best eigenvector
        for current :math:`R_i` matrix.

        Parameters
        ----------
        R: numpy.ndarray
            Current :math:`R_i` matrix
            (as defined in RCA paper (Ngole et al.), sect. 5.5.3.)
        e_range: numpy.ndarray
            List of :math:`e` values to be tested.
        a_range: numpy.ndarray
            List of :math:`a` values to be tested.
        """
        current_a = 0.5
        for i in range(self.n_iter):
            # optimize over e
            Peas = np.array([gen_Pea(self.distances, e, current_a) for e in e_range])
            all_eigenvects = np.array([self.gen_eigenvects(Pea) for Pea in Peas])
            ea_idx, eigen_idx, _ = select_vstar(all_eigenvects, R, self.obs_weights)
            current_e = e_range[ea_idx]

            # optimize over a
            Peas = np.array([gen_Pea(self.distances, current_e, a) for a in a_range])
            all_eigenvects = np.array([self.gen_eigenvects(Pea) for Pea in Peas])
            ea_idx, eigen_idx, best_VT = select_vstar(
                all_eigenvects, R, self.obs_weights
            )
            current_a = a_range[ea_idx]

        return current_e, current_a, eigen_idx, best_VT

    def gen_eigenvects(self, mat):
        r"""Compute input matrix's eigenvectors.

        Keep only the ``n_eigenvects`` associated
        with the smallest eigenvalues.
        """
        U, s, vT = np.linalg.svd(mat, full_matrices=True)
        vT = vT[-self.n_eigenvects :]
        return vT


def select_vstar(eigenvects, R, weights):
    r"""Pick best eigenvector from a set of :math:`(e,a)`.

    i.e., solve (35) from RCA paper (Ngole et al.).

    Parameters
    ----------
    eigenvects: numpy.ndarray
        Array of eigenvects to be tested over.
    R: numpy.ndarray
        :math:`R_i` matrix.
    weights: numpy.ndarray
        Entry-wise weights for :math:`R_i`.
    """
    loss = np.sum((weights * R) ** 2)
    for i, Pea_eigenvects in enumerate(eigenvects):
        for j, vect in enumerate(Pea_eigenvects):
            colvect = np.copy(vect).reshape(1, -1)
            current_loss = np.sum(
                (weights * R - colvect.T.dot(colvect.dot(weights * R))) ** 2
            )
            if current_loss < loss:
                loss = current_loss
                eigen_idx = j
                ea_idx = i
                best_VT = np.copy(Pea_eigenvects)

    return ea_idx, eigen_idx, best_VT


def pairwise_distances(obs_pos):
    r"""Compute pairwise distances."""
    ones = np.ones(obs_pos.shape[0])
    out0 = np.outer(obs_pos[:, 0], ones)
    out1 = np.outer(obs_pos[:, 1], ones)
    return np.sqrt((out0 - out0.T) ** 2 + (out1 - out1.T) ** 2)


def gen_Pea(distances, e, a):
    r"""Compute the graph Laplacian for a given set of parameters.

    Parameters
    ----------
    distances: numpy.ndarray
        Array of pairwise distances
    e: float
        Exponent to which the pairwise distances should be raised.
    a: float
        Constant multiplier along Laplacian's diagonal.

    Returns
    -------
    Pea: numpy.ndarray
        Graph laplacian.

    Notes
    -----
    Computes :math:`P_{e,a}` matrix for given ``e``, ``a`` couple.
    See Equations (16-17) in RCA paper (Ngole et al.).
    Watch out with the ``e`` parameter as it plays a vital role in the graph
    definition as it is a parameter of the distance that defines the
    graph's weights.

    """
    Pea = np.copy(distances**e)
    np.fill_diagonal(Pea, 1.0)
    Pea = -1.0 / Pea
    for i in range(Pea.shape[0]):
        Pea[i, i] = a * (np.sum(-1.0 * Pea[i]) - 1.0)
    return Pea
