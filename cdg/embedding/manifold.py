# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Implements embedding on Riemannian manifolds, in particular. those of constant
# curvature.
# Some classes implement the manifold geometry, others the embedding on this
# manifolds.
#
#
# References:
# ---------
# [ijcnn18]
#   Zambon, Daniele, Lorenzo Livi, and Cesare Alippi.
#   Anomaly and Change Detection in Graph Streams through Constant-Curvature
#   Manifold Embeddings.
#   IEEE International Joint Conference on Neural Networks (2018).
#
# [1]
#   Wilson, R. C., Hancock, E. R., Pekalska, E., & Duin, R. P.
#   Spherical and hyperbolic embeddings of data.
#   IEEE transactions on pattern analysis and machine intelligence (2014), 36(11), 2255-2269.
#
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 11/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import cdg.util.logger
import cdg.util.errors
import cdg.util.prototype
import cdg.util.geometry
import cdg.embedding.embedding


def _get_real_eig(A, tol=1e-5):
    """
    Compute the real eigenvalues of a symmetric matrix.

    :param A: (n, n) symmetric np.array
    :param tol: tolerance in considering a complex number as a real
    :return:
        - eigenvalues : sorted in descending order
        - eigenvectors : corresponding to eigenvalues
    """
    # check symmetry
    if not np.allclose(A, A.transpose(), atol=tol):
        raise ValueError('matrix not symmetrix')

    # compute spectrum
    # https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
    # https://stackoverflow.com/a/8765592
    val_complex, vec_complex = np.linalg.eigh(A)
    # cast to real
    val_real = np.real(val_complex)
    vec_real = np.real(vec_complex)
    if not np.allclose(val_real, val_complex, atol=tol) \
            or not np.allclose(vec_real, vec_complex, atol=tol):
        raise cdg.util.errors.CDGImpossible('complex numbers')

    # sort output
    sorted_index = np.argsort(val_real)[::-1]
    eigenvalues = val_real[sorted_index]
    eigenvectors = vec_real[:, sorted_index]

    return eigenvalues, eigenvectors


def _arrange_as_matrix(X):
    X_mat = X.copy()
    if len(X.shape) == 1:
        X_mat = X_mat.reshape(1, X_mat.shape[0])
    return X_mat


class RiemannianManifold(cdg.embedding.embedding.Embedding):
    """
    Defines the structure of a Riemannian manifold.
    The general notation is:
        - x, X: points on the manifold
        - nu, Nu: points on the tangent space in local coordinates
        - v, V: points on the tangent space in global coordinetes, that is with
            respect to the representation of the manifold
    """

    _name = 'GenericRiemannianManifold'
    # dimension of the manifold, i.e., dimension of the tangent space
    manifold_dimension = None

    def __init__(self):
        super().__init__()

    def set_parameters(self, **kwargs):
        self.manifold_dimension = kwargs.pop('d')
        super().set_parameters(**kwargs)

    @classmethod
    def distance(cls, X1, X2=None, **kwargs):
        """
        Geodesic distance between points.
        :param X1: (n1, d)
        :param X2: (n2, d)
        :param kwargs:
        :return: distance matrix (n1, n2)
        """

        # input parsing
        X1_mat = _arrange_as_matrix(X=X1)
        if X2 is None:
            only_upper_triangular = True
            X2_mat = X1_mat.copy()
        else:
            only_upper_triangular = False
            X2_mat = _arrange_as_matrix(X=X2)

        if X1_mat.shape[1] != X2_mat.shape[1]:
            raise cdg.util.errors.CDGError("dimension mismatch")

        # actual computation
        D = cls._distance(X1_mat=X1_mat, X2_mat=X2_mat,
                          only_upper_triangular=only_upper_triangular,
                          **kwargs)

        # check nan
        if np.argwhere(np.isnan(D)).shape[0] > 0:
            D = cls._distance(X1_mat=X1_mat, X2_mat=X2_mat,
                              only_upper_triangular=only_upper_triangular, **kwargs)
            raise ValueError("Some values in the distance matrix are NaN")

        return D

    @classmethod
    def exp_map(cls, x0_mat, Nu_mat):
        """
        Exponential map from tangent space to the manifold
        :param x0_mat: point of tangency
        :param Nu_mat: (n, man_dim) points on the tangent space to be mapped
        :return: X_mat: points on the manifold
        """
        raise cdg.util.errors.CDGAbstractMethod()

    @classmethod
    def log_map(cls, x0_mat, X_mat):
        """
        Logarithm map from manifold to tangent space.
        :param x0_mat: point of tangency
        :param X_mat: (n, emb_dim) points on the manifold to be mapped
        :return: Nu_mat: points on the tangent space
        """
        raise cdg.util.errors.CDGAbstractMethod()


class ConstantCurvarure(RiemannianManifold):
    """
    Subclass of constant curvature manifolds.
    """
    _name = 'GenericConstantCurvarure'
    # curvature
    curvature = None
    # geometry of the hosting (immersion, embedding) space.
    _geo = None

    @property
    def radius(self):
        if self.curvature == 0 or self.curvature is None:
            return None
        else:
            return 1. / np.sqrt(np.abs(self.curvature))

    def set_parameters(self, **kwargs):
        self.curvature = kwargs.pop('curvature', None)
        super().set_parameters(**kwargs)

    @classmethod
    def sample_mean(cls, X, **kwargs):
        # pbar = tqdm(leave=True)
        # pbar.set_description('sample mean')
        # pbar.update()
        X_mat = _arrange_as_matrix(X=X)
        # find argmin_{x\in X} \sum_i \rho(x,X_i)^2
        Dn = cls.distance(X1=X_mat, **kwargs)
        mean_id, _ = cdg.util.prototype.mean(dissimilarity_matrix=Dn, power=2)
        x_new = X_mat[mean_id:mean_id + 1, :].copy()

        # Optimise
        xk = x_new.copy() + 10.  # hack to be neq x_new
        iter_max = 10
        ct = 0

        while ct < iter_max and not np.allclose(xk, x_new):
            ct += 1
            xk = x_new.copy()
            Nu = cls.log_map(x0_mat=xk, X_mat=X_mat)
            mean_log = _arrange_as_matrix(np.mean(Nu, axis=0))
            if np.isnan(mean_log[0]).any():
                raise cdg.util.errors.CDGImpossible()
            x_new = cls.exp_map(x0_mat=xk, Nu_mat=mean_log)

            # check overflow
            if np.linalg.norm(x_new) > 1e4:
                raise ValueError('Risk of overflow')
                # pbar.update()
        # pbar.close()
        return x_new


class EuclideanManifold(ConstantCurvarure):
    """
    Null-curvature manifold.
    Here points are naturally represented, hence points on a d-dimensional manifold
    are represented by d components.
    """
    _name = 'E'
    _geo = cdg.util.geometry.Eu()

    @property
    def curvature(self):
        return 0

    @curvature.setter
    def curvature(self, value):
        if value != 0:
            raise cdg.util.errors.CDGError('You are not allowed to change the curvature.')

    def set_parameters(self, **kwargs):
        d = kwargs.pop('d')
        if kwargs.pop('curvature', 0) != 0:
            raise cdg.util.errors.CDGError("curvature has to be 0")
        super().set_parameters(ed=d, d=d, curvature=0, **kwargs)

    @classmethod
    def _distance(cls, X1_mat, X2_mat, **kwargs):
        D2 = cls._geo.distance_squared(X1_mat=X1_mat, X2_mat=X2_mat)
        condition = np.logical_and(D2 < 0, D2 > -1e-8)
        D2_clip = np.where(condition, 0., D2)
        return np.sqrt(D2_clip)

    @classmethod
    def scalarprod2distance(cls, scalar_prod_mat, radius):
        if scalar_prod_mat.shape[0] != scalar_prod_mat.shape[1]:
            raise NotImplementedError()

        D = -2. * scalar_prod_mat.copy()
        for i in range(scalar_prod_mat.shape[0]):
            D[i, :] += scalar_prod_mat[i, i]
        for j in range(scalar_prod_mat.shape[1]):
            D[:, j] += scalar_prod_mat[j, j]

        return D

    def distance2scalarprod(self, dist_mat):
        # Gn = -1/2 (dist_mat J - U D2_prot_mat J)
        D2n = np.power(dist_mat, 2)
        t, M = D2n.shape
        J = np.eye(M) - np.ones((M, M)) * 1. / M
        U = np.ones((t, M)) * 1. / M
        return -0.5 * (np.dot(D2n, J) - np.dot(np.dot(U, self._D2_prot), J))

    @classmethod
    def exp_map(cls, x0_mat, Nu_mat):
        return Nu_mat.copy()

    @classmethod
    def log_map(cls, x0_mat, X_mat):
        return X_mat.copy()

    @classmethod
    def clip(cls, X_mat, **kwargs):
        return X_mat.copy()

    @classmethod
    def sample_mean(cls, X, **kwargs):
        return _arrange_as_matrix(np.mean(X, axis=0))


class ConstantNonNullCurvarure(ConstantCurvarure):
    """
    Nonnull, yet constant, curvature. Indeed, many procedures can be abstracted.
    If the manifold dimension is d, then points are represented by d+1 coordinates in a hosting
    (embedding/immersion) vector space.
    """

    # sinus and cosinus functions to be instantiated
    _sinm = None
    _cosm = None

    @classmethod
    def _local_basis(cls, x0_mat, curvature=None):
        """
        Basis in the global frame of the tangent space on point x0
        :param x0_mat: (1, d+1)
        :param curvature:
        :return: (d+1, d)
        """

        dim = x0_mat.shape[1] - 1
        curvature = cls._curvature_from_datum(x0_mat)
        B_tmp = cls._geo._I(dim + 1) - np.dot(x0_mat.transpose(), x0_mat) * curvature
        indices = [i for i in range(dim + 1)]
        # check if its trivial
        found = False
        for i in range(dim + 1):
            noti = indices[:i] + indices[i + 1:]
            if np.isclose(x0_mat[0, noti], np.zeros((1, dim)), rtol=1e-4, atol=1e-4).all():
                Bp = B_tmp[:, noti]
                found = True
                break

        # select a non orthogonal column to drop
        if not found:
            for i in range(dim + 1):
                noti = indices[:i] + indices[i + 1:]
                if not np.isclose(np.dot(B_tmp[:, noti].transpose(), B_tmp[:, i]), \
                                  np.zeros((1, dim)), \
                                  rtol=1e-4, atol=1e-4).all():
                    Bp = B_tmp[:, noti]
                    found = True
                    break

        if np.linalg.matrix_rank(Bp) != dim or not found:
            raise cdg.util.errors.CDGImpossible()

        # gram schmidt
        B = np.zeros(Bp.shape)
        for i in range(dim):
            B[:, i] = Bp[:, i]
            for j in range(i):
                B[:, i] -= B[:, j] * cls._geo.scalar_product(B[None, :, i], B[None, :, j])[0, 0]
            B[:, i] /= cls._geo.norm(B[None, :, i])[0]

        return B

    @classmethod
    def exp_map(cls, x0_mat, Nu_mat):
        # tangent vectors in global coordinates
        B = cls._local_basis(x0_mat=x0_mat)
        V = np.dot(Nu_mat, B.transpose())
        # tangent vectors in global coordinates
        X = np.zeros(V.shape)
        for n in range(V.shape[0]):
            X[n:n + 1, :] = cls._exp_map_global_coord(x0=x0_mat, v=V[n:n + 1, :])
        return X

    @classmethod
    def log_map(cls, x0_mat, X_mat):
        # tangent vectors in global coordinates
        V = np.zeros(X_mat.shape)
        for n in range(X_mat.shape[0]):
            V[n:n + 1, :] = cls._log_map_global_coord(x0=x0_mat, x=X_mat[n:n + 1, :])
        # tangent vectors in local coordinates
        B = cls._local_basis(x0_mat=x0_mat)
        Nu = cls._geo.scalar_product(V, B.transpose())
        return Nu

    @classmethod
    def _exp_map_global_coord(cls, x0, v, theta=None):
        if theta is None:
            theta = cls._theta_v(x0=x0, v=v)
        if theta == 0:
            return x0
        elif np.isinf(cls._cosm(theta)) or np.isinf(cls._sinm(theta)):
            return x0
        return cls._cosm(theta) * x0 + cls._sinm(theta) / theta * (v)

    @classmethod
    def _log_map_global_coord(cls, x0, x, theta=None):
        if theta is None:
            theta = cls._theta_x(x0=x0, x=x)
        if theta == 0:
            return np.zeros(x0.shape)
        elif np.isinf(cls._cosm(theta)) or np.isinf(cls._sinm(theta)):
            return np.zeros(x0.shape)
        return theta / cls._sinm(theta) * (x - cls._cosm(theta) * x0)

    @classmethod
    def _theta_v(cls, x0, v):
        x0_mat = _arrange_as_matrix(X=x0)
        radius = cls._radius_from_datum(x_mat=x0_mat)
        v_mat = _arrange_as_matrix(X=v)
        th = cls._geo.norm(v_mat)[0, 0] / radius
        if th < 0:
            # From a theoretical point of view this can't happen because, despite the
            # pseudo-Euclidean geometry, the tangent space is Euclidean.
            raise cdg.util.errors.CDGImpossible()
        return th

    @classmethod
    def _theta_x(cls, x0, x):
        x0_mat = _arrange_as_matrix(X=x0)
        radius = cls._radius_from_datum(x_mat=x0_mat)
        return cls.distance(X1=x0_mat, X2=x, radius=radius)[0, 0] / radius

    @classmethod
    def _radius_from_datum(cls, x_mat):
        r = cls._geo.norm(X_mat=x_mat)[0, 0]
        return r if r > 0 else -r

    @classmethod
    def _curvature_from_datum(cls, x_mat):
        r2 = cls._geo.norm_squared(X_mat=x_mat)[0, 0]
        return 1 / r2

    @classmethod
    def _distance(cls, X1_mat, X2_mat, **kwargs):
        try:
            radius = kwargs.pop('radius', cls._radius_from_datum(x_mat=X1_mat[None, 0]))
        except KeyError:
            raise cdg.util.errors.CDGError('you didnt provide a radius')

        gramiam = cls._geo.scalar_product(X1_mat=X1_mat, X2_mat=X2_mat)
        D = cls.scalarprod2distance(gramiam, radius)
        return D

    def set_parameters(self, **kwargs):
        d = kwargs.pop('d')
        super().set_parameters(ed=d + 1, d=d, **kwargs)

    def reset(self):
        super().reset()
        self.curvature = None

    def __str__(self, extra=""):
        current_extra = '|r' + str(self.radius)
        current_extra += extra
        return super().__str__(extra=current_extra)


class SphericalManifold(ConstantNonNullCurvarure):
    """
    Positive curvatures.
    """
    _name = 'S'
    # Euclidean geometry of the hosting vector space
    _geo = cdg.util.geometry.Eu()
    # ordinary sine and cosine
    _sinm = np.sin
    _cosm = np.cos

    def set_radius(self, value):
        self.curvature = 1. / value ** 2

    @classmethod
    def clip(cls, X_mat, radius):
        norms = cls._geo.norm(X_mat=X_mat)
        X_mat = X_mat / norms
        return X_mat * radius

    @classmethod
    def scalarprod2distance(cls, scalar_prod_mat, radius):
        corr = scalar_prod_mat * 1. / radius ** 2
        condition = np.logical_and(np.abs(corr) > 1., np.abs(corr) < (1. + 1e-4))
        corr_clip = np.where(condition, 1., corr)
        with np.errstate(invalid='raise'):
            try:
                dtmp = np.arccos(corr_clip)
            except FloatingPointError:
                raise FloatingPointError('np.arccos:'
                                         'Out of bounds points: {}/1.0 '.format(condition.mean()) +
                                         'Min minus_corr_clip: {}'.format(np.min(corr_clip)))
        return radius * dtmp

    def distance2scalarprod(self, dist_mat):
        return np.cos(dist_mat * 1. / self.radius) * (self.radius ** 2)


class HyperbolicManifold(ConstantNonNullCurvarure):
    """
    Negative curvatures.
    """
    _name = 'H'
    # Pseudo-Euclidean geometry with signature (d,1) of the hosting vector space
    _geo = cdg.util.geometry.PEu1()
    # hyperbolic sine and cosine
    _sinm = np.sinh
    _cosm = np.cosh

    def set_radius(self, value):
        self.curvature = - 1. / value ** 2

    @classmethod
    def clip(cls, X_mat, radius):
        # norms = cls._geo.norm(X_mat = X_mat)
        # X_mat = X_mat / norms
        X_clipped = X_mat.copy()
        norms = cls._geo.norm(X_mat)
        for n in range(X_mat.shape[0]):
            if norms[n, 0] != -radius:
                norm2_pe = norms[n, 0] ** 2 * (-1 if norms[n, 0] < 0 else +1)
                norm2_e = norm2_pe + 2 * X_mat[n, -1] ** 2
                X_clipped[n, :] = cls.clip_single(X_mat[n, :], norm2_pe=norm2_pe, norm2_e=norm2_e,
                                                  radius=radius)
        return X_clipped

    @staticmethod
    def clip_single(y_vec, norm2_pe, norm2_e, radius):
        clipped = None
        if np.all(np.isclose(y_vec[:-1], 0)):
            clipped = np.zeros(y_vec.shape)
            clipped[-1] = radius
        elif np.isclose(y_vec[-1], 0):
            clipped = y_vec / 2
            clipped[-1] = np.sqrt(sum(clipped[:-1] ** 2) + radius ** 2)
        else:
            a = norm2_pe
            a /= (radius ** 2)
            b = norm2_e
            b /= (radius ** 2)
            coeff = [0] * 5
            coeff[0] = 1
            coeff[2] = a - 2
            coeff[3] = - 2 * b
            coeff[4] = a + 1
            roots = np.roots(coeff)
            t_list = []
            for root in roots:
                root_real = np.real_if_close(root, tol=1e-5).item()
                if not isinstance(root_real, complex):
                    t_list.append(root_real)

            for t in t_list:
                if t > -1:
                    xp = y_vec[:-1] / (1. + t)
                    xn = y_vec[-1:] / (1. - t)
                    if xn >= 0:
                        clipped = np.concatenate((xp, xn))
        if clipped is None:
            raise cdg.util.errors.CDGImpossible()
        return clipped

    @classmethod
    def scalarprod2distance(cls, scalar_prod_mat, radius):
        minus_corr = - scalar_prod_mat * 1. / radius ** 2
        condition = np.abs(minus_corr) < 1.
        minus_corr_clip = np.where(condition, 1., minus_corr)
        with np.errstate(invalid='raise'):
            dtmp = np.arccosh(minus_corr_clip)

        return radius * dtmp

    def distance2scalarprod(self, dist_mat):
        return - np.cosh(dist_mat * 1. / self.radius) * (self.radius ** 2)


class _ConstantCurvarureDR(ConstantCurvarure, cdg.embedding.embedding.DissimilarityRepresentation):
    """
    Embedding technique for costant curvature manifolds based on dissimilarity representation.
    This is a prototype based techniques trying to preserve the input distances.
    See alseo [1].
    """

    def __init__(self):
        super().__init__()
        self.skip_from_serialising(['X_training', '_XtX_prototypes', '_D2_prot'])

    def fit(self, dissimilarity_matrix, no_annealing=1):
        """
        Fit the embedding on a training set. Actually, the procedure is based only on
        a matrix of mutual dissimilarities between the (not given) training datapoints.
        :param dissimilarity_matrix: (n, n)
        :param no_annealing: number of annealing for the prototype selection
        :return: distances between the embedded points
        """
        self._set_dissimilarity_matrix(dissimilarity_matrix=dissimilarity_matrix)
        if self.radius is None:
            self._fit_radius()
        D_emb, err_drop = self._embed_training_set()
        _, err_prot = self._prototype_selection(dissimilarity_matrix=dissimilarity_matrix,
                                                no_annealing=no_annealing)
        self._XtX_prototypes = np.dot(self._Xt_prototypes, self._Xt_prototypes.transpose())
        return D_emb, err_drop, err_prot

    def _fit_radius(self):
        """ Finds the best performing radius """
        # create a discretized list of radii
        no_radii = 15
        min_norm_radius = 1. / np.pi  # see Wilson et al. [1]
        max_norm_radius = 3
        # rad_i = min_radius * fact**i
        # fact = (max_radius / min_radius)**(1/(n-1))
        fact = (max_norm_radius / min_norm_radius) ** (1. / (no_radii - 1))
        radius_list = [self._max_training_distance * min_norm_radius * fact ** i for i in
                       range(0, no_radii)]

        # check every radius
        error_list = []
        for r in tqdm(radius_list, desc='fit radius'):
            self.set_radius(r)
            D_emb, error_drop = self._embed_training_set_simplified()
            # debug
            error_dist = np.linalg.norm(self._dissimilarity_training - D_emb, 'fro')
            error_list.append(error_dist)

        self.set_radius(radius_list[np.argmin(error_list)])
        self.log.debug("radius set to " + str(self.radius))
        rmin = min(radius_list)
        rmax = max(radius_list)
        radii_norm = [1. * (r - rmin) / (rmax - rmin) for r in radius_list]

        diss_train_fro = np.linalg.norm(self._dissimilarity_training, 'fro')
        error_norm = [1. * e / diss_train_fro for e in error_list]
        error_norm = [1. * e for e in error_list]

        # self.logplot([[radii_norm ,error_norm]],style='+-g')
        self.logrun(lambda: plt.semilogy(radius_list, error_norm, '+-r'))
        self.log.debug("diss_train_fro " + str(diss_train_fro))
        self.log.debug("radii norm and err normalised " + str([radii_norm, error_norm]))
        self.log.debug("radii and err normalised " + str([radius_list, error_norm]))

    def _embed_training_set_simplified(self):
        return self._embedding_wilson_simplest()

    def _embed_training_set(self):
        return self._embedding_wilson_diagonal_constraint()

    def _embedding_wilson_simplest(self):
        """ This is the coarsest technique present in [1]."""
        gramiam = self.distance2scalarprod(dist_mat=self._dissimilarity_training)
        eig_val, eig_vec = _get_real_eig(gramiam)
        X_mat, error_eig = self._geo.reduced_solution(eig_vec, eig_val, self.embedding_dimension)

        self.X_training = self.clip(X_mat=X_mat, radius=self.radius)
        D_embedded = self.distance(X1=self.X_training, radius=self.radius)

        return D_embedded, error_eig

    def _embedding_wilson_diagonal_constraint(self):
        """ The improved technique present in [1]."""
        if not np.allclose(self._dissimilarity_training, self._dissimilarity_training.transpose(),
                           atol=1e-5):
            raise NotImplementedError("matrix not symmetric")

        # Compute embedding -- versione approssimata
        gramiam = self.distance2scalarprod(self._dissimilarity_training)
        eig_val, eig_vec = _get_real_eig(gramiam)

        b_star, exit = self.optimise_b(lam=eig_val, kappa=self.curvature)

        # sortet_index = np.array([i for i in range(0,U_s.shape[0])])
        sorted_index = np.argsort(b_star.ravel())[::-1]
        X, error_dropped_b = self._geo.reduced_solution(eig_vec[:, sorted_index],
                                                        b_star[sorted_index].ravel(),
                                                        self.embedding_dimension)

        self.X_training = self.clip(X_mat=X, radius=self.radius)

        D_embedded = self.distance(self.X_training, radius=self.radius)

        return D_embedded, error_dropped_b

    @classmethod
    def _optimise_b(cls, lam):
        raise cdg.util.errors.CDGAbstractMethod()

    def _embed_single_datum(self, y):
        # A xt = b  -->  Ipe XtX Ipe xt = Ipe Xt zt
        y_mat = _arrange_as_matrix(y)
        z = self.distance2scalarprod(y_mat)

        # Direct solution
        A = self._geo.scalar_product(self._XtX_prototypes, np.eye(self.embedding_dimension))
        b = np.dot(self._Xt_prototypes, z.transpose())
        x = np.linalg.solve(A, b)
        x = self.clip(x.transpose(), radius=self.radius)

        x_best = x.copy()

        # self.logplot([np.array(err_list)])
        return x_best.ravel()

    @classmethod
    def _embedding_wilson_optimisation(cls, x0, y, X_mat, curvature):
        """ Refinement of the embeddings exploiting the tangent space. See in [1]."""

        radius = 1. / np.sqrt(np.abs(curvature))

        num, dim = X_mat.shape
        # dim -= 1

        d_true = y.copy()
        d_emb = cls.distance(x0, X_mat, radius=radius).ravel()

        Delta = np.power(d_emb, 2) - np.power(d_true, 2)

        x0_mat = _arrange_as_matrix(x0)
        # Nu = np.zeros((num, dim-1))
        # for j in range(num):
        #     Nu[j, :] = cls.log_map(x0_mat = x0_mat, X_mat= X_mat[j:j+1,:])
        Nu = cls.log_map(x0_mat=x0_mat, X_mat=X_mat)

        Delta_mat = _arrange_as_matrix(Delta)
        # B_tmp = np.eye(dim) - np.dot(x0_mat.transpose(),x0_mat)*curvature
        # B2 = B_tmp[:,:-1]
        B = cls._local_basis(curvature=curvature, x0_mat=x0_mat, v=None)
        # if not np.allclose(B,B2):
        #     raise cdg.util.errors.CDGImpossible

        grad_EBa = - 4 * np.dot(cls._geo.scalar_product(B, Nu), Delta_mat.transpose())
        Bgrad_EBa = np.dot(B.transpose(), grad_EBa).transpose()

        alpha = cls._geo.scalar_product(Bgrad_EBa, Nu).ravel()

        norm2_grad = cls._geo.scalar_product(Bgrad_EBa, Bgrad_EBa)[0, 0]
        coeff = [0] * 4
        coeff[0] = num * norm2_grad ** 2
        coeff[1] = - 3 * norm2_grad * np.sum(alpha)
        coeff[2] = 2 * np.sum(np.power(alpha, 2)) + norm2_grad * np.sum(Delta)
        coeff[3] = - np.dot(alpha, Delta)

        roots = np.roots(coeff)
        err = np.linalg.norm(d_emb - d_true)
        err_new = err
        err_new = np.inf
        x_new = x0.copy()

        for i in range(3):
            eta_tmp = np.real_if_close(roots[i], tol=1e-8).item()
            if not isinstance(eta_tmp, complex):
                nu_tmp = eta_tmp * Bgrad_EBa
                x_tmp = cls.exp_map(x0_mat=x0_mat, Nu_mat=_arrange_as_matrix(nu_tmp))
                d_emb_tmp = cls.distance(x_tmp, X_mat, radius=radius).ravel()
                err_tmp = np.linalg.norm(d_emb_tmp - d_true)
                if err_tmp < err_new:
                    x_new = x_tmp.copy()
                    err_new = err_tmp

        return x_new, err_new, err > err_new

    def _prototype_selection(self, dissimilarity_matrix, no_annealing=1):
        prototypes, val = super()._prototype_selection(dissimilarity_matrix=dissimilarity_matrix,
                                                       no_annealing=no_annealing)
        self._Xt_prototypes = self.X_training[prototypes, :].transpose()
        return prototypes, val


class EuclideanDR(_ConstantCurvarureDR, EuclideanManifold):
    def __init__(self):
        super().__init__()
        self.skip_from_serialising(['_D2_prot'])

    def _prototype_selection(self, dissimilarity_matrix, no_annealing=1):
        prototypes, val = super()._prototype_selection(dissimilarity_matrix=dissimilarity_matrix,
                                                       no_annealing=no_annealing)
        self._D2_prot = self.distance(X1=self.X_prototypes, X2=self.X_prototypes)
        self._D2_prot = np.power(self._D2_prot, 2)
        return prototypes, val

    def _embed_training_set_simplified(self):
        return self._embedding_classical_MDS()

    def _embed_training_set(self):
        return self._embedding_classical_MDS()

    def _embedding_classical_MDS(self):
        n = self._dissimilarity_training.shape[0]
        J = np.eye(n) - np.ones((n, n)) * 1. / n
        D2 = np.power(self._dissimilarity_training, 2)
        gramiam = -.5 * np.dot(np.dot(J, D2), J)
        eig_val, eig_vec = _get_real_eig(gramiam)

        X, error_dropped_eig = self._geo.reduced_solution(eig_vec, eig_val,
                                                          self.embedding_dimension)

        self.X_training = X.copy()

        D_emb = self.distance(X, X)

        # debug
        error_dist = np.linalg.norm(self._dissimilarity_training - D_emb, 'fro')
        diss_train_fro = np.linalg.norm(self._dissimilarity_training, 'fro')
        error_norm = 1. * error_dist  # / diss_train_fro
        self.log.debug("diss_train_fro " + str(diss_train_fro))
        self.log.debug("radii and err normalised " + str([0, error_norm]))

        return D_emb, error_dropped_eig

    def _embed_single_datum(self, y):
        # Direct solution
        # Gn^t = X xt ->  XtX xt = XtGn^t
        y_mat = _arrange_as_matrix(y)
        G = self.distance2scalarprod(dist_mat=y_mat)
        x = np.linalg.solve(self._XtX_prototypes, np.dot(self._Xt_prototypes, G.transpose()))

        return x.ravel()

    @classmethod
    def optimise_b(cls, lam, kappa):
        return lam, True

    def _fit_radius(self):
        pass


class SphericalDR(_ConstantCurvarureDR, SphericalManifold):
    @classmethod
    def optimise_b(cls, lam, kappa):

        if lam[-1] < - 1. / kappa:
            beta = 1. / (1 - kappa * lam[-1])
            b = (1. - beta) * np.ones(lam.shape) / kappa + beta * lam
        else:
            b = lam

        return b, True


class HyperbolicDR(_ConstantCurvarureDR, HyperbolicManifold):
    @classmethod
    def optimise_b(cls, lam, kappa):

        exit = True
        if lam[-2] < .1 / kappa:  # this is an unfortunate case
            beta = 1. / (1 + kappa * lam[-2])
            cdg.util.logger.glog().warn("this may happen sometimes")
            exit = False
        elif lam[-1] < .1 / kappa and lam[-2] > 1. / kappa and lam[-2] < 0:
            beta = 1. / (1 + kappa * lam[-2])

        elif lam[-1] < 0 and lam[-2] > 0:
            beta = 1.

        elif lam[-1] > 0:
            beta = 1. / (1 + kappa * lam[-1])

        b = (1. - beta) * np.ones(lam.shape) / kappa + beta * lam
        return b, exit


#############################################################
# Tests
#############################################################


def demo_euclidean():
    n = 200
    d = 2
    seed = 1234
    radius = 5
    perturb = .0

    cdg.util.logger.set_stdout_level(cdg.util.logger.DEBUG)
    cdg.util.logger.enable_logrun(level=True)
    np.random.seed(seed=seed)

    # X_true=generate_banana_data(d=d, n=n, perturb=perturb)
    # X_true =generate_uniformplane_data(d=d, n=n, perturb=perturb)
    X_true = generate_uniformsemisphere_data(d=d, n=n, radius=radius, perturb=perturb)

    diss_matrix = EuclideanManifold.distance(X_true, X_true)

    man = EuclideanDR()
    man.set_parameters(d=d, M=d + 1)
    man.fit(dissimilarity_matrix=diss_matrix)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # ax.plot(X_true[:,0],X_true[:,1], 'g.')
    ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], 'g')
    ax.plot(man.X_training[:, 0], man.X_training[:, 1], 'r.')
    # ax.scatter(man.X_training[:,0],man.X_training[:,1],man.X_training[:,2], 'r')
    ax.plot(man.X_prototypes[:, 0], man.X_prototypes[:, 1], 'b.')
    plt.show()


def demo_spherical():
    n = 200
    d = 2
    seed = 1234
    radius = 5
    perturb = .0

    cdg.util.logger.set_stdout_level(cdg.util.logger.DEBUG)
    cdg.util.logger.enable_logrun(level=True)
    np.random.seed(seed=seed)

    # X_true=generate_banana_data(d=d, n=n, perturb=perturb)
    # X_true =generate_uniformplane_data(d=d, n=n, perturb=perturb)
    X_true = generate_uniformsemisphere_data(d=d, n=n, radius=radius, perturb=perturb)

    # diss_matrix = EuclideanManifold.distance(X_true, X_true)
    diss_matrix = SphericalManifold.distance(X_true, X_true, radius=radius)

    man = SphericalDR()
    man.set_parameters(d=d, M=d + 1)
    man.fit(dissimilarity_matrix=diss_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    x, y, z = get_sphere_coord(man.radius)
    ax.plot_surface(x, y, z, alpha=.2)
    # ax.plot(X_true[:,0],X_true[:,1], 'g.')
    ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], c='g')
    # ax.plot(man.X_training[:,0],man.X_training[:,1], 'r.')
    ax.scatter(man.X_training[:, 0], man.X_training[:, 1], man.X_training[:, 2], marker='+', c='r')
    ax.scatter(man.X_prototypes[:, 0], man.X_prototypes[:, 1], man.X_prototypes[:, 2], marker='*',
               c='k')
    # fig.show()
    plt.show()


def demo_hyperbolic():
    n = 200
    d = 2
    seed = 1234
    radius = 5
    perturb = .0

    cdg.util.logger.set_stdout_level(cdg.util.logger.DEBUG)
    cdg.util.logger.enable_logrun(level=True)
    np.random.seed(seed=seed)

    # X_true=generate_banana_data(d=d, n=n, perturb=perturb)
    # X_true =generate_uniformplane_data(d=d, n=n, perturb=perturb)
    X_true = generate_uniformsemisphere_data(d=d, n=n, radius=radius, perturb=perturb)

    # diss_matrix = EuclideanManifold.distance(X_true, X_true)
    diss_matrix = SphericalManifold.distance(X_true, X_true, radius=radius)

    man = HyperbolicDR()
    man.set_parameters(d=d, M=d + 1)
    man.fit(dissimilarity_matrix=diss_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # x, y, z = get_sphere_coord(man.radius)
    x, y, z = get_hyperboloid_coord(man.radius)
    ax.plot_surface(x, y, z, alpha=.2)
    # ax.plot(X_true[:,0],X_true[:,1], 'g.')
    ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], c='g')
    # ax.plot(man.X_training[:,0],man.X_training[:,1], 'r.')
    ax.scatter(man.X_training[:, 0], man.X_training[:, 1], man.X_training[:, 2], marker='+', c='r')
    ax.scatter(man.X_prototypes[:, 0], man.X_prototypes[:, 1], man.X_prototypes[:, 2], marker='*',
               c='k')
    # fig.show()
    plt.show()


def generate_uniformplane_data(n, d, perturb=0):
    # XX = np.random.rand(n, d-1)
    XX = np.random.rand(n, d - 1)
    X1 = np.dot(XX, np.ones((d - 1, 1)))
    X = np.zeros((n, d))
    X[:, :1] += X1
    X[:, 1:] += XX

    if perturb != 0:
        X += np.random.normal(scale=perturb, size=X.shape)

    return X


def generate_banana_data(n, d, perturb=0):
    # XX = np.random.rand(n, d-1)
    alpha = np.random.rand(n, 1) * np.pi * .7
    X2 = np.concatenate((np.sin(alpha), np.cos(alpha)), axis=1)
    X = np.zeros((n, d))
    X[:, :2] += X2

    if perturb != 0:
        X += np.random.normal(scale=perturb, size=X.shape)

    return X


def generate_uniformsemisphere_data(n, d, radius, perturb=0):
    # XX = np.random.rand(n, d-1)
    XX = np.random.normal(size=(n, d)) * .03
    # X1 = np.random.rand(n,1)*.02 + 0.1
    X1 = np.ones((n, 1)) * .1

    X = np.zeros((n, d + 1))
    X[:, :1] = X1
    X[:, 1:] = XX

    X = SphericalManifold.clip(X_mat=X, radius=radius)

    if perturb != 0:
        raise NotImplementedError()

    return X


def get_sphere_coord(radius):
    if radius is None:
        radius = 1
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z


def get_hyperboloid_coord(radius):
    if radius is None:
        radius = 1
    u = np.linspace(0, 2 * np.pi, 20)
    rho = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), rho)
    y = np.outer(np.sin(u), rho)
    z = np.sqrt(x ** 2 + y ** 2 + radius ** 2)

    return x, y, z


if __name__ == "__main__":
    # demo_euclidean()
    # demo_spherical()
    demo_hyperbolic()
