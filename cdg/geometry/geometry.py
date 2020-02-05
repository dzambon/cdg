# --------------------------------------------------------------------------------
# Copyright (c) 2017-2020, Daniele Zambon, All rights reserved.
#
# Defines the Euclidean and pseudo-Euclidean geometries.
# --------------------------------------------------------------------------------
import numpy as np
import cdg.utils
import cdg.geometry

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

    X = cdg.geometry.manifold.SphericalManifold.clip(X_mat=X, radius=radius)

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


class Geometry(object):

    @classmethod
    def _I(cls, n):
        raise cdg.utils.AbstractMethodError()

    @classmethod
    def scalar_product(cls, X1_mat, X2_mat):
        raise cdg.utils.AbstractMethodError()

    @classmethod
    def norm_squared(cls, X_mat):
        raise cdg.utils.AbstractMethodError()

    @classmethod
    def norm(cls, X_mat):
        raise cdg.utils.AbstractMethodError()

    @classmethod
    def reduced_solution(cls, eig_vec, eig_val, dim):
        raise cdg.utils.AbstractMethodError()


class Eu(Geometry):

    @classmethod
    def _I(cls, n):
        return np.eye(n)

    @classmethod
    def scalar_product(cls, X1_mat, X2_mat):
        return X1_mat.dot(X2_mat.T)

    @classmethod
    def norm(cls, X_mat):
        return np.linalg.norm(X_mat, axis=1)[..., None]

    @classmethod
    def norm_squared(cls, X_mat):
        return cls.norm(X_mat)**2

    @classmethod
    def reduced_solution(cls, eig_vec,eig_val,dim):
        lambda_abs = np.abs(eig_val[:dim])
        lambda_mat = np.diag(np.sqrt(lambda_abs))
        return np.dot(eig_vec[:, :dim], lambda_mat), sum(lambda_abs[dim:])


    @classmethod
    def distance_squared(cls, X1_mat, X2_mat):
        D2 = -2. * cls.scalar_product(X1_mat=X1_mat, X2_mat=X2_mat)
        D2 += cls.norm_squared(X_mat=X1_mat)
        D2 += cls.norm_squared(X_mat=X2_mat).T
        assert D2.min() > -1e-10
        return np.clip(D2, a_min=0., a_max=None)

    @classmethod
    def distance(cls, X1_mat, X2_mat):
        return np.sqrt(cls.distance_squared(X1_mat=X1_mat, X2_mat=X2_mat))

class PEu1(Geometry):

    @classmethod
    def _I(cls, n):
        a = np.eye(n)
        a[-1,-1]=-1
        return a

    @classmethod
    def scalar_product(cls, X1_mat, X2_mat):
        return X1_mat.dot(cls._I(n=X2_mat.shape[1]).dot(X2_mat.T) )

    @classmethod
    def norm_squared(cls, X_mat):
        return np.sum(X_mat.dot(cls._I(n=X_mat.shape[1])) * X_mat, axis=1)[..., None]

    @classmethod
    def norm(cls, X_mat):
        norm2 = cls.norm_squared(X_mat)
        return np.sqrt(np.abs(norm2)) *np.sign(norm2)

    @classmethod
    def reduced_solution(cls, eig_vec, eig_val, dim):
        X = np.zeros((len(eig_val),dim))
        lambda_abs = np.abs(eig_val)
        lambda_mat = np.diag(np.sqrt(lambda_abs))
        X[:,:dim-1] = cls.scalar_product(eig_vec[:,:dim-1], lambda_mat[:dim-1,:dim-1])
        X[:, -1:]   = cls.scalar_product(eig_vec[:, -1:], lambda_mat[-1:, -1:])
        return X, sum(lambda_abs[dim-1:-1])


class CCRiemannianManifold(cdg.utils.Loggable, cdg.utils.Pickable):
    """
    Defines the structure of a constant-curvature Riemannian manifold.
    This class is intended to be extended as spherical and hyperbolic manifolds.
    Notice that if the manifold dimension is d, then points are represented by
    d+1 coordinates in an ambient space vector space.
    The general notation is:
        - x, X: points on the manifold
        - nu, Nu: points on the tangent space in local coordinates
        - v, V: points on the tangent space in global coordinates, that is with
            respect to the representation of the manifold
    """
    
    _name = 'ConstantCurvatureRiemannianManifold'
    curvature = None
    manifold_dimension = None
    _geo = None  # geometry of the ambient space.
    _sinm = None  # sine function to be instantiated
    _cosm = None  # cosine function to be instantiated
    
    def __init__(self, **kwargs):
        self.log.debug('{} created'.format(self))
        self.set_parameters(**kwargs)
    
    def __str__(self, extra=''):
        return self._name + "(d{}|r{}{})".format(self.manifold_dimension, self.radius, extra)
    
    def set_parameters(self, **kwargs):
        self.manifold_dimension = kwargs.pop('man_dim', self.manifold_dimension)
        self.curvature = kwargs.pop('curvature', None)
        self.set_radius(kwargs.pop('radius', None))
    
    @property
    def radius(self):
        if self.curvature == 0 or self.curvature is None:
            return None
        else:
            return 1. / np.sqrt(np.abs(self.curvature))
    
    @classmethod
    def exp_map(cls, x0_mat, Nu_mat):
        """
        Exponential map from tangent space to the manifold
        :param x0_mat: point of tangency
        :param Nu_mat: (n, man_dim) points on the tangent space to be mapped
        :return: X_mat: points on the manifold
        """
        # tangent vectors in global coordinates
        B = cls._local_basis(x0_mat=x0_mat)
        V_mat = np.dot(Nu_mat, B.transpose())
        # tangent vectors in global coordinates
        X_mat = cls._exp_map_global_coord(x0_mat=x0_mat, v_mat=V_mat)
        return X_mat
    
    @classmethod
    def log_map(cls, x0_mat, X_mat):
        """
        Logarithm map from manifold to tangent space.
        :param x0_mat: point of tangency
        :param X_mat: (n, emb_dim) points on the manifold to be mapped
        :return: Nu_mat: points on the tangent space
        """
        # tangent vectors in global coordinates
        V_mat = cls._log_map_global_coord(x0_mat=x0_mat, x_mat=X_mat)
        # tangent vectors in local coordinates
        B = cls._local_basis(x0_mat=x0_mat)
        Nu_mat = cls._geo.scalar_product(V_mat, B.transpose())
        return Nu_mat
    
    @classmethod
    def sample_mean(cls, X, **kwargs):
        X_mat = cdg.utils.arrange_as_matrix(X=X)
        # find argmin_{x\in X} \sum_i \rho(x,X_i)^2
        Dn = cls.distance(X1=X_mat, **kwargs)
        mean_id, _ = cdg.geometry.prototype.mean(dissimilarity_matrix=Dn, power=2)
        x_new = X_mat[mean_id:mean_id + 1, :].copy()
        
        # Optimise
        xk = x_new.copy() + 10.  # hack to be neq x_new
        iter_max = 10
        ct = 0
        
        while ct < iter_max and not np.allclose(xk, x_new):
            ct += 1
            xk = x_new.copy()
            Nu = cls.log_map(x0_mat=xk, X_mat=X_mat)
            mean_log = cdg.utils.arrange_as_matrix(np.mean(Nu, axis=0))
            assert not np.isnan(mean_log[0]).any()
            x_new = cls.exp_map(x0_mat=xk, Nu_mat=mean_log)
            
            # check overflow
            if np.linalg.norm(x_new) > 1e4:
                raise ValueError('Risk of overflow')
        return x_new
    
    @classmethod
    def clip(cls, X_mat, radius):
        raise cdg.utils.AbstractMethodError()
    
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
        X1_mat = cdg.utils.arrange_as_matrix(X=X1)
        if X2 is None:
            only_upper_triangular = True
            X2_mat = X1_mat.copy()
        else:
            only_upper_triangular = False
            X2_mat = cdg.utils.arrange_as_matrix(X=X2)
        assert X1_mat.shape[1] == X2_mat.shape[1]
        
        # actual computation
        D = cls._distance(X1_mat=X1_mat, X2_mat=X2_mat,
                          only_upper_triangular=only_upper_triangular,
                          **kwargs)
        assert np.argwhere(np.isnan(D)).shape[0] == 0
        return D
    
    @classmethod
    def _local_basis(cls, x0_mat, curvature=None):
        """
        Basis in the global frame of the tangent space on point x0
        :param x0_mat: (1, d+1)
        :param curvature:
        :return: (d+1, d)
        """
        dim = x0_mat.shape[1] - 1
        curvature = cls._curvature_from_datum(x0_mat) if curvature is None else curvature
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
    def _exp_map_global_coord(cls, x0_mat, v_mat, theta=None):
        if theta is None:
            theta = cls._theta_v(x0_mat=x0_mat, v_mat=v_mat)
        if theta.ndim == 0:
            theta = theta[..., None]
        mask = theta[..., 0] == 0  # Check which elements would be divided by 0
        theta[mask] = 1.  # Replace theta=0 with theta=1 so the division will have no effects
        output = cls._cosm(theta) * x0_mat + cls._sinm(theta) / theta * (v_mat)  # Compute values
        output[mask] = x0_mat  # Replace values that vould have been NaNs with x0
        return output
    
    @classmethod
    def _log_map_global_coord(cls, x0_mat, x_mat, theta=None):
        if theta is None:
            theta = cls._theta_x(x0_mat=x0_mat, x_mat=x_mat)
        if theta.ndim == 0:
            theta = theta[..., None]
        mask = theta[..., 0] == 0  # Check which elements would result in division by 0
        theta[mask] = 1.  # Replace theta=0 with theta=1 so the division will have no effects
        output = theta / cls._sinm(theta) * (x_mat - cls._cosm(theta) * x0_mat)
        output[mask] = np.zeros(x0_mat.shape)  # Replace values that vould have been NaNs with zeros
        return output
    
    @classmethod
    def _theta_v(cls, x0_mat, v_mat):
        # x0_mat = arrange_as_matrix(X=x0)
        radius = cls._radius_from_datum(x_mat=x0_mat)
        th = cls._geo.norm(v_mat) / radius
        assert np.all(th >= 0)  # From a theoretical point of view this can't happen because, despite the
        # pseudo-Euclidean geometry, the tangent space is Euclidean.
        return th
    
    @classmethod
    def _theta_x(cls, x0_mat, x_mat):
        radius = cls._radius_from_datum(x_mat=x0_mat)
        # raise NotImplementedError('qui devo vedere se usare una distanza paired')
        return cls.distance(X1=x_mat, X2=x0_mat, radius=radius) / radius
    
    @classmethod
    def _radius_from_datum(cls, x_mat):
        r = cls._geo.norm(X_mat=x_mat[:1])[0, 0]
        return r if r > 0 else -r
    
    @classmethod
    def _curvature_from_datum(cls, x_mat):
        r2 = cls._geo.norm_squared(X_mat=x_mat)[0, 0]
        return 1 / r2
    
    def reduced_solution(self, eig_vec, eig_val, emb_dim):
        return self._geo.reduced_solution(eig_vec=eig_vec, eig_val=eig_val, dim=emb_dim)
    
    @classmethod
    def _distance(cls, X1_mat, X2_mat, **kwargs):
        """
        Core part of the distance computation.
        :param X1_mat: input points (n1, emb_dim)
        :param X2_mat: input points (n2, emb_dim)
        """
        radius = kwargs.pop('radius', None)
        if radius is None:
            radius = cls._radius_from_datum(x_mat=X1_mat[:1])
            # nn = max([X1_mat.shape[0], 10])
            # radius = 0
            # for x in X1_mat[:nn]:
            #     radius += cls._radius_from_datum(x_mat=x[None, ...]) / nn
        
        gramiam = cls._geo.scalar_product(X1_mat=X1_mat, X2_mat=X2_mat)
        D = cls._scalarprod2distance(gramiam, radius)
        return D
    
    @classmethod
    def _scalarprod2distance(cls, scalar_prod_mat, radius):
        raise cdg.utils.AbstractMethodError()
    
    def distance2scalarprod(self, dist_mat):
        raise cdg.utils.AbstractMethodError()


class SphericalManifold(CCRiemannianManifold):
    """
    Positive curvatures.
    """
    _name = 'S'
    # Euclidean geometry of the hosting vector space
    _geo = Eu()
    # ordinary sine and cosine
    _sinm = np.sin
    _cosm = np.cos
    
    def set_radius(self, value):
        if value is not None:
            self.curvature = 1. / value ** 2
    
    @classmethod
    def clip(cls, X_mat, radius):
        norms = cls._geo.norm(X_mat=X_mat)
        X_mat = X_mat / norms
        return X_mat * radius
    
    @classmethod
    def _scalarprod2distance(cls, scalar_prod_mat, radius):
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

    @staticmethod
    def optimise_b(lam, kappa):
        if lam[-1] < - 1. / kappa:
            beta = 1. / (1 - kappa * lam[-1])
            b = (1. - beta) * np.ones(lam.shape) / kappa + beta * lam
        else:
            b = lam
    
        return b, True


class HyperbolicManifold(CCRiemannianManifold):
    """
    Negative curvatures.
    """
    _name = 'H'
    # Pseudo-Euclidean geometry with signature (d,1) of the hosting vector space
    _geo = PEu1()
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
                X_clipped[n, :] = cls.clip_single(X_mat[n, :],
                                                  norm2_pe=norm2_pe, norm2_e=norm2_e,
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
    def _scalarprod2distance(cls, scalar_prod_mat, radius):
        minus_corr = - scalar_prod_mat * 1. / radius ** 2
        condition = np.abs(minus_corr) < 1.
        minus_corr_clip = np.where(condition, 1., minus_corr)
        with np.errstate(invalid='raise'):
            dtmp = np.arccosh(minus_corr_clip)
        
        return radius * dtmp
    
    def distance2scalarprod(self, dist_mat):
        return - np.cosh(dist_mat * 1. / self.radius) * (self.radius ** 2)

    @staticmethod
    def optimise_b(lam, kappa):

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
