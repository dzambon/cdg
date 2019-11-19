# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements several embedding methods.
# --------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm

import cdg.utils

import cdg.geometry.prototype
import cdg.geometry

NO_EMBEDDING = 'NO_EMBEDDING'

class Embedding(cdg.utils.Loggable, cdg.utils.Pickable):
    '''
    Very generic embedding.
    '''

    _name = 'GenericEmbedding'
    # dimension of the vector representation
    embedding_dimension = None

    def __init__(self, **kwargs):
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)
        self.log.debug(str(self) + " created")
        self.set_parameters(**kwargs)

    def __str__(self, extra=''):
        return '{}(ed{})'.format(self._name, self.embedding_dimension)

    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - emb_dim : (required) dimension of the embedding representation.
        """
        self.embedding_dimension = kwargs.pop('emb_dim', None)
        # raise cdg.utils.AbstractMethodError()

    def reset(self):
        """ Clean the embedding instance from dependency"""
        raise cdg.utils.AbstractMethodError()

    def fit(self):
        raise cdg.utils.AbstractMethodError()

    def transform(self):
        raise cdg.utils.AbstractMethodError()

    def fit_transform(self):
        raise cdg.utils.AbstractMethodError()


class DissimilarityRepresentation(Embedding):

    _name = 'DissRep'
    _distance_function = None
    _use_precomputed_distance = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skip_from_serialising(['_distance_function'])

    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - emb_dim or nprot: dimension of the embedding representation which is equal to
                the number of prototypes/landmarks.
        """
        nprot = kwargs.get('nprot', None)
        emb_dim = kwargs.get('emb_dim', None)
        if nprot is not None and emb_dim is not None:
            assert nprot == emb_dim, 'This is required in DissimilarityRepresentation.'
        elif nprot is None and emb_dim is None:
            return
        elif nprot is not None:
            kwargs['emb_dim'] = nprot
        elif emb_dim is not None:
            nprot = emb_dim

        self.no_prototypes = nprot
        super().set_parameters(**kwargs)

    def fit(self, graphs=None, dist_fun=None, prec_diss_mat=None, **kwargs):
        self.fit_transform(graphs=graphs, dist_fun=dist_fun, prec_diss_mat=prec_diss_mat, **kwargs)

    def fit_transform(self, graphs=None, dist_fun=None, prec_diss_mat=None, **kwargs):
        # Process input + compute diss_mat + select and store prototypes
        diss_mat = self._fit_transform_core(graphs=graphs, dist_fun=dist_fun, prec_diss_mat=prec_diss_mat, **kwargs)
        # Extract dissimilarity representation
        return diss_mat[:, self._prototypes_idx]

    def _fit_transform_core(self, graphs, dist_fun, prec_diss_mat, **kwargs):
        
        # Compute dissimilarity matrix parsing the inputs
        if prec_diss_mat is not None:
            self._use_precomputed_distance = True
            return prec_diss_mat
        assert graphs is not None and dist_fun is not None, "I expect you to pass me a list of graphs and a distance function."
        self._distance_function = dist_fun
        diss_mat = self._distance_function(graphs, graphs)
        
        # Select and store prototypes
        fun = lambda: cdg.geometry.prototype.k_centers(dissimilarity_matrix=diss_mat, n_prototypes=self.no_prototypes)
        self._prototypes_idx, _ = cdg.utils.anneal(no_annealing=3, fun=fun, verbose=True, desc='prot. sel. annealing')
        if self._use_precomputed_distance:
            self.prototypes = None
        else:
            self.prototypes = [graphs[p] for p in self._prototypes_idx]
        
        return diss_mat

    def transform(self, data, **kwargs):
        '''
        
        :param data: If it has been fit with graphs and a distance function,
            than data should be a list of graphs, otherwise a dissimilarity representation
            of the graphs with respect to the prototypes (recoverable from mds.prototypes_idx)
            
        :param n_jobs:
        :return:
        '''
        if not self._use_precomputed_distance:
            graphs = data
            diss_rep = self._distance_function(graphs, self.prototypes)
        else:
            diss_rep = data
        return diss_rep
        

class MultiDimensionalScaling(DissimilarityRepresentation):
    _name = 'MDS'
    _mds = None
    
    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - emb_dim : (required) dimension of the embedding representation.
            - nprot : (required) number of prototypes/landmarks.
        """
        self.embedding_dimension = kwargs.pop('emb_dim')
        self.no_prototypes = kwargs.pop('nprot', self.embedding_dimension+1)
    
    def fit(self, graphs=None, dist_fun=None, prec_diss_mat=None, n_jobs=None, **kwargs):
        self.fit_transform(graphs=graphs, dist_fun=dist_fun, prec_diss_mat=prec_diss_mat, **kwargs)
    
    def fit_transform(self, graphs=None, dist_fun=None, prec_diss_mat=None, **kwargs):
        # Process input + compute diss_mat + select and store prototypes
        diss_mat = self._fit_transform_core(graphs=graphs, dist_fun=dist_fun, prec_diss_mat=prec_diss_mat, **kwargs)
        # Multidimensional Scaling
        # x = self._sklearn_fit_transform(diss_mat=diss_mat, embedding_dimension=self.embedding_dimension)
        x = self._classical_fit_transform(diss_mat=diss_mat)
        # Precompute matrices for out-of-sample embedding
        self._Xt_prot = (x[self._prototypes_idx].T).copy()
        self._XtX_prot = self._Xt_prot.dot(self._Xt_prot.T)
        self._D2_prot = cdg.geometry.Eu.distance(self._Xt_prot.T, self._Xt_prot.T)
        #debug
        if np.any(np.isnan(self._D2_prot)):
            self._D2_prot = cdg.geometry.Eu.distance(self._Xt_prot.T, self._Xt_prot.T)
        return x
    
    @staticmethod
    def _sklearn_fit_transform(diss_mat, embedding_dimension):
        from sklearn.manifold import MDS
        mds = MDS(n_components=embedding_dimension)
        return mds.fit_transform(X=diss_mat, dissimilarity='precomputed')
    
    def _classical_fit_transform(self, diss_mat):
        if not np.allclose(diss_mat, diss_mat.T):
            rel_error = (np.abs(diss_mat - diss_mat.T)/diss_mat.max()).max()
            msg = 'Distance matrix needs to be symmetric (rel_error: {:0.3f}).'.format(rel_error)
            if rel_error < 0.05:
                self.log.warning(msg)
                self.log.warning('I am making it symmetric: A = (A + A.T) * .5 ')
                diss_mat = (diss_mat + diss_mat.T) * .5
            else:
                raise ValueError(msg)
        n = diss_mat.shape[0]
        J = np.eye(n) - np.ones((n, n)) * 1. / n
        D2 = np.power(diss_mat, 2)
        gramiam = -.5 * np.dot(np.dot(J, D2), J)
        eig_val, eig_vec = cdg.utils.get_real_eig(gramiam)
        X, error_dropped_eig = cdg.geometry.Eu.reduced_solution(eig_vec, eig_val, self.embedding_dimension)
        return X
    
    def transform(self, data):
        # Compute dissimilarity representation
        # dist_mat = super(DissimilarityRepresentation, self).transform(data)
        dist_mat = DissimilarityRepresentation.transform(self, data)

        # Compute Graham matrix
        # Gn = -1/2 (dist_mat J - U D2_prot_mat J)
        # G = cdg.geometry.manifold.EuclideanManifold().distance2scalarprod(dist_mat=diss_rep)
        ngraphs, nprot = dist_mat.shape
        D2n = np.power(dist_mat, 2)
        J = np.eye(nprot) - np.ones((nprot, nprot)) * 1. / nprot
        U = np.ones((ngraphs, nprot)) * 1. / ngraphs
        Gn = -0.5 * (D2n - U.dot(self._D2_prot)).dot(J)
        
        # Compute embeddings
        # Gn = Xn X^T  =>  X Xn^T = Gn^T  =>  X^T X Xn^T = X^T Gn^T
        # Gn = Xn X^T
        x = np.linalg.solve(self._XtX_prot, self._Xt_prot.dot(Gn.T))
        return x.T



class NonNullCurvatureMDS(DissimilarityRepresentation):
    _name = 'CCM-MDS'
    _manifold = None
    
    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - manifold : it can be either a manifold instance, or a string
                in set {'spherical', 'hyperbolic'}.
            - man_dim : (required if manifold is a string, ignored otherwise)
                dimension of the manifold representation.
            - nprot : (optional if manifold is a string, ignored otherwise)
                number of prototypes/landmarks.
        """
        # self.manifold_dimension = kwargs.pop('man_dim')
        # self.embedding_dimension = self.manifold_dimension + 1
        # self.no_prototypes = kwargs.pop('nprot', None)
        # if self.no_prototypes is None:
        #     self.no_prototypes = self.manifold_dimension + 1
        # assert self.no_prototypes > self.manifold_dimension
        
        manifold = kwargs.pop('manifold')
        
        if isinstance(manifold, str):
            man_dim = kwargs.pop('man_dim')
            if manifold == 'spherical':
                self._manifold = cdg.geometry.SphericalManifold(man_dim=man_dim)
            elif manifold == 'hyperbolic':
                self._manifold = cdg.geometry.HyperbolicManifold(man_dim=man_dim)
            else:
                raise ValueError('manifold <{}> not recognised.'.format(manifold))
        else:
            self._manifold = manifold
            man_dim = self._manifold.manifold_dimension
        # self.embedding_dimension = man_dim + 1 # todo check if necessary
        self.no_prototypes = kwargs.pop('nprot', man_dim + 1)
        assert self.no_prototypes > man_dim
    
    def fit(self, graphs=None, dist_fun=None, prec_diss_mat=None):
        self.fit_transform(graphs=graphs, dist_fun=dist_fun, prec_diss_mat=prec_diss_mat)
    
    def fit_transform(self, graphs=None, dist_fun=None, prec_diss_mat=None):
        # Process input + compute diss_mat + select and store prototypes
        dist_mat = self._fit_transform_core(graphs=graphs, dist_fun=dist_fun, prec_diss_mat=prec_diss_mat)
        # CCM Multidimensional Scaling
        if self._manifold.radius is None:
            self._fit_radius(manifold=self._manifold, dist_mat=dist_mat)
        # x = self._sklearn_fit_transform(diss_mat=diss_mat, embedding_dimension=self.embedding_dimension)
        x, _, _ = self._embed_training_set(dist_mat=dist_mat, manifold=self._manifold)
        # Precompute matrices for out-of-sample embedding
        # todo
        self._X_training = x
        self._Xt_prot = (x[self._prototypes_idx].T).copy()
        self._XtX_prot = self._Xt_prot.dot(self._Xt_prot.T)
        # self._D2_prot = cdg.geometry.manifold.EuclideanManifold.distance(x[:, self._prototypes_idx],
        #                                                                  x[:, self._prototypes_idx])
        return x
    
    def _fit_radius(self, manifold, dist_mat):
        """ Finds the best performing radius """
        # create a discretized list of radii
        no_radii = 15
        min_norm_radius = 1. / np.pi  # see Wilson et al. [1]
        max_norm_radius = 3
        # rad_i = min_radius * fact**i
        # fact = (max_radius / min_radius)**(1/(n-1))
        fact = (max_norm_radius / min_norm_radius) ** (1. / (no_radii - 1))
        max_training_distance = np.max(dist_mat)
        radius_list = [max_training_distance * min_norm_radius * fact ** i for i in
                       range(0, no_radii)]
        
        # check every radius
        error_list = []
        for r in tqdm(radius_list, desc='fit radius'):
            manifold.set_radius(r)
            _, D_emb, error_drop = self._embed_training_set_simplified(manifold, dist_mat)
            # debug
            error_dist = np.linalg.norm(dist_mat - D_emb, 'fro')
            error_list.append(error_dist)
        
        manifold.set_radius(radius_list[np.argmin(error_list)])
        self.log.debug("radius set to " + str(manifold.radius))
        rmin = min(radius_list)
        rmax = max(radius_list)
        radii_norm = [1. * (r - rmin) / (rmax - rmin) for r in radius_list]
        
        diss_train_fro = np.linalg.norm(dist_mat, 'fro')
        error_norm = [1. * e / diss_train_fro for e in error_list]
        error_norm = [1. * e for e in error_list]
        
        # self.logplot([[radii_norm ,error_norm]],style='+-g')
        self.logrun(lambda: plt.semilogy(radius_list, error_norm, '+-r'))
        self.log.debug("diss_train_fro " + str(diss_train_fro))
        self.log.debug("radii norm and err normalised " + str([radii_norm, error_norm]))
        self.log.debug("radii and err normalised " + str([radius_list, error_norm]))
    
    @classmethod
    def _embed_training_set_simplified(cls, manifold, dist_mat):
        '''
        Simplified and lighter embedding method used for, e.g., estimate the radius
        '''
        return cls._embedding_wilson_simplest(manifold, dist_mat)
    
    def _embed_training_set(self, manifold, dist_mat):
        '''
        More sofisticated embedding method used for the actual embedding
        '''
        return self._embedding_wilson_diagonal_constraint(manifold=manifold, dist_mat=dist_mat)
    
    @classmethod
    def _embedding_wilson_simplest(cls, manifold, dist_mat):
        """ This is the coarsest technique presented in [1]."""
        gramiam = manifold.distance2scalarprod(dist_mat=dist_mat)
        eig_val, eig_vec = cdg.utils.get_real_eig(gramiam)
        X_mat, error_eig = manifold.reduced_solution(eig_vec=eig_vec,
                                                     eig_val=eig_val,
                                                     emb_dim=manifold.manifold_dimension+1)
        
        X_training = manifold.clip(X_mat=X_mat, radius=manifold.radius)  # todo radius shouldnt be here
        D_embedded = manifold.distance(X1=X_training, radius=manifold.radius)  # todo radius shouldnt be here
        
        return X_training, D_embedded, error_eig
        
    def _embedding_wilson_diagonal_constraint(self, manifold, dist_mat):
        """ The improved technique present in [1]."""
        assert np.allclose(dist_mat, dist_mat.T, atol=1e-5)

        # Compute embedding -- versione approssimata
        gramiam = manifold.distance2scalarprod(dist_mat)
        eig_val, eig_vec = cdg.utils.get_real_eig(gramiam)

        b_star, exit = manifold.optimise_b(lam=eig_val, kappa=manifold.curvature)

        # sortet_index = np.array([i for i in range(0,U_s.shape[0])])
        sorted_index = np.argsort(b_star.ravel())[::-1]
        X, error_dropped_b = manifold._geo.reduced_solution(eig_vec[:, sorted_index],
                                                        b_star[sorted_index].ravel(),
                                                        manifold.manifold_dimension+1)

        X_training = manifold.clip(X_mat=X, radius=manifold.radius)
        D_embedded = manifold.distance(X_training, radius=manifold.radius)

        return X_training , D_embedded, error_dropped_b

    def transform(self, data):
        # Compute dissimilarity representation
        diss_rep = DissimilarityRepresentation.transform(self, data)
    
        # Compute Graham matrix
        z = self._manifold.distance2scalarprod(diss_rep)

        # Direct solution
        # A xt = b  -->  Ipe XtX Ipe xt = Ipe Xt zt
        A = self._manifold._geo.scalar_product(self._XtX_prot, np.eye(self._manifold.manifold_dimension+1))
        b = self._Xt_prot.dot(z.T)
        x = np.linalg.solve(A, b).T
        return self._manifold.clip(x, radius=self._manifold.radius)
        # x_best = x.copy()
        #
        # # self.logplot([np.array(err_list)])
        # return x_best.ravel()

class SphericalMDS(NonNullCurvatureMDS):
    _name = 'S-MDS'
    
    def __init__(self, **kwargs):
        assert kwargs.get('manifold', 'spherical') == 'spherical'
        super().__init__(**kwargs)

class HyperbolicMDS(NonNullCurvatureMDS):
    _name = 'H-MDS'
    
    def __init__(self, **kwargs):
        assert kwargs.get('manifold', 'hyperbolic') == 'hyperbolic'
        super().__init__(**kwargs)

