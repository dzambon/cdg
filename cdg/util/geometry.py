# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Euclidean and psuedo-Euclidean geometry with the same framework.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 15/03/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np

class Geometry():

    @classmethod
    def distance_squared(cls, X1_mat, X2_mat):

        D2 = -2. * cls.scalar_product(X1_mat=X1_mat, X2_mat=X2_mat)
        D2 += cls.norm_squared(X_mat=X1_mat)
        D2 += cls.norm_squared(X_mat=X2_mat).transpose()

        return D2

    @classmethod
    def norm_squared(cls, X_mat):
        n = X_mat.shape[0]
        norms2 = np.zeros((n,1))
        for i in range(n):
            norms2[i, 0] = cls.scalar_product(X1_mat=X_mat[i:i+1, :], X2_mat=X_mat[i:i+1, :])[0,0]
        return norms2

    @classmethod
    def norm(cls, X_mat):
        norms2 = cls.norm_squared(X_mat)
        n = X_mat.shape[0]
        norms = np.zeros((n,1))
        for i in range(n):
             norms[i,0] = np.sqrt(norms2[i,0]) if norms2[i,0]>=0 else -np.sqrt(-norms2[i,0])
        return norms

class Eu(Geometry):

    @classmethod
    def _I(cls, n):
        return np.eye(n)

    @classmethod
    def scalar_product(cls, X1_mat, X2_mat):
        return np.dot(X1_mat, X2_mat.transpose())

    @classmethod
    def reduced_solution(cls, eig_vec,eig_val,dim):
        lambda_abs = np.abs(eig_val[:dim])
        lambda_mat = np.diag(np.sqrt(lambda_abs))
        return np.dot(eig_vec[:, :dim], lambda_mat), sum(lambda_abs[dim:])
               

class PEu1(Geometry):

    @classmethod
    def _I(cls, n):
        a = np.eye(n)
        a[-1,-1]=-1
        return a

    @classmethod
    def scalar_product(cls, X1_mat, X2_mat):
        return np.dot(X1_mat, np.dot(cls._I(n=X2_mat.shape[1]), X2_mat.transpose() ) )


    @classmethod
    def reduced_solution(cls, eig_vec,eig_val,dim):
        X = np.zeros((len(eig_val),dim))
        lambda_abs = np.abs(eig_val)
        lambda_mat = np.diag(np.sqrt(lambda_abs))
        X[:,:dim-1] = cls.scalar_product(eig_vec[:,:dim-1], lambda_mat[:dim-1,:dim-1])
        X[:, -1:]   = cls.scalar_product(eig_vec[:, -1:], lambda_mat[-1:, -1:])
        return X, sum(lambda_abs[dim-1:-1])