# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements some prototype selection algorithms based on a dissimilarity matrix.
# --------------------------------------------------------------------------------
import numpy as np

def get_kd_dissimilarity_matrix(data):
    """
    data must be nxk
    """
    gram = np.dot(data, data.transpose())
    dissimilarity_matrix = np.empty((data.shape[0],data.shape[0]), dtype=float)
    for i in range(0,data.shape[0]):
        dissimilarity_matrix[i,i] = np.sqrt(gram[i,i])

    for i in range(0,data.shape[0]):
        for j in range(i+1,data.shape[0]):
            dissimilarity_matrix[i,j] = np.sqrt( gram[i,i] -2*gram[i,j] + gram[j,j])
            dissimilarity_matrix[i,j] = dissimilarity_matrix[j,i]

    return dissimilarity_matrix


def median(dissimilarity_matrix):
    """
    Median graph. See Definition 6.4 in [1].
    """
    median = -1
    ct = -1
    median_sum = np.inf
    for row in dissimilarity_matrix:
        ct += 1
        row_sum = sum(row)
        if row_sum < median_sum:
            median = ct
            median_sum = row_sum

    return median

def center(dissimilarity_matrix):
    """
    Center graph. See Definition 6.4 in [1].
    """

    center = -1
    ct = -1
    center_max = np.inf
    for row in dissimilarity_matrix:
        ct += 1
        row_max = max(row)
        if row_max < center_max:
            center = ct
            center_max = row_max

    return center

def marginal(dissimilarity_matrix):
    """
    Marginal graph. See Definition 6.4 in [1].
    """
    marg = -1
    ct = -1
    marg_sum = 0
    for row in dissimilarity_matrix:
        ct += 1
        row_sum = sum(row)
        if row_sum > marg_sum:
            marg = ct
            marg_sum = row_sum

    return marg

def spanning(dissimilarity_matrix, n_prototypes=3):
    """
    The first prototype is the set median graph, each additional prototype
    selected by the spanning prototype selector is the graph furthest away
    from the already selected prototype graphs [1].

    :param dissimilarity_matrix:
    :param n_prototypes:
    :return:
    """

    nr, nc = dissimilarity_matrix.shape
    prototypes = []
    prototypes.append(median(dissimilarity_matrix))

    val_hat = 0

    for pi in range(1, n_prototypes):

        val = 0

        for i in range(1, nc):

            # controlla se gia selezionato
            found = False
            for p in prototypes:
                if p == i:
                    found = True
                    break

            # trova il migliore
            if not found:
                tmp = min(dissimilarity_matrix[i, prototypes])
                if tmp > val:
                    val = tmp
                    p_new = i

        if val>val_hat:
            val_hat = val

        # aggiorna
        prototypes.append(p_new)

    return prototypes, val_hat

# def MP( dissimilarity_matrix, n_prototypes=3, display_values=False, value_fun="ell-1"):
def matching_pursuit( dissimilarity_matrix, n_prototypes=3, display_values=False, value_fun="ell-1"):
    """
    Similar to the Matching Pursuit.

    prototypes = []
    while length(prototypes) < n_prototypes
        \\ell 1
        new_prototype = arg min { sum_t [ min_p d(p,t) ] }
        \\min
        new_prototype = arg min { max_t [ min_p d(p,t) ] }
        \\min max
        new_prototype = arg min { sum_t [ min_p d(p,t) ] - sum_p d(p,p')}

        prototypes.append( new_prototype )
    """
    nr, nc = dissimilarity_matrix.shape

    # init set of prototype candidates
    T = []
    for n in range(0, nr):
        T.append(n)

    # # Compute Prototype Set
    # _hat = best so far
    # _bar = current candidate
    P = []
    V = []
    # cycle until we have the num of prototypes we want
    cycles = n_prototypes
    if display_values:
        cycles = nr

    for n in range(0, cycles):
        val_hat = np.inf
        p_hat = -1

        # sweep all candidates
        for p_bar in T:
            # test on a candidate
            P.append(p_bar)

            # assess temporary maximal distance
            #   find the minima in columns of diss_mat[P][:]
            #   sum the minima
            if value_fun == "ell-1":
                # ell-1
                val = sum(np.min(dissimilarity_matrix[P, :], axis=0))
                # for t in T:
                #     val += min(dissimilarity_matrix[p][t] for p in P)
            elif value_fun == "min":
                # max dist
                val = np.max(np.min(dissimilarity_matrix[P, :], axis=0))
            elif value_fun == "min-max":
                # max distance
                val = np.max(np.min(dissimilarity_matrix[P, :], axis=0))
                val -= sum(dissimilarity_matrix[p_bar, P])
            else:
                raise ValueError("The value function '" + value_fun + "' not recognized")

            # save the maximum so far
            if val < val_hat:
                val_hat = val
                p_hat = p_bar

            # remove the tested candidate
            P.remove(p_bar)

        # once a candidate has been selected, update the sets
        P.append(p_hat)
        T.remove(p_hat)
        V.append(val_hat)

    # check for a better set of prototypes
    amv = np.argmin(V[:n_prototypes])
    n_prot_suggested = n_prototypes
    if n_prot_suggested > amv + 1:
        print("The suggested number of prototype is ", amv + 1)

    # this is necessary in case we have continued exploring
    P = P[:n_prototypes]

    return P, val_hat

def k_centers(dissimilarity_matrix, n_prototypes=3):

    nr, nc = dissimilarity_matrix.shape

    # init prototypes
    datapoints = [i for i in range(0, nr)]
    prototypes = np.random.choice(datapoints, n_prototypes, False)

    itermax = 100
    ct = 0
    changed = True
    while changed and ct < itermax:

        changed = False

        # assegna classe
        c = [[] for p in prototypes]
        for d in datapoints:
            dist = np.inf
            c_tmp = 0
            for pi in range(0, n_prototypes):
                if dissimilarity_matrix[prototypes[pi], d] < dist:
                    dist = dissimilarity_matrix[prototypes[pi], d]
                    c_tmp = pi
            c[c_tmp].append(d)

        # aggiorna prototipi
        val = [np.inf for p in prototypes]

        for pi in range(0, n_prototypes):
            p_new = prototypes[pi]
            for d in c[pi]:
                tmp = max(dissimilarity_matrix[d, c[pi]])
                if tmp < val[pi]:
                    val[pi] = tmp
                    p_new = d
            if prototypes[pi] != p_new:
                prototypes[pi] = p_new
                changed = True

        ct += 1
    return prototypes, max(val)

def mean(dissimilarity_matrix, power = 2):

    nr, nc = dissimilarity_matrix.shape

    D_power = dissimilarity_matrix.copy()
    D_power = np.power(D_power,power)

    minimum_so_far = np.inf
    candidate_mean = 0
    for mean_i in range(0,nr):
        tmp = np.sum(D_power[mean_i,:])
        if tmp < minimum_so_far:
            minimum_so_far=tmp
            candidate_mean=mean_i 

    return candidate_mean, tmp