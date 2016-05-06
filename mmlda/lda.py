"""
We provide a barebone implementation of Latent Dirichlet Allocation with Variational Inference
We depart from Blei's paper and the general literature on a couple of points:

- We don't use the empirical Bayes method to estimate the parameter for the Dirichlet that governs
topic proportions
- We don't assume a Dirichlet prior on \Beta, the topic-vocabulary matrix
"""

import numpy as np
import scipy.special as spec

from _lda_helpers import mean_change_2d, mean_change
from joblib import Parallel, delayed


def get_slices(n, n_buckets):
    bucket = n // n_buckets
    slices = []
    for i in range(n_buckets):
        if i < n_buckets - 1:
            slices.append(slice(i*bucket, (i+1)*bucket))
        else:
            slices.append(slice(i*bucket, None))
    return slices


def _slice_doc_update(X, K, gamma, beta, alpha, slice):
    """
    
    :param Xsl: X 
    :param K: number of topics
    :param gamma: 
    :param beta: 
    :param alpha: 
    :param slice: the slice itself
    :return: 
    """
    # TODO: partition X into 2-3 cores and parallelize
    
    Xsl = X[slice, :]
    
    sl_length, V = Xsl.shape  # grab slice length 
    
    _loc_beta = np.zeros(beta.shape)  # get a local beta
    _loc_gamma = gamma[:, slice]  # get local gamma
    
    for m in xrange(sl_length):
        gammad, phi, ixw = _doc_update(m, Xsl, K, gamma, beta, alpha)
        
        _loc_gamma[:, m] = gammad
        _loc_beta[:, ixw] += phi
        
    return _loc_beta, _loc_gamma


def _doc_update(m, X, K, gamma, beta, alpha):
        """
        Take an E update step for a document

        :param m: the index to the document
        :return:
        """
        ixw = X.indices[X.indptr[m]:X.indptr[m+1]]  # index optimized for sparse matrices
        
        phi = np.zeros((K, len(ixw)), dtype=float) + 1./K  # only appearing words get a phi

        # slice for the document only once
        gammad = gamma[:, m]
        beta_ixw_T = (beta[:, ixw]).T

        # store the previous values for convergence check
        phi_prev = phi.copy()
        gammad_prev = gammad.copy()

        for ctr in xrange(200):
            # update phi
            # WARN: exp digamma underflows < 1e-3!
            phi = (beta_ixw_T * np.exp(spec.digamma(gammad))).T
            phi /= np.sum(phi, 0)  # normalize phi columns

            # update gamma
            gammad = alpha + np.sum(phi, axis=1)

            if ctr % 20 == 0:  #check convergence
                dphinorm = mean_change_2d(phi, phi_prev)
                dgammadnorm = mean_change(gammad, gammad_prev)

                # print dphinorm, dgammadnorm

                phi_prev = phi.copy()
                gammad_prev = gammad.copy()

                if dphinorm < 1e-1 and dgammadnorm < 1e-1:
                # if dgammadnorm < .01:
                    break

        return gammad, phi, ixw


class LDA(object):

    def __init__(self, K=5, alpha=None, n_jobs=3):
        """

        :param K: the number of topics
        :param alpha: is the hyperparameter to the model, this implementation assumes an exchangeable Dirichlet
        :param n_jobs: how many CPUs?
        :return:
        """
        self.alpha = alpha
        if self.alpha is None:
            self.alpha = 1./K
        self.K = K
        
        self.n_jobs = n_jobs

    def fit(self, X):
        """

        :param X: the term-document matrix, of type (n_documents, n_terms)
        :type X: scipy.sparse.csr_matrix
        :return:
        """

        K = self.K # number of topics
        alpha = self.alpha
        M, V = X.shape

        nr_terms = X.sum(axis=1)
        nr_terms = np.array(nr_terms).squeeze()

        # model parameters
        beta = np.random.rand(K, V)
        
        # initialize the parallel processing pool
        par = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")
        
        # slice the documents for multiprocessing
        slices = get_slices(M, self.n_jobs)

        for epoch in xrange(10):
            # E-step

            print "Epoch:", epoch
            
            # initialize variables
            gamma = np.zeros((K, M)) + alpha + (nr_terms/float(K)) # mth document, i th topic
            beta_acc = np.zeros((K, V))
            
            # work on each slice in parallel
            res = par(delayed(_slice_doc_update)(X, K, gamma, beta, alpha, slice) for slice in slices)
            
            # res = []
            # # DEBUG: instead work in series
            # for i in range(len(slices)):
            #     res.append(_doc_update()
            
            # sync barrier
            for ix, r in enumerate(res):
                gamma[:, slices[ix]] = r[1]  # update gammas
                beta_acc += r[0]  # update betas

            # M-step
            # TODO: check for numerical stability
            beta = self._m_step(beta_acc)

        return (beta, gamma) # the parameters learned

    def _m_step(self, beta_acc):
        """
        Take the Maximization step of the algorithm
        :param beta_acc:
        :return: normalized betas
        """
        return (beta_acc.T / np.sum(beta_acc, axis=1)).T # normalize beta rows

    def _bound(self):
        """
        TODO: Calculate the lower bound function to check convergence
        :return:
        """
        # TODO: implement bound function
        # TODO: implement perplexity
        pass