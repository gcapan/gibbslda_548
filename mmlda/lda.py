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


def get_slices(n, n_buckets):
    bucket = n // n_buckets
    slices = []
    for i in range(n_buckets):
        if i < n_buckets - 1:
            slices.append(slice(i*bucket, (i+1)*bucket))
        else:
            slices.append(slice(i*bucket, None))


class LDA(object):

    def __init__(self, K=5, alpha=None):
        """

        :param K: the number of topics
        :param alpha: is the hyperparameter to the model, this implementation assumes an exchangeable Dirichlet
        :return:
        """
        self.alpha = alpha
        if self.alpha is None:
            self.alpha = 1./K
        self.K = K

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

        for epoch in xrange(10):
            # E-step

            print "Epoch:", epoch

            gamma = np.zeros((K, M)) + alpha + (nr_terms/float(K)) # mth document, i th topic
            beta_acc = np.zeros((K, V))

            for m in xrange(M):  # iterate over all documents

                gammad, phi, ixw = self._doc_update(m, X, K, gamma, beta, alpha)

                # if m > 5:
                #     break

                gamma[:, m] = gammad
                beta_acc[:, ixw] += phi

            # M-step
            # TODO: check for numerical stability
            beta = self._m_step(beta_acc)

        return (beta, gamma) # the parameters learned

    def _doc_update(self, m, X, K, gamma, beta, alpha):
        """
        Take an E update step for a document

        :param m: the index to the document
        :return:
        """
        #ixw = X[m, :].nonzero()[1]  # an index to words which have appeared in the document
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
        pass