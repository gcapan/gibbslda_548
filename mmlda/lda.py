"""
We provide a barebone implementation of Latent Dirichlet Allocation with Variational Inference
We depart from Blei's paper and the general literature on a couple of points:

- We don't use the empirical Bayes method to estimate the parameter for the Dirichlet that governs
topic proportions
- We don't assume a Dirichlet prior on \Beta, the topic-vocabulary matrix
"""

import numpy as np
import scipy.special as spec

class LDA(object):

    def __init__(self, K=5, alpha=.1):
        """

        :param K: the number of topics
        :param alpha: is the hyperparameter to the model, this implementation assumes an exchangeable Dirichlet
        :return:
        """
        self.alpha = alpha
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

        for epoch in range(30):
            # E-step

            gamma = np.zeros((K, M)) + alpha + (nr_terms/float(K)) # mth document, i th topic
            beta_acc = np.zeros((K, V))

            for m in range(M):  # iterate over all documents

                phi = np.zeros((K, V), dtype=float) + 1./K
                ixw = (X[m, :] > 0).toarray().squeeze()  # an index to words which have appeared in the document

                gammad = gamma[:, m]  # slice for the document only once

                for ctr in range(int(1000)):
                    # store the previous values
                    # TODO: make this more efficient
                    phi_prev = phi.copy()
                    gammad_prev = gammad.copy()

                    # update phi
                    # WARN: exp digamma underflows < 1e-3!
                    phi[:, ixw] = ((beta[:, ixw]).T * np.exp(spec.digamma(gammad))).T
                    phi = phi / np.sum(phi, 0)  # normalize phi columns

                    # update gamma
                    gammad = alpha + np.sum(phi, axis=1)

                    # check for convergence
                    dphinorm = np.linalg.norm(phi - phi_prev, "fro")
                    dgammadnorm = np.linalg.norm(gammad - gammad_prev)

                    if dphinorm < .01 and dgammadnorm < .01:
                        break

                gamma[:, m] = gammad
                beta_acc[:, ixw] += phi[:, ixw]

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