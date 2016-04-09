"""
We provide a barebone implementation of Latent Dirichlet Allocation with Variational Inference
We depart from Blei's paper and the general literature on a couple of points:

- We don't use the empirical Bayes method to estimate the parameter for the Dirichlet that governs
topic proportions
- We don't assume a Dirichlet prior on \Beta, the topic-vocabulary matrix
"""

class LDA(object):

    def __init__(self, K=5):
        """

        :param K: the number of topics
        :return:
        """
        pass

    def fit(self, X, alpha):
        """

        :param X: the term-document matrix
        :param alpha: is the hyperparameter to the model, this implementation assumes an exchangeable Dirichlet
        :return:
        """
        pass