"""
We provide a barebone implementation of Latent Dirichlet Allocation with Variational Inference
We diverge from Blei's paper and the general literature on a couple of points:

- We don't use the empirical Bayes method to estimate the parameter for the Dirichlet that governs
topic proportions
- We don't assume a Dirichlet prior on \Beta, the topic-vocabulary matrix
"""

import numpy as np
import scipy.special as spec
import scipy.stats as stats
from scipy.misc import logsumexp

from _lda_helpers import mean_change_2d, mean_change
from joblib import Parallel, delayed


def get_slices(n, n_buckets):
    """
    Given a number n, and the number of buckets to divide it into,
    this routine produces a list of slices to index an array of n dimensions
    into even parts. 
    
    e.g. given 300 and 45, this routine produces a list 
    [slice(0, 45, None), slice(45, 90, None ...]
    
    The remainder is added to the last slice. As a quick reference, the python
    `slice` is a primitive that corresponds to the shorthand notation 1:5. 
    
    e.g. let a = np.array([1,2,3,4,5])
    a[1:4] is the same thing as a[slice(1,4,None)]
    
    :param n: number of items to bucket
    :param n_buckets: the number of buckets
    :rtype: list
    :return: the list of slice objects
    """
    bucket = n // n_buckets
    slices = []
    for i in range(n_buckets):
        if i < n_buckets - 1:
            slices.append(slice(i*bucket, (i+1)*bucket))
        else:
            slices.append(slice(i*bucket, None))
    return slices


def _slice_doc_update(X, gamma, beta, alpha, slice, eta=None, f=None):
    """
    Given a data array X, and a slice object `slice`, take the 
    subset of documents corresponding to the slice and for each
    document, maximize the variational lower bound on gamma (variational
    Dirichlet prior) and accumulate sufficient statistics over documents
    that will be used to estimate the new Beta (in the M-step)
    
    This method had to be removed from the LDA class scope in order to 
    parallelize using joblib. Each subprocess will invoke this method
    and optimize over a subset of documents.
    
    :type X: scipy.sparse.csr_matrix
    :param X: The document-term matrix of shape (n_documents, n_terms)
    
    :type gamma: numpy.array
    :param gamma: The variational Dirichlet prior of shape (n_topics, n_documents) 
        i.e. the document-topic prior
    
    :type beta: numpy.array
    :param beta: The current value (last calculated in the M-step) of the topic-word distributions
        array of shape (n_topics, n_terms)
    
    :type alpha: float
    :param alpha: Hyperparameter, the parameter to the exchangeable Dirichlet prior over Theta
    
    :type slice: slice
    :param slice: the slice object to index the documents assigned to this subprocess
    
    :rtype: tuple
    :return: _loc_beta, the element of accumulated beta that will come from this subprocess
             _loc_gamma, the slice of gamma that is updated by this subprocess
    """
    
    Xsl = X[slice, :]
    
    sl_length, V = Xsl.shape  # grab slice length 
    
    _loc_beta = np.zeros(beta.shape)  # get a local beta
    _loc_gamma = gamma[:, slice]  # get local gamma slice
    _loc_bound = 0
    _loc_logw = 0
    
    if eta is not None:
        _loc_eta = np.zeros(eta.shape)

    for m in xrange(sl_length):
        # an index to the words of this document is generated
        ixw = Xsl.indices[Xsl.indptr[m]:Xsl.indptr[m+1]]  # index optimized for sparse matrices
        
        if eta is not None:
            eta_ixw = eta[:, f[ixw]]
            logw, bound, gammad, phi = _doc_update(_loc_gamma[:, m], np.log(beta[:, ixw]), alpha,
                                                   logeta_ixw=np.log(eta_ixw))
        else:
            logw, bound, gammad, phi = _doc_update(_loc_gamma[:, m], np.log(beta[:, ixw]), alpha)
        
        _loc_gamma[:, m] = gammad  # assignment by reference!!
        
        counts = Xsl[m, ixw].A.squeeze()
        _loc_beta[:, ixw] += phi * counts 
        # _loc_eta[:, ]
        
        if eta is not None:
            _loc_eta += _phi_for_f(ixw, phi, f, counts)
            # for s in range(_loc_eta.shape[1]):
            #     etym_ix = (f[ixw] == s)
                
        
        _loc_bound += bound
        _loc_logw += np.sum(logw)
    
    return_tuple = (_loc_beta, _loc_gamma, _loc_bound, _loc_logw)
    if eta is not None: return_tuple += (_loc_eta,)
    
    return return_tuple


def _phi_for_f(ixw, phi, f, counts):
    F = len(np.unique(f))
    K, V = phi.shape
    phi_adjusted = np.zeros((K, F))
    
    for s in range(F):
        etym_ix = (f[ixw] == s)
        phi_adjusted[:, s] = np.sum(phi[:, etym_ix] * counts[etym_ix], axis=1)
        
    return phi_adjusted


def _doc_lowerbound(phi, gamma, logbeta_ixw, alpha, logeta_ixw = None):
    tmp = (spec.digamma(gamma) - spec.digamma(np.sum(gamma)))
    mean_log_ptheta = np.log(spec.gamma(np.sum(alpha))) - \
                      np.sum(np.log(spec.gamma(alpha))) +\
                      np.sum((alpha - 1) * tmp)
    mean_log_pz = np.sum(phi.T * tmp)
    mean_log_pw = np.sum(phi * logbeta_ixw)
    if logeta_ixw is not None:
        mean_log_pw += np.sum(phi * logeta_ixw)
    neg_mean_log_qtheta = stats.dirichlet.entropy(gamma)
    neg_mean_log_qz = -np.sum(phi * np.log(phi))

    bound = mean_log_ptheta + mean_log_pz + mean_log_pw + neg_mean_log_qtheta + neg_mean_log_qz

    return bound


def _doc_probability(gammad, logbeta_ixw, logeta_ixw=None):
    '''
    Compute p(w_d) whose parameters we know,
    :return: log-probability distribution over words of the document
    '''

    logpw = logsumexp(logbeta_ixw.T + np.log(stats.dirichlet.mean(gammad)), axis = 1)
    return logpw

def _heldout_doc_probability(alpha, logbeta_ixw, logeta_ixw = None):
    '''
    Compute p(w_d) for a held-out document
    :return: log-probability
    '''
    K, V = logbeta_ixw.shape
    gammad = np.zeros(K) + alpha + (V/float(K))
    _, inferred_gamma, _ = _doc_update(gammad, logbeta_ixw, alpha, logeta_ixw = logeta_ixw)
    return _doc_probability(gammad, logbeta_ixw, logeta_ixw=logeta_ixw)


def _doc_update(gammad, logbeta_ixw, alpha, tol=1e-2, logeta_ixw=None):
    """
    Take an E update step for a document. Runs the variational inference iteration
    per document until convergence or maxiter of 200 is reached. 

    :type ixw: numpy.array
    :param ixw: the index to the words appearing in the document
    
    :type gammad: numpy.array
    :param gammad: current assignment to this document's gamma, the var. Dir. prior (n_topics, n_documents)
    
    :type beta_ixw: numpy.array
    :param beta_ixw: current assignment to beta, the topic-word distribution (n_topics, n_words_appearing)
        beta_ixw = beta[:, ixw]
    
    :type eta_ixw: numpy.array
    :param eta_ixw:
    
    :type alpha: float
    :param alpha: parameter to the Dirichlet prior over topic-document distribution
    
    :rtype: tuple
    :return: gammad: the updated gamma column for this document
             phi: the variational multinomial prior for this document
             ixw: index to the words appearing in this document (array of ints)
    """
    # TODO: this method should see only what it should see!
    
    # index to the words appearing in the document
    
    K, N = logbeta_ixw.shape
    
    phi = np.zeros((K, N), dtype=float) + 1./K  # only appearing words get a phi

    # store the previous values for convergence check
    phi_prev = phi.copy()
    gammad_prev = gammad.copy()
    
    # calculate bounds
    bound = -float("inf")
    bound_prev = _doc_lowerbound(phi, gammad, logbeta_ixw, alpha)

    for ctr in xrange(200):
        # update phi
        # WARN: exp digamma underflows < 1e-3!
        # TODO: carry this to the log domain?
        log_mult = logbeta_ixw.T
        if logeta_ixw is not None:
            log_mult += logeta_ixw.T
        
        logphi = (log_mult + spec.digamma(gammad)).T
        logphi -= logsumexp(logphi, 0)
        phi = np.exp(logphi)  # normalize phi columns

        # update gamma
        gammad = alpha + np.sum(phi, axis=1)

        if ctr % 20 == 0:  # check convergence
            bound = _doc_lowerbound(phi, gammad, logbeta_ixw, alpha, logeta_ixw)
            if bound - bound_prev < tol:
                break
            bound_prev = bound

    bound = _doc_lowerbound(phi, gammad, logbeta_ixw, alpha)
    log_w = _doc_probability(gammad, logbeta_ixw, logeta_ixw)
    return log_w, bound, gammad, phi


class LDA(object):

    def __init__(self, K=5, alpha=None, n_jobs=8, nr_em_epochs=10):
        """
        Construct the LDA model (i.e. do not run it yet)
        
        :type K: int
        :param K: the number of topics
        
        :type alpha: float
        :param alpha: is the hyperparameter to the model, this implementation assumes an exchangeable Dirichlet
        
        :type n_jobs: int
        :param n_jobs: how many CPUs to use?
        
        :type nr_em_epochs: int
        :param nr_em_epochs: number of EM iterations to perform
        """
        self.alpha = alpha
        if self.alpha is None:
            self.alpha = 1./K
        self.K = K

        self.n_jobs = n_jobs
        self.nr_em_epochs = nr_em_epochs

    def fit(self, X, f=None):
        """
        Fit the LDA model using the variational-EM algorithm (Blei et al., 2003).
        
        :type X: scipy.sparse.csr_matrix
        :param X: the term-document matrix, of type (n_documents, n_terms)
        
        :param f: np.array [1,2,3,3,3,4,1,2,3...] (nr_terms,)
        
        :rtype: tuple
        :return: beta: the fitted topic-term distribution (n_topics, n_terms)
                 gamma: the fitted var. Dir prior (n_topics, n_documents)
        """
        
        perplexity = float("inf")
        K = self.K # number of topics
        alpha = self.alpha
        M, V = X.shape

        nr_terms = X.sum(axis=1)
        nr_terms = np.array(nr_terms).squeeze()

        # model parameters
        beta = np.random.rand(K, V)
        
        # multimodal model parameters
        if f is not None: 
            S = len(np.unique(f))
            eta = np.random.rand(K, S)
        
        # initialize the parallel processing pool
        par = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")

        # slice the documents for multiprocessing
        slices = get_slices(M, self.n_jobs)

        for epoch in xrange(self.nr_em_epochs):
            log_w = 0.
            
            # TODO: calculate bound function and check EM convergence 
            # E-step
            print "Epoch:", epoch

            # initialize variables
            gamma = np.zeros((K, M)) + alpha + (nr_terms/float(K))  # mth document, i th topic
            beta_acc = np.zeros((K, V))
            

            # work on each slice in parallel
            if f is not None:
                eta_acc = np.zeros((K, S))
                res = par(delayed(_slice_doc_update)(X, gamma, beta, alpha, slice, eta, f) for slice in slices)
            else:
                res = par(delayed(_slice_doc_update)(X, gamma, beta, alpha, slice) for slice in slices)
            
            # do things in series - for profiling purposes
            # res = [_slice_doc_update(X, gamma, beta, alpha, slice) for slice in slices]

            # sync barrier
            for ix, r in enumerate(res):
                gamma[:, slices[ix]] = r[1]  # update gammas
                beta_acc += r[0]  # update betas
                log_w += r[3]
                if f is not None:
                    eta_acc += r[4]

            # M-step
            beta = self._m_step(beta_acc)
            if f is not None:
                eta = self._m_step(eta_acc)

            # quality - p(w) is the normalizing constant of the posterior
            # and it is intractable - bound gives an estimate
            perplexity = self._perplexity(X, log_w)
            print "Perplexity:", perplexity

        return_tuple = (beta, gamma)
        if f is not None:
            return_tuple += (eta,)
        
        return return_tuple  # the parameters learned

    def _m_step(self, beta_acc):
        """
        Take the Maximization step of the algorithm
        
        :param beta_acc: The beta suff stats accumulated in the E-step
        :return: normalized new betas
        """
        # TODO: check for numerical stability
        return (beta_acc.T / np.sum(beta_acc, axis=1)).T # normalize beta rows


    def _perplexity(self, X, log_w):
        """
        This is just an approximation. Ideally, it should compute the doc probabilities
        after the M-step, but here we use the previously assigned \Beta's
        :return: perplexity, higher means high surprise (not desired)
        """
        return np.exp(-log_w/X.sum())


