import numpy as np
import scipy as sp
import scipy.sparse as spar
import scipy.special as spec
import sys
from matplotlib import pyplot as plt
from lda import LDA

from sklearn.decomposition import LatentDirichletAllocation as SKLDA
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import cProfile

M = 100
V = 1000
X = np.random.binomial(1,.3, size=M*V).reshape(M,V)
X = spar.csr_matrix(X, dtype=float)


# get the data
ng = fetch_20newsgroups(subset='train')

# vectorize the data
vec = CountVectorizer(max_df=.7, min_df=20)
ngvec = vec.fit_transform(ng.data)

lda = LDA(n_jobs=8)
b, g = lda.fit(ngvec)