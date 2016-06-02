import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from lda import LDA, _doc_update, _slice_doc_update

# Generate some data:
V = 1000
M = 500
numwords = 40
K = 5
alpha = 1./5
lmda = 1

#Top down LDA data
X = sp.coo_matrix((M, V)).tolil()
beta = np.zeros((K, V))
for k in range(K):
    beta[k, :] = np.random.dirichlet(np.ones(V)*lmda)
for d in range(M):
    theta_d = np.random.dirichlet(np.ones(K)*alpha)
    zs = np.random.choice(np.arange(K), size=numwords, p=theta_d)
    for z in zs:
        w_n = np.random.choice(np.arange(V), p=beta[z, :])
        X[d, w_n] += 1

lda = LDA(alpha=alpha, lmda=lmda, nr_em_epochs=2)
print "All collapsed"
props, word_props, log_Xs, perp = lda.collapsed_gibbs_sample(X)
plt.plot(range(len(log_Xs)), log_Xs, '*-')
plt.show()

plt.plot(range(len(perp)), perp, 'o-')
plt.show()


print "No collapsing"
props, word_props, log_Xs, perp = lda.gibbs_sample(X)
plt.plot(range(len(log_Xs)), log_Xs, '*-')
plt.show()

plt.plot(range(len(perp)), perp, 'o-')
plt.show()


print "Collapsed theta, not beta"
props, word_props, log_Xs, perp = lda.collapsed_theta_gibbs_sample(X)
plt.plot(range(len(log_Xs)), log_Xs, '*-')
plt.show()

plt.plot(range(len(perp)), perp, 'o-')
plt.show()









