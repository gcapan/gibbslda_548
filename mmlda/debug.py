import pickle

from lda import LDA

from data.datafile import AADataFile
dfile = pickle.load(open("data/datafile.pkl"))

dt = dfile.DT
te = dfile.TE


lda = LDA(K=10, n_jobs=8, nr_em_epochs=20)

b, g = lda.fit(dt)