import numpy as np

class AADataFile(object):

    def __init__(self, DT, TE, etym_classes, categories, vocabulary):
        """
        A python object for serializing the data needed for the AA data set
        experiments. The parameters are as follows:

        :param DT: a 1337x6226 sparse matrix containing the document-term counts
            of shape (nr_documents, nr_terms)
        :param TE: a 6226x9 sparse matrix containing the term-etymology assignments
            (nr_terms, nr_etymologies)
        :param etym_classes: a (9,) shaped array containing the names of etymology
            classes
        :param categories: a (1337,) shaped array containing the coarse-grained
            categories of the news items
        :param vocabulary: a (6226,) python list of 2-tuples. The first element of
            the tuple is the word itself, and the second element refers to the index
            of the term (i.e. to index the columns of DT)
        """
        self.DT = DT
        self.TE = TE
        self.etym_classes = etym_classes
        self.categories = categories
        self.vocabulary = vocabulary