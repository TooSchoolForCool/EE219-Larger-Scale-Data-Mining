import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

import utils
import evaluate
from data import DataLoader
from feature import DataVectorizer
from kmeans import KMeans


# tester for task 1
def tester_1():
    docs = DataLoader(category="8_class", mode="all")
    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())

    utils.print_title("Task 1: Building TF-IDF Matrix")
    print("The dimension of the TF-IDF matrix is (%r, %r)" % (docs_tfidf.shape))


# tester for task 2
def tester_2():
    docs = DataLoader(category="8_class", mode="all")
    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)
    kmeans = KMeans(n_clusters=2)

    # renaming groud truth labels, since we treat 8 classes as only 2 classes
    ground_truth = [label / 4 for label in docs.get_labels()]

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())
    predicted_labels = kmeans.predict(docs_tfidf)

    utils.print_title("Task 2: K-means clustering with k = 2")
    evaluate.eval_report(ground_truth, predicted_labels)


# tester for task 3
def tester_3():
    docs = DataLoader(category="debug", mode="test")

    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)
    svd = TruncatedSVD(n_components=10, random_state=13)
    nmf = NMF(n_components=10, random_state=13)

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())
    lsi_docs_tfidf = svd.fit_transform(docs_tfidf)
    nmf_docs_tfidf = nmf.fit_transform(docs_tfidf)

    # start task 3 (part a)
    tester_3_a(docs_tfidf, lsi_docs_tfidf, nmf_docs_tfidf)

    # start task 3 (part b)
    tester_3_b(docs_tfidf, lsi_docs_tfidf, nmf_docs_tfidf)


# tester for task 3 part a
def tester_3_a(tfidf, lsi_tfidf, nmf_tfidf):
    tfidf_variance = utils.calc_mat_variance(tfidf)

    dimension = lsi_tfidf.shape[1]

    lsi_variances = [utils.calc_mat_variance(lsi_tfidf[:, :i]) for i in range(1, dimension + 1)]
    nmf_variances = [utils.calc_mat_variance(nmf_tfidf[:, :i]) for i in range(1, dimension + 1)]

    lsi_raitos = [var / tfidf_variance for var in lsi_variances]
    nmf_raitos = [var / tfidf_variance for var in nmf_variances]

    # create x-axis
    r = [i + 1 for i in range(0, dimension)]

    plt.plot(r, lsi_raitos, "r", r, nmf_raitos, "b")
    plt.legend(("LSI", "NMF"), loc=0)
    plt.title('[LSI & NMF] The ratio of variance the top %d principle components' % dimension)
    plt.xlabel("r")
    plt.ylabel("Ratio of Variance")
    plt.show()


# a list of function
tester_functions = {
    "1" : tester_1,
    "2" : tester_2,
    "3" : tester_3
}


# tester function booter
def startTester(task):
    if task not in tester_functions:
        print("Do NOT have task %s" % task)
        exit(1)

    tester_functions[task]()


def main():
    pass


if __name__ == '__main__':
    main()