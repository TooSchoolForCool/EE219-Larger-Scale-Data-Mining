import numpy as np
import matplotlib as mpl

mpl.use('Agg')

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

    # renaming ground truth labels, since we treat 8 classes as only 2 classes
    ground_truth = [label / 4 for label in docs.get_labels()]

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())
    predicted_labels = kmeans.predict(docs_tfidf)

    utils.print_title("Task 2: K-means clustering with k = 2")
    evaluate.eval_report(ground_truth, predicted_labels)


# tester for task 3
def tester_3():
    docs = DataLoader(category="8_class", mode="all")

    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)
    svd = TruncatedSVD(n_components=1000, random_state=13)

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())
    lsi_docs_tfidf = svd.fit_transform(docs_tfidf)

    # renaming ground truth labels, since we treat 8 classes as only 2 classes
    ground_truth = [label / 4 for label in docs.get_labels()]

    # start task 3 (part a)
    utils.print_title("Task 3 part(a): Plot ratio of variance")
    tester_3_a(docs_tfidf, lsi_docs_tfidf)

    # start task 3 (part b)
    utils.print_title("Task 3 part(b): Testing param for LSI & NMF")
    tester_3_b(docs_tfidf, lsi_docs_tfidf, ground_truth)


# tester for task 3 part(a)
def tester_3_a(tfidf, lsi_tfidf):
    tfidf_variance = utils.calc_mat_variance(tfidf)

    dimension = lsi_tfidf.shape[1]
    lsi_variances = [utils.calc_mat_variance(lsi_tfidf[:, :i]) for i in range(1, dimension + 1)]
    lsi_raitos = [var / tfidf_variance for var in lsi_variances]

    # create x-axis
    r = [i + 1 for i in range(0, dimension)]
    
    plt.plot(r, lsi_raitos, "r")
    plt.legend(['SVD'])
    plt.title('[SVD] The ratio of variance the top %d principle components' % dimension)
    plt.xlabel("r")
    plt.ylabel("Ratio of Original Variance")
    plt.savefig('task_3_a.png')

    print("Figure is saved at ./task_3_a.png")


# tester for task 3 part(b)
def tester_3_b(tfidf, lsi_tfidf, ground_truth):
    kmeans = KMeans(n_clusters=2)
    testcases = [1, 2, 3, 5, 10, 20, 50, 100, 300]

    # test LSI
    for case in testcases:
        predicted_labels = kmeans.predict(lsi_tfidf[:, :case])
        evaluate.eval_report(ground_truth, predicted_labels, "-----[LSI] r = %d-----" % case)
        evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
            msg="Contingency Matrix")

    # test NMF
    for case in testcases:
        nmf_k_tfidf = NMF(n_components=case, random_state=13).fit_transform(tfidf)
        predicted_labels = kmeans.predict(nmf_k_tfidf)
        evaluate.eval_report(ground_truth, predicted_labels, "-----[NMF] r = %d-----" % case)
        evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
            msg="Contingency Matrix")


# tester for task 4
def tester_4():
    docs = DataLoader(category="8_class", mode="all")
    lsi_best_r, nmf_best_r = 20, 10

    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)
    svd = TruncatedSVD(n_components=lsi_best_r, random_state=13)
    nmf = NMF(n_components=nmf_best_r, random_state=13)

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())
    lsi_docs_tfidf = svd.fit_transform(docs_tfidf)
    nmf_docs_tfidf = nmf.fit_transform(docs_tfidf)

    # renaming ground truth labels, since we treat 8 classes as only 2 classes
    ground_truth = [label / 4 for label in docs.get_labels()]

    # task 4 part(a)
    utils.print_title("Task 4 part(a): Visulize Clustering Result")
    tester_4_a(lsi_docs_tfidf, "LSI with r = %d" % lsi_best_r)
    tester_4_a(nmf_docs_tfidf, "NMF with r = %d" % nmf_best_r)

    # task 4 part(b)
    utils.print_title("Task 4 part(b): Normalization & non-linear transformation")
    tester_4_b(lsi_docs_tfidf, ground_truth, nonlinear_transform=False, title="LSI with r = %d" % lsi_best_r)
    tester_4_b(nmf_docs_tfidf, ground_truth, nonlinear_transform=True, title="NMF with r = %d" % nmf_best_r)


# tester for task 4 part(a)
def tester_4_a(feature_vecs, title):
    kmeans = kmeans = KMeans(n_clusters=2)

    predicted_labels = kmeans.predict(feature_vecs)

    print("-----%s-----" % title)
    utils.plot_cluster_result(feature_vecs, predicted_labels, title)


# tester for task 4 part(b)
def tester_4_b(feature_vecs, ground_truth, nonlinear_transform=False, title=""):
    kmeans = KMeans(n_clusters=2)

    # original result
    # evaluate normalization
    predicted_labels = kmeans.predict(feature_vecs)
    evaluate.eval_report(ground_truth, predicted_labels, "-----Original %s-----" % title)
    evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
        msg="Contingency Matrix")
    utils.plot_cluster_result(feature_vecs, predicted_labels, "Original %s" % title)

    # evaluate normalization
    normalized_features = utils.normalize(feature_vecs)
    predicted_labels = kmeans.predict(normalized_features)
    evaluate.eval_report(ground_truth, predicted_labels, "-----Normalization %s-----" % title)
    evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
        msg="Contingency Matrix")
    utils.plot_cluster_result(normalized_features, predicted_labels, "Normalization %s" % title)

    if nonlinear_transform:
        # Non-linear transform (logarithm)
        log_features = utils.log_transform(feature_vecs)
        predicted_labels = kmeans.predict(log_features)
        evaluate.eval_report(ground_truth, predicted_labels, "-----Logarithm %s-----" % title)
        evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
            msg="Contingency Matrix")
        utils.plot_cluster_result(log_features, predicted_labels, "Logarithm %s" % title)

        # Logarithm -> normalization
        log_normalized_features = utils.normalize(log_features)
        predicted_labels = kmeans.predict(log_normalized_features)
        evaluate.eval_report(ground_truth, predicted_labels, "-----Logarithm + Normalization %s-----" % title)
        evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
            msg="Contingency Matrix")
        utils.plot_cluster_result(log_normalized_features, predicted_labels, "log_Normalize %s" % title)

        # normalization -> logarithm
        normalized_log_features = utils.log_transform(normalized_features)
        predicted_labels = kmeans.predict(normalized_log_features)
        evaluate.eval_report(ground_truth, predicted_labels, "-----Normalization + Logarithm %s-----" % title)
        evaluate.contingency_matrix(ground_truth, predicted_labels, n_clusters=2, 
            msg="Contingency Matrix")
        utils.plot_cluster_result(normalized_log_features, predicted_labels, "Normalize_log %s" % title)


# a list of function
tester_functions = {
    "1" : tester_1,
    "2" : tester_2,
    "3" : tester_3,
    "4" : tester_4
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