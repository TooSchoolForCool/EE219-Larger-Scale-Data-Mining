import math

import numpy as np
import matplotlib as mpl

from sklearn.decomposition import PCA

# Use Agg backend for supporing no-display environment
mpl.use('Agg')

import matplotlib.pyplot as plt


def print_title(msg, length = 60):
    """

    Print out a title format

    Args:
        msg: title name
        length: length of '*'

    ***********************************
    * <Title Name>
    ***********************************
    """
    print('*' * length)
    print('* %s' % msg)
    print('*' * length)


def calc_mat_variance(mat):
    """calculate variance of a matrix

    Suppose a matrix A, the variance of A is the sum of the squares of
    its singular values. Inorder the get this, we calculate AA'. The sum
    of diagonal of AA' is the sum of the squares of its (matrix A) sigular
    values, which is the variance we want.

    Args:
        mat: A matrix, which could be a np.ndarray or a scipy sparse matrix
    """

    # if mat is not np.ndarray, then convert it into np.ndarray
    if type(mat).__module__ != np.__name__:
        mat = mat.toarray()

    mat_t = mat.transpose()
    mat_mat_t = mat.dot(mat_t)

    # sum up diagonal
    variance = 0.0
    for i in range(0, mat_mat_t.shape[0]):
        variance += mat_mat_t[i, i]

    return variance


def plot_cluster_result(feature_vecs, labels, title):
    """plot clustering result

    Project high-dimensional vector into 2-dimensional plane by selecting
    the most 2 siginicant feature components (the first 2 in LSI or NMF),
    and coloring each data sample based on the given label.

    Args:
        feature_vecs: A feature vector matrix (a np.ndarray with shape 
            (n_docs, n_features)), generated through LSI or NMF. each 
            row represents a data sample, each column represents a feature.
        labels: A list of labels, each label is an integer which represents
            a class id. The i-th row in feature_vecs is associated with the
            i-th label in the list.
        title: [string] title of the figure and saving file name
    """
    # project each data sample into 2-dimensional space
    
    data_points = PCA(n_components=2).fit_transform(feature_vecs)
    # calculate number of clusters
    n_clusters = len( np.unique(labels) )

    # generate color palette for marking different clusters with different color
    palette = [i for i in "rgbcmyk"]
    # generate marker list, for different clusters we adopt different markers
    markers = [i for i in "o<.^"]

    clusters_x = [[] for i in range(n_clusters)]
    clusters_y = [[] for i in range(n_clusters)]
    clusters = [0 for i in range(n_clusters)]

    # assign each data point to its predicted cluster
    for point, label in zip(data_points, labels):
        clusters_x[label].append(point[0])
        clusters_y[label].append(point[1])

    # Plot scatter figure
    for i in range(n_clusters):
        clusters[i] = plt.scatter(clusters_x[i], clusters_y[i], s=3,
            c=palette[i % len(palette)], marker=markers[i % len(markers)])

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(clusters, ('Cluster #' + str(i) for i in range(1, n_clusters + 1)), loc=1)
    plt.savefig(title + ".png", dpi=512)

    print("Figure is save at ./%s" % (title + ".png"))


def normalize(feature_vecs):
    """normalize feature vectors

    Normalizing features, s.t. each feature has unit variance.

    Args:
        feature_vecs: A feature vector matrix (a np.ndarray with shape 
            (n_docs, n_features)). Each row represents a data sample,
            each column represents a feature.
    """
    # calculate standard ariance on each column
    col_var = np.var(feature_vecs, axis=0)
    col_std_var = [math.sqrt(var) for var in col_var]
    
    # deep copy
    unit_var_features = np.array(feature_vecs)

    for i in range(0, unit_var_features.shape[1]):
        unit_var_features[:, i] = unit_var_features[:, i] / col_std_var[i]

    return unit_var_features


def log_transform(feature_vecs):
    """Apply logarithm to each feature

    A nonlinear transformation, apply logarithm on each feature

    Args:
        feature_vecs: A feature vector matrix (a np.ndarray with shape 
            (n_docs, n_features)). Each row represents a data sample,
            each column represents a feature.
    """
    # deep copy
    log_feature = np.array(feature_vecs)

    vectorizer = np.vectorize(lambda x : math.log(x + 1e-5, 2))
    
    log_feature = vectorizer(log_feature)

    return log_feature


def main():
    pass


if __name__ == '__main__':
    main()