import numpy as np
import matplotlib as mpl

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
    """
    # project each data sample into 2-dimensional space
    data_points = feature_vecs[:, :2]
    n_clusters = len( np.unique(labels) )

    # generate color palette for marking different cluster with different color
    palette = [i for i in "bgrcmyk"]



def main():
    pass


if __name__ == '__main__':
    main()