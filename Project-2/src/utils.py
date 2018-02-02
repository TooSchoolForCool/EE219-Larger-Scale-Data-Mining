import numpy as np


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


def main():
    pass


if __name__ == '__main__':
    main()