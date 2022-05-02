import numpy as np


def nonzeros(v):
    n = 0
    for x in v:
        if x != 0:
            n += 1
    return n


def random_dag(dim, sparsity):
    # generates random upper triangular matrix with given upper triangular sparsity,representing the weighted
    # adjacency matrix of a DAG

    A = np.random.rand(dim, dim)

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    # make binary

    A[np.abs(A) > 0] = 1

    return A


def random_weighted_dag(dim, sparsity):
    # generates random  non-binary upper triangular matrix with given upper triangular sparsity, representing the
    # weighted adjacency matrix of a DAG

    A = np.random.rand(dim, dim)  # threshold the matrix entries in [.1, .9]

    A = A + (0.1 / (0.9 - 0.1)) * np.ones((dim, dim))

    A = A * (0.9 - 0.1)

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    return A


def remove_duplicate_matrices(list_of_matrices):
    # input: list of matrices as square numpy arrays of the same size, possibly with repeats
    # output: list of matrices with repeats removed
    n = np.shape(list_of_matrices[0])[0]
    matrices_as_vecs = [list(mat.flatten()) for mat in list_of_matrices]
    tupled_vecs = set(map(tuple, matrices_as_vecs))
    lst = list(map(list, tupled_vecs))

    return [np.array(arr).reshape(n, n) for arr in lst]


def child_matrix(matrix, ind, depth):
    # swaps rows and columns ind, depth and applies gaussian elimination
    n = np.shape(matrix)[0]
    i = depth
    copy = np.copy(matrix)

    # swap the rows to put desired row in pivot row
    copy[[i, ind]] = copy[[ind, i]]

    # swap the corresponding columns
    copy[:, [i, ind]] = copy[:, [ind, i]]

    # note that copy[i, i] != 0 so it's never necessary to move on to the next row prematurely.
    # gaussian elimination step
    for j in range(i + 1, n):

        f = copy[j, i] / copy[i, i]

        copy[j, i] = 0

        for k in range(i + 1, n):
            copy[j, k] = copy[j, k] - copy[i, k] * f

    return copy


def ldl(matrix):
    dim = np.shape(matrix)[0]
    copy = np.copy(matrix).astype(float)
    diag = np.zeros(dim).astype(float)
    for i in range(dim):

        d = copy[i, i]
        diag[i] = d

        copy[i] = copy[i] / d

        for j in range(i + 1, dim):
            copy[j] = copy[j] - copy[j, i] * copy[i]

        print(copy)

    diag = np.diag(diag)
    print(matrix)
    print(copy.T @ diag @ copy)
    return [copy, diag]


def ldl_child_matrix(matrix, ind, depth):
    # swaps rows and columns ind, depth and applies gaussian elimination
    dim = np.shape(matrix)[0]
    i = depth
    copy = np.copy(matrix).astype(float)

    # swap the rows to put desired row in pivot row
    copy[[i, ind]] = copy[[ind, i]]

    # swap the corresponding columns
    copy[:, [i, ind]] = copy[:, [ind, i]]

    d = copy[i, i]

    copy[i] = copy[i] / d

    for j in range(i + 1, dim):
        copy[j] = copy[j] - copy[j, i] * copy[i]

    return copy
