# sketch of non-binary (dag non-binary) version
import numpy as np

from methods import dag_from_ltr


def random_nb_dag(dim, sparsity):
    # generates random  non-binary upper triangular matrix with given upper triangular sparsity, representing the
    # weighted adjacency matrix of a DAG

    A = np.random.rand(dim, dim)  # might want to range the matrix entries in [.2, .8] for example

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    return A

