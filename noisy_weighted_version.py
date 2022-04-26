import numpy as np

from noisy_version import ldl_child_matrix, noisy_dag_from_dfs
from weighted_version import random_nb_dag


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def noisy_df_search(invcov, pivots):
    # returns Node class whose permutation attribute P such that the Cholesky factor of P@invcov@P.T has ones on its
    # diagonal. Uses a depth-first search.
    n = np.shape(invcov)[0]
    depth = 0  # need to track depth in tree as we need to complete n passes of the matrix
    initial_children = [i for i in range(n) if True in [invcov[i, i] == pivot for pivot in pivots]]
    # initial_pivots = [invcov[i] for i in initial_children]
    # initialise node instance with the inverse covariance matrix
    node = Node(matrix=invcov, children=initial_children, parent=None, permutation=np.eye(n))
    while depth < n:
        if node.children:
            # update children to exclude first child and pass to first child, increase depth by 1
            index = node.children[0]

            # update the permutation matrix
            this_perm = np.eye(n)
            this_perm[[depth, index]] = this_perm[[index, depth]]
            new_perm = this_perm @ node.permutation

            # update the matrix and children attributes
            node.children = node.children[1:]
            new_matrix = ldl_child_matrix(node.matrix, index, depth)

            depth += 1

            new_children = [j for j in range(depth, n) if
                            True in [abs(new_matrix[j, j] - pivot) < 1e-4 for pivot in pivots]]
            # new_pivots = [new_matrix[j] for j in new_children]

            # update the parent attribute to be the previous node but with the current matrix removed from its children
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node


def recovered_dfs_noisy_weighted_dag_count(n, N, spar):
    # counts the number of permutation matrices that the dag_from_dfs method successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    for i in range(N):
        U = random_nb_dag(n, spar)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        noise_cov = .1 * np.diag(np.random.rand(n))  # diagonal as in the setup in Uhler paper
        invcov = (np.eye(n) - dag).T @ np.linalg.inv(noise_cov) @ (np.eye(n) - dag)
        pivots = 1 / np.diag(noise_cov)
        eligible_mat = noisy_dag_from_dfs(invcov, pivots)

        if (abs(eligible_mat - dag) < 1e-4).all():
            count += 1

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' DAGs'
