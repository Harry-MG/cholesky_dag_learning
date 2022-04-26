import numpy as np

from utils import random_dag, child_matrix, remove_duplicate_matrices


def sample_inverse_covariance(dag, noise_cov, N):
    n = np.shape(dag)[0]
    data = np.zeros((n, N))
    for i in range(N):
        noise = np.random.multivariate_normal(mean=np.zeros(n), cov=noise_cov)
        X = np.linalg.inv(np.eye(n) - dag) @ noise
        data[:, i] = X
    covmat = np.cov(data)
    invcov = np.linalg.inv(covmat)
    return invcov


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


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


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


def noisy_bf_search(A, pivots):  # note - rewrite this function with the Node class instead of list pairs
    # returns permutations P of A such that the Cholesky factors of PAP^T have ones on the diagonal. Uses a
    # breadth-first search

    n = np.shape(A)[0]

    i = 0

    # use a copy version of A to avoid changing A while running the function
    A_copy = np.copy(A)

    # initialise [matrix, permutation, used_pivots] pair
    current = [[A_copy, np.eye(n), []]]

    while i < n:

        new_current = []

        for pair in current:

            # get the indices of the rows that are in position i or greater and that have 1 on the diagonal
            eligible_rows = [j for j in range(i, n) if True in [abs(pair[0][j, j] - pivot) < 1e-4 for pivot in pivots]]

            # only proceed when B has at least one eligible row
            if eligible_rows:

                for ind in eligible_rows:

                    B_copy = np.copy(pair[0])

                    perm_copy = np.copy(pair[1])

                    path_copy = pair[2].copy()

                    # update the permutation matrix associated to the current tree
                    this_perm = np.eye(n)
                    this_perm[[i, ind]] = this_perm[[ind, i]]
                    perm_copy = this_perm @ perm_copy

                    # note that B[i, i] = 1 != 0 so it's never necessary to move on to the next row prematurely.
                    # gaussian elimination step
                    B_copy = ldl_child_matrix(B_copy, ind, i)

                    path_copy.append(pair[0][ind, ind])

                    # append B to current_matrices
                    new_current.append([B_copy, perm_copy, path_copy])

        # move on to the next row
        i += 1

        # update current matrices and permutations
        current = new_current

    return current


def noisy_dag_from_dfs(invcov, pivots):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrix A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    perm = noisy_df_search(invcov, pivots).permutation
    estimate = np.eye(n) - perm.T @ ans.matrix @ perm

    return estimate


def noisy_dags_from_bfs(invcov, pivots):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrices A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    pairs = noisy_bf_search(invcov, pivots)
    for pair in pairs:
        print('est')
        print(np.eye(n) - pair[1].T @ pair[0] @ pair[1])
        print('path')
        print(pair[2])
    estimates = [np.eye(n) - pair[1].T @ pair[0] @ pair[1] for pair in pairs]

    return remove_duplicate_matrices(estimates)


n = 5
spar = .75
U = random_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
noise_cov = .1 * np.diag(np.random.rand(n))  # diagonal as in the setup in Uhler paper
true_invcov = (np.eye(n) - dag).T @ np.linalg.inv(noise_cov) @ (np.eye(n) - dag)
print('inverse covariance')
print(true_invcov)
pivots = 1 / np.diag(noise_cov)
print('pivots')
print(pivots)
print('DAG')
print(dag)

ans = noisy_dags_from_bfs(true_invcov, pivots)
print('estimates')
for est in ans:
    print(est)

ans = noisy_df_search(true_invcov, pivots)
perm = ans.permutation
est = np.eye(n) - perm.T @ ans.matrix @ perm
print('estimate')
print(est)


# check if it's a true LDL factorisation
# print(ans.matrix.T @ perm @ np.diag(pivots) @ perm.T @ ans.matrix)
# print(perm @ true_invcov @ perm.T)

def recovered_dfs_noisy_dag_count(n, N, spar):
    # counts the number of permutation matrices that the dag_from_dfs method successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    for i in range(N):
        U = random_dag(n, spar)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        noise_cov = .1 * np.diag(np.random.rand(n))  # diagonal as in the setup in Uhler paper
        invcov = (np.eye(n) - dag).T @ np.linalg.inv(noise_cov) @ (np.eye(n) - dag)
        pivots = 1 / np.diag(noise_cov)
        eligible_mat = noisy_dag_from_dfs(invcov, pivots)

        if ((eligible_mat - dag) < 1e-4).all():
            count += 1

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' DAGs'
