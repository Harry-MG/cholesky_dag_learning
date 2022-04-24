import numpy as np

from utils import random_dag, child_matrix


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


def noisy_bf_search(invcov, pivots):
    # notes - introduce inexact pivots
    n = np.shape(invcov)[0]
    pivot_elements = list(np.diag(np.linalg.inv(pivots)))
    initial_children = [i for i in range(n) if True in [invcov[i, i] == piv for piv in pivots]]
    depth = 0
    node = Node(matrix=invcov, children=initial_children, permutation=np.eye(n))
    current_nodes = [node]
    while depth < n:
        new_current_nodes = []
        for node in current_nodes:
            if node.children:
                for index in node.children:
                    # update matrix attribute by gaussian elimination and update pivots for next matrix derived from
                    # current
                    children_copy = node.children.copy()
                    children_copy.remove(node.matrix[index, index])

                    matrix = ldl_child_matrix(node.matrix, index, depth)

                    this_perm = np.eye(n)
                    this_perm[[depth, index]] = this_perm[[index, depth]]
                    new_perm = this_perm @ node.permutation

                    node = Node(matrix=matrix, children=children_copy, permutation=new_perm)
                    new_current_nodes.append(node)
        current_nodes = new_current_nodes
        depth += 1
    return current_nodes


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
                            True in [(new_matrix[j, j] - pivot) < 1e-4 for pivot in pivots]]
            # new_pivots = [new_matrix[j] for j in new_children]

            # update the parent attribute to be the previous node but with the current matrix removed from its children
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node


def noisy_dag_from_dfs(invcov, pivots):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrix A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    perm = noisy_df_search(invcov, pivots).permutation
    estimate = np.eye(n) - perm.T @ ans.matrix @ perm

    return estimate


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
print('df')
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
