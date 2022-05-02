import numpy as np

from cholesky_methods import bf_search, df_search
from utils import random_weighted_dag


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def dag_from_weighted_dfs(invcov):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrix A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    perm = df_search(invcov).permutation
    estimate = np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm

    return estimate


def recovered_weighted_dfs_nb_dag_count(n, N, spar):
    # counts the number of permutation matrices that the weighted_dag_from_dfs method successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    for i in range(N):
        U = random_weighted_dag(n, spar)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        print(A)
        invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
        eligible_mat = dag_from_weighted_dfs(invcov)

        if ((eligible_mat - A) < 1e-3).all():
            count += 1

        else:
            print('True DAG')
            print(A)
            print('-------------------------------------')
            print('Incorrect estimate')
            print(eligible_mat)

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' weighted DAGs'


def dags_from_weighted_bfs(invcov):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrices A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    permutations = bf_search(invcov)
    estimates = [np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm for perm in permutations]

    return close_to_avg(estimates)


def recovered_bfs_weighted_dag_count(n, N, spar):
    # counts the number of permutation matrices that dags_from_weighted_bfs successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    ambiguous_count = 0
    for i in range(N):
        U = random_weighted_dag(n, spar)  # generates a random weighted upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
        eligible_mats = dags_from_weighted_bfs(invcov)

        if len(eligible_mats) > 1:
            ambiguous_count += 1

        else:
            if (eligible_mats[0] == A).all():
                count += 1

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' DAGs and ' + str(
        ambiguous_count) + ' were ambiguous'


def close_to_avg(list_of_matrices, tol=1e-4):
    avg_mat = sum(list_of_matrices) / len(list_of_matrices)
    diffs = [np.linalg.norm(mat - avg_mat) for mat in list_of_matrices]
    if max(diffs) < tol:
        return avg_mat
    else:
        return list_of_matrices


n = 5
spar = .7
U = random_weighted_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
print('true DAG')
print(A)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
dags = dags_from_weighted_bfs(invcov)
if len(dags) == n:
    mat = dags
    print(mat)
    print(((mat - A) < 1e-4).all())
    print('---------------------------------------------')
else:
    print(False)
    print(dags)
