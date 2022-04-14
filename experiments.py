import numpy as np
import time

from matplotlib import pyplot as plt

from methods import dags_from_bfs, dag_from_dfs
from utils import random_dag


def recovered_bfs_dag_count(n, N, spar):
    # counts the number of permutation matrices that dags_from_bfs successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    ambiguous_count = 0
    for i in range(N):
        U = random_dag(n, spar)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
        eligible_mats = dags_from_bfs(invcov)

        if len(eligible_mats) > 1:
            ambiguous_count += 1

        else:
            if (eligible_mats[0] == A).all():
                count += 1

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' DAGs and ' + str(
        ambiguous_count) + ' were ambiguous'


def recovered_ltr_dag_count(n, N, spar):
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
        A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
        eligible_mat = dag_from_dfs(invcov)

        if (eligible_mat == A).all():
            count += 1

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' DAGs'


def dfs_vs_bfs_speed(dims, N, spar):
    # compare speeds of dfs and bfs
    dfs_times = []
    bfs_times = []

    for dim in dims:
        dfs_dim_times = []
        bfs_dim_times = []

        for i in range(N):
            U = random_dag(dim, spar)  # generates a random upper triangular matrix A
            rand_perm = np.random.permutation(dim)
            P = np.eye(dim)
            P[list(range(dim))] = P[list(rand_perm)]
            A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
            invcov = np.transpose(np.eye(dim) - A) @ (np.eye(dim) - A)

            start1 = time.time()
            dag_from_dfs(invcov)
            end1 = time.time() - start1
            dfs_dim_times.append(end1)

            start2 = time.time()
            dags_from_bfs(invcov)
            end2 = time.time() - start2
            bfs_dim_times.append(end2)

        dfs_times.append(np.mean(dfs_dim_times))
        bfs_times.append(np.mean(bfs_dim_times))

    plt.plot(dims, dfs_times, 'r')
    plt.plot(dims, bfs_times, 'b')
    plt.show()


def speed_of_method(dims, N, spar, method):
    # compare speeds of method (dag_from_dfs or dags_from_bfs) over different dimensions
    times = []

    for dim in dims:
        dim_times = []

        for i in range(N):
            U = random_dag(dim, spar)  # generates a random upper triangular matrix A
            rand_perm = np.random.permutation(dim)
            P = np.eye(dim)
            P[list(range(dim))] = P[list(rand_perm)]
            A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
            invcov = np.transpose(np.eye(dim) - A) @ (np.eye(dim) - A)

            start1 = time.time()
            method(invcov)
            end1 = time.time() - start1
            dim_times.append(end1)

        times.append(np.mean(dim_times))

    plt.plot(dims, times)
    plt.xlabel('number of nodes')
    plt.ylabel('average runtime (s) over ' + str(N) + ' samples')
    plt.show()
    print(np.sum(times))
