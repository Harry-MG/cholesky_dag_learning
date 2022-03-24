# sketch of noisy version
import numpy as np

from utils import random_dag

n = 10
spar = .75
U = random_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
noise_cov = ...


def sample_inverse_covariance(dag, noise_cov, N):
    n = np.shape(dag)[0]
    data = np.zeros((n, N))
    for i in range(N):
        noise = ...
        X = np.linalg.inv(np.eye(n) - dag) @ noise
        data[:, i] = X
    covmat = np.cov(data)
    invcov = np.linalg.inv(covmat)
