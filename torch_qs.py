import numpy as np
import torch

from utils import sample_covariance, random_dag

n = 5
nsamples = 1000
U = random_dag(n, 0.75)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ np.transpose(P)

S = sample_covariance(A, np.eye(n), nsamples)
Sinv = np.linalg.inv(S)

N = 1000
n = 5
data_mat = np.zeros((n, N))
noise_cov = np.eye(n)
for i in range(N):
    noise = np.random.multivariate_normal(mean=np.zeros(n), cov=noise_cov)
    X = np.linalg.inv(np.eye(n) - A) @ noise
    data_mat[:, i] = X
data_mat = torch.from_numpy(data_mat)
cov_mat = np.cov(data_mat)
inv_cov = np.linalg.inv(cov_mat)
inv_cov = torch.from_numpy(inv_cov)


def score_fn(diagonal, invcov):
    # inputs:
    # invcov: dim x dim torch array
    # diagonal: 1 x dim torch array
    # output:
    # score: 1 dimensional tensor

    dim = diagonal.shape[0]
    dag = torch.zeros((dim, dim))

    for t in range(dim):
        ind = torch.argmin(torch.diag(invcov * torch.diag(diagonal)))
        dag[ind, :] = - invcov[ind, :] / invcov[ind, ind]
        dag[ind, ind] = 0
        invcov = invcov - torch.outer(invcov[:, ind].T, invcov[ind, :]) / invcov[ind, ind]
        invcov[ind, ind] = 1e6

    score = torch.linalg.norm(dag)

    return score


n = 5
d = torch.ones(n)
d.requires_grad = True
sc = score_fn(d, inv_cov)
