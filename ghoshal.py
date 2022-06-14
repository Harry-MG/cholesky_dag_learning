import numpy as np

from utils import random_dag, random_weighted_dag, sample_covariance


def ghoshal_chol(invcov, diagonal):
# note - diagonal is a vector here not a matrix
    dim = np.shape(invcov)[0]
    dag = np.zeros((dim, dim))
    ind_list = np.zeros(dim)

    for t in range(dim):
        ind = np.argmin(np.diag(invcov * np.diag(diagonal)))
        ind_list[t] = ind
        dag[ind, :] = - invcov[ind, :] / invcov[ind, ind]
        dag[ind, ind] = 0
        invcov = invcov - np.outer(invcov[:, ind].T, invcov[ind, :]) / invcov[ind, ind]
        invcov[ind, ind] = 1e6

    return dag


n = 5
U = random_dag(n, .65)
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ P.T
D = np.diag(np.array([0.5, 0.3, 0.9, 0.8, 1]))
prec = (np.eye(n) - A).T @ np.linalg.inv(D) @ (np.eye(n) - A)
D = np.eye(n)
print(A)
print(ghoshal_chol(prec, D))

''''''''''''''''''''''''''''''''''''''''''

n = 5
nsamples = 1000
U = random_dag(n, 0.75)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ np.transpose(P)

S = sample_covariance(A, np.eye(n), nsamples)
Sinv = np.linalg.inv(S)

print(ghoshal_chol(Sinv, np.eye(n)))
print(A)


def SEM_noise_score(data, invcov, diagonal):
    return np.linalg.norm(data - ghoshal_chol(invcov, diagonal) @ data)


diagonal = diagonal - t * torch.autograd.grad(phi, diagonal)
