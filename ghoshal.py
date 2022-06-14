import numpy as np

from utils import random_dag


def ghoshal_chol(invcov, diagonal):

    dim = np.shape(invcov)[0]
    dag = np.zeros((dim, dim))
    ind_list = np.zeros(dim)

    for t in range(dim):
        ind = np.argmin(np.diag(invcov * diagonal))
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
prec = (np.eye(n) - A).T @ (np.eye(n) - A)
D = np.eye(n)
print(A)
print(ghoshal_chol(prec, D))
