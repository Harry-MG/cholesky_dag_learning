import numpy as np
import torch as torch
from utils import random_dag, random_weighted_dag, sample_covariance


def ghoshal_chol(invcov, diagonal):
    # note - diagonal is a vector here not a matrix
    dim = np.shape(invcov)[0]
    dag = np.zeros((dim, dim))

    for t in range(dim):
        ind = np.argmin(np.diag(invcov * np.diag(diagonal)))
        dag[ind, :] = - invcov[ind, :] / invcov[ind, ind]
        dag[ind, ind] = 0
        invcov = invcov - np.outer(invcov[:, ind].T, invcov[ind, :]) / invcov[ind, ind]
        invcov[ind, ind] = 1e6

    return dag


def torch_ghoshal_chol(invcov, diagonal):
    # note - diagonal is a vector here not a matrix
    dim = invcov.shape[0]
    dag = torch.zeros((dim, dim))

    for t in range(dim):
        ind = torch.argmin(torch.diag(invcov * torch.diag(diagonal)))
        dag[ind, :] = - invcov[ind, :] / invcov[ind, ind]
        dag[ind, ind] = 0
        invcov = invcov - torch.outer(invcov[:, ind].T, invcov[ind, :]) / invcov[ind, ind]
        invcov[ind, ind] = 1e6

    return dag


d = torch.ones(5)
d.requires_grad = True


def tgc(dia):
    return torch_ghoshal_chol(inv_cov, dia)


g = tgc(d)
print(g.requires_grad)
h = data_mat - g @ data_mat.float()
print(h.requires_grad)
p = torch.linalg.norm(h)
print(p.requires_grad)
print(torch.autograd.grad(p, d))

n = 5
U = random_dag(n, .65)
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ P.T
D = np.array([0.5, 0.3, 0.9, 0.8, 1])
prec = (np.eye(n) - A).T @ np.linalg.inv(np.diag(D)) @ (np.eye(n) - A)
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


def SEM_noise_score(data, invcov, diagonal):
    return torch.linalg.norm(data - torch_ghoshal_chol(invcov, diagonal) @ data.float())


def noise_search(data, invcov):
    max_iter = 1000
    step_size = 0.1
    n = invcov.shape[0]
    diagonal = torch.ones(n)
    diagonal.requires_grad = True

    def noise_score(diags):
        return SEM_noise_score(data, invcov, diags)

    for j in range(max_iter):
        phi = noise_score(diagonal)
        phi.requires_grad = True
        diagonal = diagonal - step_size * torch.autograd.grad(phi, diagonal)[0]

    return diagonal


''''''''''''''''''''''''''''''''''''''''''

d = torch.ones(5)
d.requires_grad = True


def noise_score(dia):
    return SEM_noise_score(data_mat, inv_cov, dia)


p = noise_score(d)
grad = torch.autograd.grad(p, d)


def noise_search2(data, invcov):
    max_iter = 1000
    step_size = 0.1
    n = invcov.shape[0]
    diagonal = torch.ones(n)
    diagonal.requires_grad = True

    def tgc(dia):
        return torch_ghoshal_chol(invcov, dia)

    for j in range(max_iter):
        g = tgc(diagonal)
        g.requires_grad = True
        print(g.requires_grad)
        h = data - g @ data.float()
        print(h.requires_grad)
        phi = torch.linalg.norm(h)
        print(phi.requires_grad)
        diagonal = diagonal - step_size * torch.autograd.grad(phi, diagonal)[0]

    return diagonal


def test_fn(d):
    mat = torch.diag(d)
    norm = torch.linalg.norm(mat)
    return norm


d = torch.ones(5)
d.requires_grad = True

out = test_fn(d)
grad = torch.autograd.grad(out, d)


def tf(di):
    n = di.shape[0]
    mat = torch_ghoshal_chol(inv_cov, di)
    norm = torch.linalg.norm(torch.eye(n) - mat)
    return norm


d = torch.ones(5)
d.requires_grad = True


out = tf(d)
out.retain_grad()
grad = torch.autograd.grad(out, d)

mat = torch_ghoshal_chol(inv_cov, d)
n = mat[2, 3]
grad1 = torch.autograd.grad(n, d)
