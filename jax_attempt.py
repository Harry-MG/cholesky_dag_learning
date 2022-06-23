import jax
import jax.numpy as jnp
import numpy as np

from utils import random_dag

n = 5
N = 1000

U = random_dag(n, 0.75)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ np.transpose(P)
A = jnp.array(A)

data_mat = np.zeros((n, N))
noise_cov = np.eye(n)
for i in range(N):
    noise = np.random.multivariate_normal(mean=np.zeros(n), cov=noise_cov)
    X = np.linalg.inv(np.eye(n) - A) @ noise
    data_mat[:, i] = X
cov_mat = np.cov(data_mat)
inv_cov = np.linalg.inv(cov_mat)
inv_cov = jnp.array(inv_cov)


def jax_score_fn(diagonal, invcov):
    # inputs
    # invcov: dim x dim jnp array
    # diagonal: 1 x dim jnp array
    # output
    # score: 1 dimensional jnp

    dim = diagonal.shape[0]
    dag = jnp.zeros((dim, dim))

    for t in range(dim):
        ind = jnp.argmin(jnp.diag(invcov * jnp.diag(diagonal)))
        dag[ind, :] = - invcov[ind, :] / invcov[ind, ind]
        dag[ind, ind] = 0
        invcov = invcov - jnp.outer(invcov[:, ind].T, invcov[ind, :]) / invcov[ind, ind]
        invcov[ind, ind] = 1e6

    score = jnp.linalg.norm(dag)

    return score


d = jnp.ones(n)
d_grad = jax.grad(jax_score_fn, argnums=0)(d, inv_cov)
