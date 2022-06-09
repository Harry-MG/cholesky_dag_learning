import numpy as np
import sklearn.covariance
import matplotlib.pyplot as plt
import time

from utils import child_matrix, sample_covariance, random_dag, random_weighted_dag


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def inexact_df_search(invcov, pivot_tol):  # note - could speed up by stopping early when the diagonal is all ones
    """
    depth-first search for a permutation P such that the Cholesky factor of P@invcov@P^T has ones on its diagonal

    Args:
        invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + I (noise matrix identity)
        pivot_tol (float): tolerance of pivoting

    Returns:
        node (Node class): Node class node such that node.permutation satisfies the above criteria
    """
    n = np.shape(invcov)[0]
    depth = 0  # need to track depth in tree as we need to complete n passes of the matrix
    initial_children = [j for j in range(depth, n) if abs(invcov[j, j] - 1) < pivot_tol]  # inexact pivoting
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
            parent_children = node.children[1:]
            node.children = parent_children
            new_matrix = child_matrix(node.matrix, index, depth)

            depth += 1

            new_children = [j for j in range(depth, n) if abs(new_matrix[j, j] - 1) < pivot_tol]

            # update the parent attribute to be the previous node but with the current matrix removed from its children
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node


n = 5
spar = .75
N = 100
U = random_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
# noise_cov = .1 * np.diag(np.random.rand(n))
noise_cov = np.eye(n)
true_invcov = (np.eye(n) - dag).T @ np.linalg.inv(noise_cov) @ (np.eye(n) - dag)
sample_cov = sample_covariance(dag, noise_cov, N)
sample_invcov = np.linalg.inv(sample_cov)
glasso_invcov = sklearn.covariance.graphical_lasso(sample_cov, alpha=0.05)[1]
print('true_invcov')
print(true_invcov)
print('sample_invcov')
print(sample_invcov)
print('GLASSO_invcov')
print(glasso_invcov)


def inexact_dag_est(invcov_est, pivot_tol):
    n = np.shape(invcov_est)[0]
    node_ans = inexact_df_search(invcov_est, pivot_tol)
    perm = node_ans.permutation

    return np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov_est @ perm.T).T @ perm


def tol_search(invcov_est, tol_init, step_size):
    tol = tol_init
    while True:
        try:
            inexact_df_search(invcov_est, tol)
            break
        except AttributeError:
            tol += step_size
    return tol


def tol_search_list(dim, samples, runs):
    tols = np.zeros(runs)
    for N in range(runs):
        U = random_dag(dim, 0.75)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(dim)
        P = np.eye(dim)
        P[list(range(dim))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        # noise_cov = .1 * np.diag(np.random.rand(n))
        noise_cov = np.eye(dim)
        sample_invcov = np.linalg.inv(sample_covariance(dag, noise_cov, samples))
        tols[N] = tol_search(sample_invcov, 0.01, 0.01)
    return(tols)


def glasso_tol_search_list(dim, samples, runs, sparse_reg):
    tols = np.zeros(runs)
    for N in range(runs):
        U = random_dag(dim, 0.75)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(dim)
        P = np.eye(dim)
        P[list(range(dim))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        # noise_cov = .1 * np.diag(np.random.rand(n))
        noise_cov = np.eye(dim)
        sample_cov = sample_covariance(dag, noise_cov, samples)
        glasso_invcov = sklearn.covariance.graphical_lasso(sample_cov, alpha=sparse_reg)[1]
        tols[N] = tol_search(glasso_invcov, 0.01, 0.01)
    return(tols)


tol_list = tol_search_list(5, 1000, 1000)
glasso_tol_list_1 = glasso_tol_search_list(5, 1000, 1000, 0.1)
glasso_tol_list_2 = glasso_tol_search_list(5, 1000, 1000, 0.05)
glasso_tol_list_3 = glasso_tol_search_list(5, 1000, 1000, 0.02)
plt.hist(tol_list, alpha=.7, label='non-sparse')
plt.hist(glasso_tol_list_1, alpha=.7, label='lambda=0.1')
plt.hist(glasso_tol_list_2, alpha=.7, label='lambda=0.05')
plt.hist(glasso_tol_list_3, alpha=.7, label='lambda=0.02')
plt.legend(loc='upper right')
plt.show()


tol_list = tol_search_list(dim=10, samples=1000, runs=1000)
plt.hist(tol_list)
plt.show()


sample_list = [10, 50, 100, 500, 1000, 5000]
max_tols = []
for n_samples in sample_list:
    tol_list = tol_search_list(dim=5, samples=n_samples, runs=1000)
    max_tols.append(max(tol_list))

plt.plot(sample_list, max_tols)
plt.show()


def SHD_hist(dim, nsamples, runs):
    SHD_list = []
    for n in range(runs):
        if n % 50 == 0:
            print(n)
        U = random_dag(dim, 0.75)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(dim)
        P = np.eye(dim)
        P[list(range(dim))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)

        S = sample_covariance(dag, np.eye(dim), nsamples)
        Sinv = np.linalg.inv(S)

        pivot_tol = tol_search(Sinv, 0.01, 0.01)

        dag_est = inexact_dag_est(Sinv, pivot_tol)

        SHD = np.linalg.norm((dag - dag_est), 1)/dim**2

        SHD_list.append(SHD)

    return SHD_list


dim = 5
nsamples = 1000
U = random_weighted_dag(dim, 0.75)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(dim)
P = np.eye(dim)
P[list(range(dim))] = P[list(rand_perm)]
dag = P @ U @ np.transpose(P)

S = sample_covariance(dag, np.eye(dim), nsamples)
Sinv = np.linalg.inv(S)

pivot_tol = tol_search(Sinv, 0.01, 0.01)

dag_est = inexact_dag_est(Sinv, pivot_tol)

print('true DAG')
print(dag)
print('DAG estimate')
print(dag_est)
print('SHD')
print(np.linalg.norm((dag - dag_est), 1)/dim**2)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

dims = [16]
runs_per_dim = 10
nsamples = 1000
SHDs = []
for dim in dims:
    print('dim = '+str(dim))
    dim_SHDs = []
    for n in range(runs_per_dim):
        U = random_dag(dim, 0.75)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(dim)
        P = np.eye(dim)
        P[list(range(dim))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)

        S = sample_covariance(dag, np.eye(dim), nsamples)
        Sinv = np.linalg.inv(S)

        t = time.time()

        pivot_tol = tol_search(Sinv, 0.01, 0.01)

        dag_est = inexact_dag_est(Sinv, pivot_tol)

        t_tot = time.time() - t

        print(n)
        print('time taken = '+str(t_tot))
        print('pivot_tol = '+str(pivot_tol))

        SHD = np.linalg.norm((dag - dag_est), 1) / dim ** 2

        dim_SHDs.append(SHD)

    SHDs.append(np.mean(dim_SHDs))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Compare np.cov and normalising the scatter matrix

def mod_tol_search_list(dim, samples, runs):
    tols = np.zeros(runs)
    for N in range(runs):
        U = random_dag(dim, 0.75)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(dim)
        P = np.eye(dim)
        P[list(range(dim))] = P[list(rand_perm)]
        dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        # noise_cov = .1 * np.diag(np.random.rand(n))
        noise_cov = np.eye(dim)
        sample_invcov = np.linalg.inv(((samples-1)/(samples-dim-1)) * sample_covariance(dag, noise_cov, samples))
        tols[N] = tol_search(sample_invcov, 0.01, 0.01)
    return(tols)


tol_list = tol_search_list(5, 1000, 100)
mod_tol_list = mod_tol_search_list(5, 1000, 100)
plt.hist(tol_list, alpha=.7, label='non-sparse')
plt.hist(mod_tol_list, alpha=.7, label='mod')
plt.legend(loc='upper right')
plt.show()
