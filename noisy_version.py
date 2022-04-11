import numpy as np

from utils import random_dag, child_matrix

n = 5
spar = .75
U = random_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
dag = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
noise_cov = .1*np.diag(np.random.rand(n))  # diagonal as in the setup in Uhler paper
true_invcov = (np.eye(n) - dag).T @ np.linalg.inv(noise_cov) @ (np.eye(n) - dag)


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


def dfs_search(A):  # note - rewrite this function with the Node class instead of list pairs
    # returns permutations P of A such that the Cholesky factors of PAP^T have ones on the diagonal. Uses a depth-first
    # search

    n = np.shape(A)[0]

    i = 0

    # use a copy version of A to avoid changing A while running the function
    A_copy = np.copy(A)

    # initialise [matrix, permutation] pair
    current = [[A_copy, np.eye(n)]]

    while i < n:

        new_current = []

        for pair in current:

            # get the indices of the rows that are in position i or greater and that have 1 on the diagonal
            eligible_rows = [j for j in range(i, n) if pair[0][j, j] == 1]

            # only proceed when B has at least one eligible row
            if eligible_rows:

                for ind in eligible_rows:

                    B_copy = np.copy(pair[0])

                    perm_copy = np.copy(pair[1])

                    # swap the rows to put desired row in pivot row
                    B_copy[[i, ind]] = B_copy[[ind, i]]

                    # swap the corresponding columns
                    B_copy[:, [i, ind]] = B_copy[:, [ind, i]]

                    # update the permutation matrix associated to the current tree
                    this_perm = np.eye(n)
                    this_perm[[i, ind]] = this_perm[[ind, i]]
                    perm_copy = this_perm @ perm_copy

                    # note that B[i, i] = 1 != 0 so it's never necessary to move on to the next row prematurely.
                    # gaussian elimination step
                    for j in range(i + 1, n):

                        f = B_copy[j, i] / B_copy[i, i]

                        B_copy[j, i] = 0

                        for k in range(i + 1, n):
                            B_copy[j, k] = B_copy[j, k] - B_copy[i, k] * f

                    # append B to current_matrices
                    new_current.append([B_copy, perm_copy])

        # move on to the next row
        i += 1

        # update current matrices and permutations
        current = new_current

    eligible_perms = [pair[1] for pair in current]

    return eligible_perms


class Node:
    def __init__(self, matrix, children, parent=None, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def dfs_noisy_search(invcov, noise):
    # notes - introduce inexact pivots
    n = np.shape(invcov)[0]
    pivot_elements = list(np.diag(np.linalg.inv(noise)))
    initial_children = [i for i in range(n) if True in [invcov[i, i] == piv for piv in pivot_elements]]
    depth = 0
    node = Node(matrix=invcov, children=pivot_elements, permutation=np.eye(n))
    current_nodes = [node]
    while depth < n:
        new_current_nodes = []
        for node in current_nodes:
            for pivot in node.children:
                # update matrix attribute by gaussian elimination and update pivots for next matrix derived from current
                children_copy = node.children.copy()
                children_copy.remove(node.matrix[pivot, pivot])

                matrix = child_matrix(node.matrix, pivot, depth)

                this_perm = np.eye(n)
                this_perm[[depth, pivot]] = this_perm[[pivot, depth]]
                new_perm = this_perm @ node.permutation
                
                node = Node(matrix=matrix, children=children_copy, permutation=new_perm)
                new_current_nodes.append(node)
        current_nodes = new_current_nodes
        depth += 1
    return node
