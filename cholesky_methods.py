import numpy as np

from utils import remove_duplicate_matrices, child_matrix


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def df_search(invcov):  # note - could speed up by stopping early when the diagonal is all ones
    """
    depth-first search for a permutation P such that the Cholesky factor of P@invcov@P^T has ones on its diagonal

    Args:
        invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + I (noise matrix identity)

    Returns:
        node (Node class): Node class node such that node.permutation satisfies the above criteria
    """
    n = np.shape(invcov)[0]
    depth = 0  # need to track depth in tree as we need to complete n passes of the matrix
    initial_children = [j for j in range(depth, n) if invcov[j, j] == 1]
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

            new_children = [j for j in range(depth, n) if (new_matrix[j, j] - 1) < 1e-4]

            # update the parent attribute to be the previous node but with the current matrix removed from its children
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node


def bf_search(A):
    """
    breadth-first search for all permutations P such that the Cholesky factor of P@invcov@P^T has ones on its diagonal

    Args:
        A (np.ndarray): true inverse covariance matrix from linear SEM (noise matrix identity)

    Returns:
        eligible_perms (list): list of np.ndarrays P satisfying the criteria
    """

    n = np.shape(A)[0]

    i = 0

    # use a copy version of A to avoid changing A while running the function
    A_copy = np.copy(A)

    # perm = np.eye(n)

    current = [[A_copy, np.eye(n)]]

    while i < n:

        new_current = []

        for pair in current:

            # get the indices of the rows that are in position i or greater and that have 1 on the diagonal
            eligible_rows = [j for j in range(i, n) if np.abs(pair[0][j, j] - 1) < 1e-4]

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


def dags_from_bfs(invcov):
    """
    DAG estimate(s) from inverse covariance matrix using bf_search

    Args:
        invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + I (noise matrix identity)

    Returns:
        (list): list of possible DAG estimates from the search
    """
    n = np.shape(invcov)[0]
    permutations = bf_search(invcov)
    estimates = [np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm for perm in permutations]

    return remove_duplicate_matrices(estimates)


def dag_from_dfs(invcov):
    """
    DAG estimate from inverse covariance matrix using df_search

    Args:
        invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + I (noise matrix identity)

    Returns:
        estimate (np.ndarray): DAG estimate
    """
    n = np.shape(invcov)[0]
    perm = df_search(invcov).permutation
    estimate = np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm

    # ans = df_search(invcov)
    # perm = ans.permutation
    # estimate = np.eye(n) - perm.T @ ans.matrix @ perm

    return estimate
