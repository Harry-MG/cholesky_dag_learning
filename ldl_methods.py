import numpy as np

from utils import remove_duplicate_matrices, ldl_child_matrix


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def noisy_df_search(invcov, pivots):
    """
    depth-first search for a permutation P such that the lower triangular LDL factor 'L' of P @ invcov@ P^T
    has ones on its diagonal

        Args:
            invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + Z (noise matrix arbitrary)

        Returns:
            node (Node class): Node class node such that node.permutation satisfies the above criteria
    """
    n = np.shape(invcov)[0]
    depth = 0  # need to track depth in tree as we need to complete n passes of the matrix
    initial_children = [i for i in range(n) if True in [invcov[i, i] == pivot for pivot in pivots]]
    # initial_pivots = [invcov[i] for i in initial_children]
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
            node.children = node.children[1:]
            new_matrix = ldl_child_matrix(node.matrix, index, depth)

            depth += 1

            new_children = [j for j in range(depth, n) if
                            True in [abs(new_matrix[j, j] - pivot) < 1e-4 for pivot in pivots]]
            # new_pivots = [new_matrix[j] for j in new_children]

            # update the parent attribute to be the previous node but with the current matrix removed from its children
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node


def noisy_bf_search(A, pivots):  # note - rewrite this function with the Node class instead of list pairs
    """
    breadth-first search for all permutations P such that the lower triangular LDL factor 'L' of P @ invcov@ P^T has
    ones on its diagonal

     Args:
         A (np.ndarray): true inverse covariance matrix from linear SEM (noise matrix identity)

     Returns:
         current (tuple): tuple [matrix, perm] such that perm satisfies the criteria above
     """

    n = np.shape(A)[0]

    i = 0

    # use a copy version of A to avoid changing A while running the function
    A_copy = np.copy(A)

    # initialise [matrix, permutation, used_pivots] pair
    current = [[A_copy, np.eye(n), []]]

    while i < n:

        new_current = []

        for pair in current:

            # get the indices of the rows that are in position i or greater and that have 1 on the diagonal
            eligible_rows = [j for j in range(i, n) if True in [abs(pair[0][j, j] - pivot) < 1e-4 for pivot in pivots]]

            # only proceed when B has at least one eligible row
            if eligible_rows:

                for ind in eligible_rows:
                    B_copy = np.copy(pair[0])

                    perm_copy = np.copy(pair[1])

                    path_copy = pair[2].copy()

                    # update the permutation matrix associated to the current tree
                    this_perm = np.eye(n)
                    this_perm[[i, ind]] = this_perm[[ind, i]]
                    perm_copy = this_perm @ perm_copy

                    # note that B[i, i] = 1 != 0 so it's never necessary to move on to the next row prematurely.
                    # gaussian elimination step
                    B_copy = ldl_child_matrix(B_copy, ind, i)

                    path_copy.append(pair[0][ind, ind])

                    # append B to current_matrices
                    new_current.append([B_copy, perm_copy, path_copy])

        # move on to the next row
        i += 1

        # update current matrices and permutations
        current = new_current

    return current


def noisy_dag_from_dfs(invcov, pivots):
    """
    DAG estimate from inverse covariance matrix using df_search

    Args:
        invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + Z (noise matrix arbitrary)

    Returns:
        estimate (np.ndarray): DAG estimate
    """
    n = np.shape(invcov)[0]
    ans = noisy_df_search(invcov, pivots)
    perm = ans.permutation
    estimate = np.eye(n) - perm.T @ ans.matrix @ perm

    return estimate


def noisy_dags_from_bfs(invcov, pivots):
    """
    DAG estimate(s) from inverse covariance matrix using bf_search

    Args:
        invcov (np.ndarray): true inverse covariance matrix from linear SEM X = AX + Z (noise matrix arbitrary)

    Returns:
        (list): list of possible DAG estimates from the search
    """
    n = np.shape(invcov)[0]
    pairs = noisy_bf_search(invcov, pivots)

    estimates = [np.eye(n) - pair[1].T @ pair[0] @ pair[1] for pair in pairs]

    return remove_duplicate_matrices(estimates)
