import numpy as np


def nonzeros(v):
    n = 0
    for x in v:
        if x != 0:
            n += 1
    return n


def random_dag(dim, sparsity):
    # generates random upper triangular matrix with given upper triangular sparsity,representing the weighted
    # adjacency matrix of a DAG

    A = np.random.rand(dim, dim)

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    # make binary

    A[np.abs(A) > 0] = 1

    return A


def bf_search(A):  # note - rewrite this function with the Node class instead of list pairs
    # returns permutations P of A such that the Cholesky factors of PAP^T have ones on the diagonal. Uses a
    # breadth-first search

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


def remove_duplicate_matrices(list_of_matrices):
    # input: list of matrices as square numpy arrays of the same size, possibly with repeats
    # output: list of matrices with repeats removed
    n = np.shape(list_of_matrices[0])[0]
    matrices_as_vecs = [list(mat.flatten()) for mat in list_of_matrices]
    tupled_vecs = set(map(tuple, matrices_as_vecs))
    lst = list(map(list, tupled_vecs))

    return [np.array(arr).reshape(n, n) for arr in lst]


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def child_matrix(matrix, ind, depth):
    # swaps rows and columns ind, depth and applies gaussian elimination
    n = np.shape(matrix)[0]
    i = depth
    copy = np.copy(matrix)

    # swap the rows to put desired row in pivot row
    copy[[i, ind]] = copy[[ind, i]]

    # swap the corresponding columns
    copy[:, [i, ind]] = copy[:, [ind, i]]

    # note that copy[i, i] = 1 != 0 so it's never necessary to move on to the next row prematurely.
    # gaussian elimination step
    for j in range(i + 1, n):

        f = copy[j, i] / copy[i, i]

        copy[j, i] = 0

        for k in range(i + 1, n):
            copy[j, k] = copy[j, k] - copy[i, k] * f

    return copy


def df_search(invcov):  # note - could speed up by stopping early when the diagonal is all ones
    # returns Node class whose permutation attribute P such that the Cholesky factor of P@invcov@P.T has ones on its
    # diagonal. Uses a depth-first search.
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

            new_children = [j for j in range(depth, n) if new_matrix[j, j] == 1]

            # update the parent attribute to be the previous node but with the current matrix removed from its children
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node
