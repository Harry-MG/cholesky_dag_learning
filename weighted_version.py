import numpy as np


def random_nb_dag(dim, sparsity):
    # generates random  non-binary upper triangular matrix with given upper triangular sparsity, representing the
    # weighted adjacency matrix of a DAG

    A = np.random.rand(dim, dim)  # might want to range the matrix entries in [.1, .9] for example

    A = A + (0.1 / (0.9 - 0.1)) * np.ones((dim, dim))

    A = A * (0.9 - 0.1)

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    return A


def weighted_bf_search(A):
    # returns permutations P of A such that the Cholesky factors of PAP^T have ones on the diagonal. Uses a
    # breadth-first search

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


def child_matrix(matrix, ind, depth):
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


class Node:
    def __init__(self, matrix, children, parent, permutation=None):
        self.matrix = matrix
        self.children = children
        self.parent = parent
        self.permutation = permutation


def weighted_df_search(invcov):
    n = np.shape(invcov)[0]
    depth = 0  # need to track depth in tree as we need to complete n passes of the matrix
    initial_children = [j for j in range(n) if invcov[j, j] == 1]
    # initialise node instance
    node = Node(matrix=invcov, children=initial_children, parent=None, permutation=np.eye(n))
    while depth < n - 1:  # and np.sum(np.diag(node.matrix)) > n + 1e-4:  # use > n because the diag elements are > 1
        # unless...
        # print(node.matrix)
        # print('depth = ' + str(depth))
        # print('children = ' + str(node.children))
        # print(node.matrix[depth, depth])

        # if np.abs(node.matrix[depth, depth] - 1) < 1e-4:
        # pass to next depth
        # depth += 1
        # node.children = node.children[1:]
        # print('pass through')

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

            new_children = [j for j in range(depth, n) if np.abs(new_matrix[j, j] - 1) < 1e-4]

            # update the parent attribute
            new_parent = node

            node = Node(matrix=new_matrix, children=new_children, parent=new_parent, permutation=new_perm)

        else:
            # go back to previous node and decrease depth by 1
            node = node.parent
            depth -= 1
    return node


n = 5
spar = .7
U = random_nb_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)


def dag_from_weighted_dfs(invcov):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrix A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    perm = weighted_df_search(invcov).permutation
    estimate = np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm

    return estimate


def recovered_weighted_dfs_nb_dag_count(n, N, spar):
    # counts the number of permutation matrices that the weighted_dag_from_dfs method successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    for i in range(N):
        U = random_nb_dag(n, spar)  # generates a random upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        print(A)
        invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
        eligible_mat = dag_from_weighted_dfs(invcov)

        if ((eligible_mat - A) < 1e-3).all():
            count += 1

        else:
            print('True DAG')
            print(A)
            print('-------------------------------------')
            print('Incorrect estimate')
            print(eligible_mat)

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' weighted DAGs'


def dags_from_weighted_bfs(invcov):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrices A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    permutations = weighted_bf_search(invcov)
    estimates = [np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm for perm in permutations]

    return close_to_avg(estimates)


def recovered_bfs_weighted_dag_count(n, N, spar):
    # counts the number of permutation matrices that dags_from_weighted_bfs successfully recovers
    # arguments:
    # n: dimension of the DAGs considered (number of nodes)
    # N: number of samples
    # spar: sparsity of the upper triangular part of the upper triangular DAG adjacency matrix
    count = 0
    ambiguous_count = 0
    for i in range(N):
        U = random_nb_dag(n, spar)  # generates a random weighted upper triangular matrix A
        rand_perm = np.random.permutation(n)
        P = np.eye(n)
        P[list(range(n))] = P[list(rand_perm)]
        A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
        invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
        eligible_mats = dags_from_weighted_bfs(invcov)

        if len(eligible_mats) > 1:
            ambiguous_count += 1

        else:
            if (eligible_mats[0] == A).all():
                count += 1

    return 'successfully recovered ' + str(count) + ' out of ' + str(N) + ' DAGs and ' + str(
        ambiguous_count) + ' were ambiguous'


def close_to_avg(list_of_matrices, tol=1e-4):
    avg_mat = sum(list_of_matrices) / len(list_of_matrices)
    diffs = [np.linalg.norm(mat - avg_mat) for mat in list_of_matrices]
    if max(diffs) < tol:
        return avg_mat
    else:
        return list_of_matrices


n = 5
spar = .7
U = random_nb_dag(n, spar)  # generates a random upper triangular matrix A
rand_perm = np.random.permutation(n)
P = np.eye(n)
P[list(range(n))] = P[list(rand_perm)]
A = P @ U @ np.transpose(P)  # now A represents a DAG not necessarily in topological order
print('true DAG')
print(A)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
invcov = np.transpose(np.eye(n) - A) @ (np.eye(n) - A)
dags = dags_from_weighted_bfs(invcov)
if len(dags) == n:
    mat = dags
    print(mat)
    print(((mat - A) < 1e-4).all())
    print('---------------------------------------------')
else:
    print(False)
    print(dags)
