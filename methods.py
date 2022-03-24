def dags_from_dfs(invcov):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrices A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    permutations = dfs_search(invcov)
    estimates = [np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm for perm in permutations]

    return remove_duplicate_matrices(estimates)


def dag_from_ltr(invcov):
    # input: inverse covariance matrix
    # output: estimated DAG adjacency matrix A such that invcov = (I-A)^T (I-A)
    n = np.shape(invcov)[0]
    perm = ltr_search(invcov).permutation
    estimate = np.eye(n) - perm.T @ np.linalg.cholesky(perm @ invcov @ perm.T).T @ perm

    return estimate
