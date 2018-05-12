def reduce_features(complete_train_features, complete_test_features):
    ''' Performs the same principle component
        analysis on given train and test features '''

    _, sigma, v_transpose = svd(complete_train_features, full_matrices=True,
                                compute_uv=True)

    rank = compute_rank(sigma)

    truncated_v = v_transpose[:rank].transpose()

    return matmul(complete_train_features, truncated_v), matmul(complete_test_features,
                                                                truncated_v)