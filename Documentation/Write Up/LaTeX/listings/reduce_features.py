def reduce_features(train_features, test_features, rank=300):
    ''' Performs the same principle component analysis on given train and test features '''
    sparse_train_features = csr_matrix(train_features).asfptype()
    sparse_test_features = csr_matrix(test_features)

    _, _, v_transpose = svds(sparse_train_features, k=rank)

    truncated_v = v_transpose.transpose()

    return sparse_train_features.dot(truncated_v), sparse_test_features.dot(truncated_v)