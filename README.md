# Required tools
- Python3
- matplotlib
- numpy
- sklearn
- scipy

# Dataset
The dataset we used can be downloaded from https://grouplens.org/datasets/movielens/

# Cosine similiarity base on user
def cos_sim(user_dict1, user_dict2):
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    if len(set_and) / len(set_all) <= Threshold:
        return 0
    d = [user_dict1, user_dict2]
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(d)
    score = cosine_similarity(x)[0][1]
    return score
This part of code used to calculate the cosine similiarity of users. Put two user dictionaries into the cosine_similarity function to get the score and the score will be record in dictionary like dict[user1][user2].

# Pearsonr similiarity base on user
def pearsonr_sim(user_dict1, user_dict2):
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    if len(set_and) / len(set_all) <= Threshold:
        return 0
    d = [user_dict1, user_dict2]
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(d)
    score = pearsonr(x[0], x[1])[0]
    return score
This part of code used to calculate the pearsonr similiarity of users. Put two user dictionaries into the cosine_similarity function to get the score and the score will be record in dictionary like dict[user1][user2].

# Euclidean distance base on user
def euclidean_sim(user_dict1, user_dict2):
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    if len(set_and) / len(set_all) <= Threshold:
        return 0
    d = [user_dict1, user_dict2]
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(d)
    sum = 0
    for a, b in zip(x[0], x[1]):
        sum += (a - b) ** 2
    score = 1 / (1 + np.sqrt(sum))
    return score
This part of code used to calculate the euclidean distance of users. Use the formula of euclidean distance to get the score and the score will be record in dictionary like dict[user1][user2].

# Weighted euclidean distance base on user
def weighted_euclidean_sim(user_dict1, user_dict2):
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    if len(set_and) / len(set_all) <= Threshold:
        return 0
    d = [user_dict1, user_dict2]
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(d)
    sum = 0
    for a, b in zip(x[0], x[1]):
        avg = (a - b) / 2
        si = ((a - avg) ** 2 + (b - avg) ** 2) ** 0.5
        sum += ((a - b) / si) ** 2
    score = 1 / (1 + np.sqrt(sum))
    return score
This part of code used to calculate the weighted euclidean distance of users. It uses an improved formula to caculate the distance which may improve the result.

# Jaccard similiarity base on user
def jaccard_sim(user_dict1, user_dict2):
    # RSME1.1305770943451297
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    return len(set_and) / len(set_all)
This part of code used to calculate the jaccard similiarity of users.

# Part of SVD algorithm
def SVD_computation(k,training_matrix,testing_matrix):
    matrixCopy= training_matrix.copy()
    tm = training_matrix.values
    tm_mean=np.nanmean(tm,axis=0,keepdims=True)
    tm=tm-tm_mean
    tm[np.isnan(tm)]=0
    ts = csc_matrix(tm).asfptype()
    u, s, vt = singular_value_decomposition(tm,k)
    X_pred = np.around(np.dot(np.dot(u, s), vt)+tm_mean)
    nz=testing_matrix.values.nonzero()
    tv = testing_matrix.values[nz[0],nz[1]]
    pv = X_pred[nz[0],nz[1]]
    mse=((pv-tv) ** 2).mean(axis=0)
    nnz=matrixCopy.values.nonzero()
    ttv = training_matrix.values[nnz]
    ppv=X_pred[nnz[0],nnz[1]]
    mmse=((ppv-ttv) ** 2).mean(axis=0)
    train_rmse = round(sqrt(mmse.real),3)
    RMSE = round(sqrt(mse.real),3)
    return X_pred
This part of code used for SVD algorithm.

# Evaluation
We run each module to get the RSME of them to compare their performance. We use RSEM to measure the deviation between the PREDICTED value and the true value. The smaller RSME, the smaller the prediction error.

  
