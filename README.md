# Required tools
- Python3
- matplotlib
- numpy
- sklearn
- scipy

# Dataset
The dataset we used can be downloaded from https://grouplens.org/datasets/movielens/

# cosine similiarity base on user/item
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
This part of code used to calculate the cosine similiarity of users. The code structure of item is really similar. Put two user dictionaries into the cosine_similarity function to get the score and the score will be record in dictionary like dict[user1][user2].

# euclidean distance base on user/item
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
This part of code used to calculate the euclidean distance of users. The code structure of item is really similar. Use the formula of euclidean distance to get the score and the score will be record in dictionary like dict[user1][user2].

# weighted euclidean distance base on user/item
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

# jaccard similiarity base on user/item
def jaccard_sim(user_dict1, user_dict2):
    # RSME1.1305770943451297
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    return len(set_and) / len(set_all)
This part of code used to calculate the jaccard similiarity of users.

We run each module to get the RSME of them to compare their performance. We use RSEM to measure the deviation between the PREDICTED value and the true value. The smaller RSME, the smaller the prediction error.
We also use recall and precision to evaluate the performance of the module.
  
