# Introduction
This project is used for building a recommendation system of COMP9417. We have 4 team members for this project: Yixi,Wei Yi,Zheng, ZeYue,Di Yizheng,Ying.
The aim of the system is to recommend top 10 movies to user. We use CF&SVD algorithm and KNN algorithm to predict the moives that users may like and recommend them to users.

# Required tools
- Python3
- matplotlib
- numpy
- sklearn
- scipy

# Dataset
The dataset we used can be downloaded from https://grouplens.org/datasets/movielens/
For our report we use the smallest one(100k),
There are some big dataset which can be used for further extension.

# How to run
After downloading the dataset, unzip the file to get the directory ml-100k and then put ml-100k and *.py files in the same directory so that the system can be run.

# Python files
We split different python files for different algorithms.
- User-based knn algorithm is in the 'user.py'.
- Item-based knn algorithm is in the 'item.py'
- CF&SVD algorithm is in the 'recommander_system.py'.

# Main function
For KNN:
We use severals methods to compute the similarity of users/items.(i.e. Jaccard,Cos,Pearsonr)
Then we predict scores for movies and sort them.
- get_list(user): input user id and return top 10 movies to users. The result is sorted by similarity scores. If the scores are same, the movies will be sorted by movieId.
- print_result(): used for generating figures and find the suitable k value.
For CF&SVD:
-rawRatingDF(): used for filterdRating.
-userAndMovieFilterdRating(): built cleanedRating.
-further_rating_filter(): input the movie rating and output the predit rating.
