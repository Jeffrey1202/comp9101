# Introduction
The aim of the system is to recommend top 10 movies to user. We use CF&SVD algorithm and KNN algorithm to predict the moives that users may like and recommend them to users.

# Required tools
- Python3
- matplotlib
- numpy
- sklearn
- scipy

# Dataset
The dataset we used can be downloaded from https://grouplens.org/datasets/movielens/

# How to run
After downloading the dataset, unzip the file to get the directory ml-100k and then put ml-100k and *.py files in the same directory so that the system can be run.

# Python files
- User-based knn algorithm is in the 'user.py'.
- CF&SVD algorithm is in the 'recommander_system.py'.

# Main function
- get_list(user): input user id and return top 10 movies to users. The result is sorted by similarity scores. If the scores are same, the movies will be sorted by movieId.
- print_result(): used for generating figures and find the suitable k value.



  
