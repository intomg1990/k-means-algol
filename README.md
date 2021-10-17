# K-means Algorithm

This repository contains a K-means Algorithm written from scratch, using only NumPy as a library. 

The goal of the algorithm in to partition the input data into $k$-cluster, $k \in \N$. The function `create_synthetic_data` may be used to create synthetic data. The data are $n$-dimensional vectors that will be belong to one of the $k$ groups, which will be similar, mathematically speaking. To achieve this goal we define the data vectors $x_i \in \R^n$ with $i = \{1, 2, ..., N\}$ and the representative vectors, or centroids, that are $z_j$ with $j = \{1, 2, ..., k\}$. 
