# scientific packages
import numpy as np
import matplotlib.pylab as plt

# auxiliary functions
from aux_functions import *

# iteration functions
from iteration_functions import *

# hyperparameters and inputs of algorithm
from parameters_INput import *

# creates synthetic data
data, generator_centroids = create_synthetic_data(n_dim, N_vectors, k_clusters, max_range)
# initializes random centroids
centroids = create_init_centroids(data, k_clusters)

# partition vectors into clusters
print(partition_vectors(data, centroids))

# X = np.vstack(data[:])
# C = np.vstack(generator_centroids[:])
# print(X.shape)
# plt.plot(X[:, 0], X[:, 1], "om")
# plt.plot(C[:, 0], C[:, 1], "ks")
# plt.show()

