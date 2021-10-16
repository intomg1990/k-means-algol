# scientific packages
import numpy as np
import matplotlib.pylab as plt

# auxiliary functions
from aux_functions import *

# hyperparameters and input of algorithm
from parameters_INput import *

# creates synthetic data
data = create_synthetic_data(n_dim, N_vectors, k_clusters, max_range)
# initializes random centroids
centroids = create_init_centroids(data, k_clusters)

# X = np.vstack(data[:])
# print(X.shape)
# plt.plot(X[:, 0], X[:, 1], "om")
# plt.show()

