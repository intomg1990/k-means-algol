###########################################
### Import Packages, Modules and Inputs ###
###########################################

# scientific packages
import numpy as np
import matplotlib.pylab as plt

# initialization functions
from init_functions import *

# iteration functions
from iteration_functions import *

# hyperparameters and inputs of algorithm
from parameters_INput import *

##################################
### Initialize Data Structures ###
##################################

np.random.seed(1)
# creates synthetic data
data, generator_centroids = create_synthetic_data(n_dim, N_vectors, k_clusters, max_range)
# initializes random centroids
centroids = create_init_centroids(data, k_clusters)
initial_centroids = centroids.copy()

######################################
### Iteration of K-means Algorithm ###
######################################

for i in range(128):
    # partition vectors into clusters
    group_list = partition_vectors(data, centroids)
    # new centroid is calculated
    centroids = calculate_new_centroids(group_list, data, k_clusters)

print(set(group_list))

X = np.vstack(data[:])
GC = np.vstack(generator_centroids[:])
C = np.vstack(centroids[:])
print(X.shape)
plt.plot(X[:, 0], X[:, 1], "om")
plt.plot(C[:, 0], C[:, 1], "bs")
plt.plot(GC[:, 0], GC[:, 1], "kx")
plt.show()

