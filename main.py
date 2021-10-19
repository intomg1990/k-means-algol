"""This programs was written to better understand k-means algorithms.
It generates synthetic data from `k_clusters_synth` random normal PDFs
and initializes random centroids. From there it iterates minimizing J cost
to clusterize the data into `k_clusters`."""

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

# the random seed, should be modified for fun
np.random.seed(4)
# creates synthetic data
data, generator_centroids = create_synthetic_data(n_dim, N_vectors, k_clusters_synth, max_range)
# initializes random centroids
centroids = create_init_centroids(data, k_clusters)
# initialize the variable the controls convergence
stop = False

######################################
### Iteration of K-means Algorithm ###
######################################

while(stop == False):
    # partition vectors into clusters
    group_list = partition_vectors(data, centroids)
    # new centroids are calculated
    new_centroids = calculate_new_centroids(group_list, data, k_clusters)

    # check for convergence
    if (np.linalg.norm(np.array(new_centroids) - np.array(centroids)) < 1e-6):
        stop = True
    else:
        centroids = new_centroids

########################
### Plotting Results ###
########################

# stack centroids and data vectors to plot them
X = np.vstack(data[:])
GC = np.vstack(generator_centroids[:])
C = np.vstack(centroids[:])
# lods plots in RAM
plt.plot(X[:, 0], X[:, 1], "om", label="Synthetic data")
plt.plot(C[:, 0], C[:, 1], "bs", label="Calculated centroids")
plt.plot(GC[:, 0], GC[:, 1], "kx", label="Generetor centroids")
# units of axes
plt.xlabel(r"$x_{(1)}$")
plt.ylabel(r"$x_{(2)}$")
# renders in-plot legend 
plt.legend()
# from RAM to screen
plt.show()

