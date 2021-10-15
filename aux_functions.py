import numpy as np

def create_sinthetic_data(n_dim: int, N_vector: int, n_cluster: int, max_range: int) -> list:
    """
    Make sinthetic data to apply k-means algol.
    
    IN:
    n_dim     --> the dimension of the sinthetic vectors
    N_vector  --> the number of vectors to make
    n_cluster --> the number of artificial clusters from which will be created the vecs
    max_range --> the range of the centroid coordinates

    OUT:
    data --> list of vectors to which we will apply k-means
    """
    # list of arrays containing the artificial centroids
    centroids = [] 

    # sample the random centroids from uniform distribution from -max_range to max_range
    # loop over clusters to create n_cluster multi-dim centroids
    for i in range(n_cluster):
        centroids.append(max_range * (2 * np.random.random((n_dim,)) - 1))

    # list cointaning the data points (vectors)
    data = []
    for i in range(N_vector):
        # sample to what cluster the random vector will be from
        cluster_number = np.random.randint(1, n_cluster + 1)
        # create the data from normal distribution
        vector = np.random.randn(n_dim) + centroids[cluster_number - 1]
        
        data.append(vector)

    return data
    
    