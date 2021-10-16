import numpy as np

def create_synthetic_data(n_dim: int, N_vectors: int, k_clusters: int, max_range: int) -> list:
    """
    Make synthetic data to apply k-means algol.
    
    INput:
    n_dim      --> the dimension of the synthetic vectors
    N_vectors  --> the number of vectors to make
    k_clusters --> the number of artificial clusters from which will be created the vecs
    max_range  --> the range of the centroid coordinates

    OUTput:
    data --> list of vectors to which we will apply k-means
    """
    # list of arrays containing the artificial centroids
    centroids = [] 

    # sample the random centroids from uniform distribution from -max_range to max_range
    # loop over clusters to create k_clusters multi-dim centroids
    for i in range(k_clusters):
        centroids.append(max_range * (2 * np.random.random((n_dim,)) - 1))

    # list cointaning the data points (vectors)
    data = []
    for i in range(N_vectors):
        # sample to what cluster the random vector will be from
        cluster_number = np.random.randint(1, k_clusters + 1)
        # create the data from normal distribution
        vector = np.random.randn(n_dim) + centroids[cluster_number - 1]
        
        data.append(vector)

    return data

def create_init_centroids(data: list, k_clusters: int) -> list:
    """This function returns initial random centroids in the appropiate range for better 
    convergence of the algorithm.
    
    INput:
    data      --> list of vectors that will be clustered
    k_cluster --> the number of clusters

    OUTput:
    list --> the initial centroids to the algol iterate over.  
    """
    # stack vectors to get the maxima in each of their dimensions
    X = np.vstack(data[:])
    maxima = np.max(np.abs(X), axis=0)
    
    # extract the dim of the vectors to create centroids with same dim
    n_dim = len(data[0])
    
    # centroids sampled at random from continuos PDF with range maxima of each vecs' dimensions
    centroids = [maxima * (2 * np.random.random(n_dim) - 1) for _ in range(k_clusters)]

    return centroids
    
