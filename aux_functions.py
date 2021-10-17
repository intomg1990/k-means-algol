import numpy as np

def create_synthetic_data(n_dim: int, N_vectors: int, k_clusters: int, max_range: int) -> list:
    """
    Make synthetic data to apply k-means algol. Create random vectors distributed normally
    around artificial centroids. The artificial centroids have coordinates ranging from 
    -max_range to max_range.
    
    INput:
    n_dim      --> the dimension of the synthetic vectors
    N_vectors  --> the number of vectors to make
    k_clusters --> the number of artificial clusters from which will be created the vecs
    max_range  --> the maximum range of each artificial centroid coordinate

    OUTput:
    data      --> list of vectors to which we will apply k-means
    centroids --> list containing artificial centroids from each synthetic data is created
    """
    # sample the random centroids from uniform PDF from -max_range to max_range
    centroids = [max_range * (2 * np.random.random((n_dim,)) - 1) for _ in range (k_clusters)]

    # list cointaning the data points (vectors)
    data = []
    for i in range(N_vectors):
        # sample to what cluster the random vector will be part of
        cluster_number = np.random.randint(1, k_clusters + 1)
        # create the data AROUND CENTROIDS from normal distribution
        vector = np.random.randn(n_dim) + centroids[cluster_number - 1]
        
        data.append(vector)

    return data, centroids

def create_init_centroids(data: list, k_clusters: int) -> list:
    """This function returns initial random centroids in the appropiate range for better 
    convergence of the algorithm. The coordinates of the centroids are sampled from uniform
    PDF ranging the maximum value of the data in each of its dimensions.
    
    INput:
    data      --> list of vectors that will be clustered
    k_cluster --> the number of clusters

    OUTput:
    centroids --> the initial centroids to the algol iterate over  
    """
    # stack vectors 
    X = np.vstack(data[:])
    # get the maxima in each of their dimensions
    maxima = np.max(np.abs(X), axis=0)
    
    # extract the dim of the vectors to create centroids with same dim
    n_dim = len(data[0])
    
    # centroids sampled at random from continuos PDF with range maxima of data in each dim
    centroids = [maxima * (2 * np.random.random(n_dim) - 1) for _ in range(k_clusters)]

    return centroids
    
