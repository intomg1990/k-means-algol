import numpy as np

def create_synthetic_data(n_dim: int, N_vectors: int, k_clusters: int, max_range: int) -> list:
    """
    Make synthetic data to apply k-means algol. Create random vectors distributed normally
    around artificial centroids. The artificial centroids have random coordinates ranging 
    from -max_range to max_range.
    
    INput:
    n_dim      --> the dimension of the synthetic vectors
    N_vectors  --> the number of vectors to make
    k_clusters --> the number of artificial clusters from which will be created the vecs
    max_range  --> the maximum range of each artificial centroid coordinate

    OUTput:
    data      --> list of vectors to which we will apply k-means
    centroids --> list containing ARTIFICIAL centroids from each synthetic data was created
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
    """This function returns initial random centroids. The coordinates of the centroids
    are means of random selected vectors. 
    
    INput:
    data      --> list of vectors that will be clustered
    k_cluster --> the number of clusters

    OUTput:
    centroids --> the initial centroids to the algol iterate over  
    """
    # extracts the number of data points
    N_vectors = len(data)
    # assign a random label (cluster) to each data point
    group_list = list(np.random.randint(k_clusters, size=N_vectors))

    # extracts the dimension of data point
    n_dim = len(data[0])
    # list with centroids as numpy arrays 
    centroids = []

    # loop to create k_clusters centroids
    for k in range(k_clusters):
        # initialize random centroid to zero
        centroid = np.zeros(n_dim)
        # counts how many vectors belong to the kth centroid
        counter = 0

        # iterates over all data to check which points belongs to kth cluster
        for i in range(N_vectors):

            # if belongs to the kth cluster, then it sums to the average
            if group_list[i] == k:
                centroid += data[i]
                counter += 1

        # the centroid is the mean of the vectors assigned the kth label
        centroid = centroid / counter

        centroids.append(centroid)

    return centroids






    
