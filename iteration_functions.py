import numpy as np

def partition_vectors(data: list, centroids: list) -> list:
    """This function assigns each vector of the data to a centroid.
    
    INput:
    data      --> list of vectors (data points) to which we will apply k-means
    centroids --> the centroids that will be paired with each vector

    OUTput:
    group_list --> list containing centroid labels associated with each data point (vector)
    """
    # the number of data points
    N_vectors = len(data)
    # the number of clusters
    k_clusters = len(centroids)
    # list pairing clusters and vectors
    group_list=[]

    # nested loop to calculated distance between data points (vectors) to centroids
    for i in range(N_vectors):
        # calculates first distance and save it as minimum
        min_dist = np.sqrt(np.sum((data[i] - centroids[0])**2))
        # assign/pair the ith vector to/with the cluster 0, but will be updated
        group_of_i = 0

        for j in range(1, k_clusters):
            # calculates the distance between the centroid and the data point (vector)
            euclides_dist = np.sqrt(np.sum((data[i] - centroids[j])**2))
            
            # saves the label of the cluster to which the vector belongs
            if (euclides_dist < min_dist):
                group_of_i = j

        group_list.append(group_of_i)
    
    return group_list


