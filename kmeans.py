import numpy as np

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : array_like
        The dataset of size (m, n) where each row is a single example.
        That is, we have m examples each of n dimensions.

    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the the data dimension.

    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset X.

    Instructions
    ------------
    Go over every example, find its closest centroid, and store
    the index inside `idx` at the appropriate location.
    Concretely, idx[i] should contain the index of the centroid
    closest to example i. Hence, it should be a value in the
    range 0..K-1

    Note
    ----
    Compute the distance to find the corresponding centroids for each data.
    Tips: It is possible to encounter 0 when computing the distance.
          Remember that you can add a very small number to the np.sqrt
    """
    # Set K
    K = centroids.shape[0]
    #idx = np.zeros((X.shape[0],1))
    #l = np.zeros((centroids.shape[0],1))
    
    
    # You need to return idx correctly.
    # min ||Xi - mu||^2
    #  k
    # ====================== YOUR CODE HERE ======================

    
    
#     for i in range(X.shape[0]):
#         for j in range(K):
#             d1 = X[i,:] - centroids[j,:]
#             d2 = np.sum(d1**2)
#             l[j] = d2
#         idx[i] = np.argmin(l)+1

#     return idx  
           
    idx = np.zeros((X.shape[0],1))
    for i in range(idx.shape[0]):
        dist_centroids = np.zeros((K,1))
        for j in range(0,K):
            dist_centroids[j] = np.sum(np.square((X[i,:] - centroids[j,:])))
        idx[i] = np.argmin(dist_centroids)
    idx = idx.astype(int)
    idx = np.squeeze(idx)                    
                   
    # =============================================================
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.

    Parameters
    ----------
    X : array_like
        The datset where each row is a single data point. That is, it
        is a matrix of size (m, n) where there are m datapoints each
        having n dimensions.

    idx : array_like
        A vector (size m) of centroid assignments (i.e. each entry in range [0 ... K-1])
        for each example.

    K : int
        Number of clusters

    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data
        points assigned to it.

    Instructions
    ------------
    Go over every centroid and compute mean of all points that
    belong to it. Concretely, the row vector centroids[i, :]
    should contain the mean of the data points assigned to
    cluster i.

    Note:
    -----
    Compute the new centroids.
    """
    # Useful variables
    m, n = X.shape
    centroids = np.zeros((K, n))
    Ck = np.zeros((K,1))
    
    # You need to return centroids correctly.
    # ====================== YOUR CODE HERE ======================
#     for i in range(m):
#         ci = int((idx[i]-1)[0])
#         centroids[ci,:]+=X[i,:]
#         ck[ci]+=1

    for i in range(m):
        index = idx[i]
        centroids[index] = centroids[index] + X[i]
        #Ck[index] = Ck[index] + 1
    for i in range(K):
        centroids[i] /=np.sum(idx == i)
    #centroids = centroids/Ck
        
    # =============================================================
    return centroids

def init_kmeans_centroids(X, K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x.

    Parameters
    ----------
    X : array_like
        The dataset of size (m x n).

    K : int
        The number of clusters.

    Returns
    -------
    centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).

    Instructions
    ------------
    You should set centroids to randomly chosen examples from the dataset X.
    """

    # You should return centroids correctly
    # ====================== YOUR CODE HERE ======================
    m,n = X.shape
    centroids = np.zeros((K,n))
    
    for i in range(K):
        centroids[i] = X[np.random.randint(0,m+1),:]
    # =============================================================
    return centroids
