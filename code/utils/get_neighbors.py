from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np

def neighborhoods(embeddings, neighborhood_size=20):
    n = len(embeddings)
    # find nearest neighbors:
    nbrs = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="ball_tree").fit(embeddings) #what is exactly ball-tree?
    # ball-tree is the algorithm for finding the nearest neighbors. Per default, it looks for neighbors with respect to
    # the euclidean norm (minkowski with p=2).
    distances, indices = nbrs.kneighbors(embeddings)
    return(indices)

def cosine_neighborhoods(embeddings, neighborhood_size=20):
    # https://stackoverflow.com/a/34145444/10844553 for a super nice explanation
    # The neighborhood of k-NN with respect to the cosine similarity and to the euclidean distance are equivalent if
    # we normalize the input beforehand.

    # normalize scales with respect to columns (features). However, we want to scale each row. Therefore we employ transpose,
    # normalize, and transpose back. axis=0 means that we normalize feature-wise and not over the whole matrix.
    embeddings = np.transpose(normalize(np.transpose(embeddings), axis=0))

    n = len(embeddings)
    # find nearest neighbors:
    nbrs = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="ball_tree").fit(
        embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    return (indices)