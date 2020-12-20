import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

def neighborhoods(embeddings, neighborhood_size=20):
    n = len(embeddings)
    # find nearest neighbors:
    nbrs = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="ball_tree").fit(embeddings) #what is exactly ball-tree?
    # ball-tree is the algorithm for finding the nearest neighbors. Per default, it looks for neighbors with respect to
    # the euclidean norm (minkowski with p=2).
    distances, indices = nbrs.kneighbors(embeddings)
    return(indices)

