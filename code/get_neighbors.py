import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

def neighborhoods(embeddings, num_samples=5, neighborhood_size=20):
    n = len(embeddings)
    # find nearest neighbors:
    nbrs = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="ball_tree").fit(embeddings) #what is exactly ball-tree?
    distances, indices = nbrs.kneighbors(embeddings)
    return(indices)

