from utils.get_neighbors import cosine_neighborhoods
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pw_sec_cosine_similarity(emb1, emb2):
    nb1 = cosine_neighborhoods(emb1)
    nb2 = cosine_neighborhoods(emb2)
    n = len(emb1)
    s = []
    for node in range(n):
        nb_union = nb1[node] | nb2[node] # nb_union contains the indices of the nearest nodes
        z_emb1 = emb1[node, :]
        z_emb2 = emb2[node, :]
        s_emb1 = []
        s_emb2 = []
        for nb in nb_union:
            s_emb1.append(cosine_similarity(emb1[node], emb2[nb]))
            s_emb2.append(cosine_similarity(emb2[node], emb1[nb]))

        # recast s_emb to numpy arrays
        s_emb1 = np.array(s_emb1).reshape(1, -1)
        s_emb2 = np.array(s_emb2).reshape(1, -1)
        s.append(cosine_similarity(s_emb1, s_emb2))
    return(np.mean(s), s)

# Tests
