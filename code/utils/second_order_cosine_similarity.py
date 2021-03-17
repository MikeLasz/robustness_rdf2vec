from utils.get_neighbors import cosine_neighborhoods
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm

def pw_sec_cosine_similarity(emb1, emb2):
    nb1 = cosine_neighborhoods(emb1)
    nb2 = cosine_neighborhoods(emb2)
    n = len(emb1)
    s = []
    for node in tqdm.trange(n):
        nb_union = np.union1d(nb1[node], nb2[node]) # nb_union contains the indices of the nearest nodes
        z_emb1 = emb1[node].reshape(1, -1) # 2D array with only 1 sample [[....]]
        z_emb2 = emb2[node].reshape(1, -1) # 2D array with only 1 sample [[....]]
        s_emb1 = []
        s_emb2 = []
        for nb in nb_union:
            s_emb1.append(cosine_similarity(z_emb1, emb2[nb].reshape(1, -1) )) # cosine_similarity can only handle 2D arrays
            s_emb2.append(cosine_similarity(z_emb2, emb1[nb].reshape(1, -1)))  # hence, we need to reshape them to 2D arrays with 1 sample

        # recast s_emb to numpy arrays
        s_emb1 = np.array(s_emb1).reshape(1, -1)
        s_emb2 = np.array(s_emb2).reshape(1, -1)
        s.append(cosine_similarity(s_emb1, s_emb2))
    return(np.mean(s), s)

def sec_cosine_similarity(ls_embeddings):
    num_embeddings = len(ls_embeddings)
    scores = []
    num_max_iterations = int(num_embeddings * (num_embeddings - 1) / 2)
    iteration = 0
    for j in range(num_embeddings - 1):
        for i in range(j + 1, num_embeddings):
            emb1 = ls_embeddings[j]
            emb2 = ls_embeddings[i]
            print("Computing pairwise cosine similarity: {}/{}".format(iteration, num_max_iterations))
            pw_score, _ = pw_sec_cosine_similarity(emb1, emb2)
            scores.append(pw_score)
            iteration += 1
    return(np.mean(scores), scores)

# Tests
