import numpy as np
from get_neighbors import neighborhoods

def jaccard_of_sets(sets):
    num_sets = len(sets)
    union = intersection = sets[0]
    for j in range(num_sets):
        union = union | sets[j]
        intersection = intersection & sets[j]
    return(len(intersection) / len(union))

A = {1,2,3,4,7}
B = {1,4,5,7,9}
C = {4, 2, 8, 9, 1}
#print(jaccard_of_sets([A, B, C]))

def jaccard_emb(ls_embeddings):
    num_embeddings = len(ls_embeddings)
    num_entities = len(ls_embeddings[0])
    nb = []
    for iter in range(num_embeddings):
        nb.append(neighborhoods(ls_embeddings[iter]))

    jaccard = np.zeros(num_entities)
    for i in range(num_entities):
        sets = []
        for j in range(num_embeddings):
            sets.append(set(nb[j][i]))
        jaccard[i] = jaccard_of_sets(sets)
    return(jaccard)

# test set:
emb_1 = np.array([[1, 2, 3, 4],
         [2, 4, 3, 9],
         [5, 8, 3, 10]])
emb_2 = np.array([[1, 2, 3, 4],
         [2, 5, 3, 9],
         [5, 1, 2, 3]
])
emb_3 = np.array([[1, 2, 3, 4],
                  [2, 4, 3, 9],
                  [5, 2, 14, 11]])
#jaccard_emb([emb_1, emb_2])