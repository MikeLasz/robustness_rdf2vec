import numpy as np
from utils.get_neighbors import neighborhoods

# inputs a list of sets, outputs its generalized jaccard distance.
# Note: for list of 2 sets, this function computes the standard jaccard distance.
# Usage: input the neighborhoods as a list of sets. sets[0] = neighborhood of node 0, sets[1] = neighborhood of node 1, etc.
def jaccard_of_sets(sets):
    num_sets = len(sets)
    union = intersection = sets[0]
    for j in range(num_sets):
        union = union | sets[j]
        intersection = intersection & sets[j]
    return(len(intersection) / len(union))



def jaccard_emb(ls_embeddings, pairwise=True):
    num_embeddings = len(ls_embeddings)
    num_entities = len(ls_embeddings[0])
    num_pairs = int(num_embeddings * (num_embeddings - 1) / 2)
    nb = []
    for iter in range(num_embeddings):
        nb.append(neighborhoods(ls_embeddings[iter], neighborhood_size=3))
    if pairwise:
        pw_jaccard = np.zeros(num_entities)

        for ent in range(num_entities):
            jaccard = np.zeros(num_pairs)
            iter = 0
            for j in range(num_embeddings - 1):
                for i in range(j+1, num_embeddings):
                    # select the set that consists of the neighborhood of ent in embedding i and j
                    sets = []
                    sets.append(set(nb[j][ent]))
                    sets.append(set(nb[i][ent]))
                    jaccard[iter] = jaccard_of_sets(sets)
                    iter += 1
            pw_jaccard[ent] = np.mean(jaccard)
        return(pw_jaccard)
    else:
        jaccard = np.zeros(num_entities)
        for i in range(num_entities):
            sets = []
            for j in range(num_embeddings):
                sets.append(set(nb[j][i]))
            jaccard[i] = jaccard_of_sets(sets)
        return(jaccard)


# testing the implementation
testing = False # if you want to run some tests, set it to True
if testing:
    emb_1 = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0.5]])
    print("nb of emb_1:")
    print(neighborhoods(emb_1, neighborhood_size=3))


    emb_2 = np.array([[1, 0, 1],
                      [1, 0, 1],
                      [1, 0.5, 1],
                      [1, 0, 0.5],
                      [1, 0, 0.8]])
    print("emb_1 and emb_2 are supposed to have the same neighborhoods and hence the same Jaccard distance")
    print("nb of emb_2:")
    print(neighborhoods(emb_2, neighborhood_size=3))

    print("jaccard distance:")
    print(jaccard_emb([emb_1, emb_2], pairwise=True))
    emb_3 = np.array([[0, 0, 0.1],
                      [0, 0, 1],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
    print("nb of emb_3:")
    print(neighborhoods(emb_3, neighborhood_size=3))
    print("jaccard distances emb_1, emb_2, emb_3:")
    print("pairwise:")
    print(jaccard_emb([emb_1, emb_2, emb_3]))
    print("non-pairwise:")
    print(jaccard_emb([emb_1, emb_2, emb_3], pairwise=False))


