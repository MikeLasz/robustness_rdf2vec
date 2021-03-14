from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pw_aligned_cosine_similarity(emb1, emb2):
    emb1, emb2, _ = procrustes(emb1, emb2) # emb1 is rescaled, and emb2 is rotated
    similarity_matrix = cosine_similarity(emb1, emb2)
    score = np.mean(np.diag(similarity_matrix))
    return(score, similarity_matrix)

def aligned_cosine_similarity(ls_embeddings):
    num_embeddings = len(ls_embeddings)
    scores = []
    for j in range(num_embeddings - 1):
        for i in range(j + 1, num_embeddings):
            emb1 = ls_embeddings[j]
            emb2 = ls_embeddings[i]
            pw_score, _ = pw_aligned_cosine_similarity(emb1, emb2)
            scores.append(pw_score)
    return(np.mean(scores), scores)

# Tests:
test = False
if test:
    # 1. emb2 is a scaled version of emb1
    print("Test 1: emb2 is a scaled version of emb1")
    emb1 = np.array([[1, 1],
                     [1, 0],
                     [0, 1],
                     [-1, -1]])
    emb2 = np.array([[2, 2],
                     [2, 0],
                     [0, 2],
                     [-2, -2]])
    print(pw_aligned_cosine_similarity(emb1, emb2)[0])

    # 2. Just like 1 but using different scales
    print("Test 2: Just like 1 but using different scales")
    emb2 = np.array([[2, 2],
                     [2, 0],
                     [0, 2],
                     [-1, -1]])
    print(pw_aligned_cosine_similarity(emb1, emb2)[0])
    print("This is due to the alignment in the aligned cosine similarity. In contrast, cosine similarity without alignment gives us:")
    print(np.mean(np.diag(cosine_similarity(emb1, emb2))))

    # 3. Swapping an embedding in emb1, i.e. swap the embedding of node 1 and 2 in emb1:
    print("Test 3: Swapping an embedding in emb1, i.e. swap the embedding of node 1 and 2 in emb1")
    emb1 = np.array([[1, 0],
                     [1, 1],
                     [0, 1],
                     [-1, -1]])
    emb2 = np.array([[2, 2],
                     [2, 0],
                     [0, 2],
                     [-2, -2]])
    print(pw_aligned_cosine_similarity(emb1, emb2)[0])

    # 4. Consider 3 embeddings: emb1, emb2 like in #1, emb3 = emb1
    print("Test 4: 3 embeddings; emb1, emb2 like in #1, emb3 = emb1")
    emb1 = np.array([[1, 1],
                     [1, 0],
                     [0, 1],
                     [-1, -1]])
    emb2 = np.array([[2, 2],
                     [2, 0],
                     [0, 2],
                     [-2, -2]])
    emb3 = emb1
    print(aligned_cosine_similarity([emb1, emb2, emb3]))

    # 5. Emb1 and emb2 are similar, emb3 swapped 2 node embeddings
    print("Test 5: Emb1 and emb2 are similar, emb3 swapped 2 node embeddings")
    emb1 = np.array([[1, 1],
                     [1, 0],
                     [0, 1],
                     [-1, -1]])
    emb2 = np.array([[1, 1],
                     [1, 0],
                     [0, 1],
                     [-1, -1]])
    emb3 = np.array([[1, 0],
                     [1, 1],
                     [0, 1],
                     [-1, -1]])
    print(aligned_cosine_similarity([emb1, emb2, emb3]))