# This was just an experiment. Cosine similarity is typically being used to quantify the simliarity of different documents.
# Recall the cricket example (https://www.machinelearningplus.com/nlp/cosine-similarity/) that I used in my presentation.

# I was sceptic if it would make sense to use it as a similarity measure for word embeddings because the direction in which an embeddings
# is pointing to is less meaningful in this case as it would be for word-count points. At least it is the case for 2-D or 3-D representations.

# But more importantly: We don't care about the orientation of an embedding. To clarify this: consider 2 2-D embeddings, where the second one is
# a 180 degrees rotated version of the first embedding. Then, we would quantify the embedding as a stable embedding (all neighborhoods remain the
# same, etc). However, the cosine similarity of these embeddings is 0.

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Import the embeddings
data = "aifb" # aifb, mutag, or bgs
nlp_model = "cbow" # skip-gram or cbow
num_embeddings = 10
embeddings = []
for iter in range(num_embeddings):
    embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))


# 2. Optional: Apply dimension reduction

# 3. Compute the cosine similarities
sim = cosine_similarity(embeddings[0], embeddings[1])

print(f'The resulting cosine-similarity matrix is {sim}, where the diagonal represents the similarity of 2 same words.')
print(f'overall Mean: {np.mean(sim)}')
print(f'overall variance: {np.var(sim)}')
print(f'Result: basically cosine similarity remains stable between all words, i.e. word a in embedding 1 and 2 is approx. equally similar '
      f'as word a and b in embedding 1 and 2.')