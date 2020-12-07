import numpy as np
from jaccard_similarity import jaccard_emb, jaccard_of_sets

data = "aifb"
num_embeddings = 10
embeddings = []
for iter in range(num_embeddings):
    embeddings.append(np.load("embeddings/" + data + "/embedding" + str(iter) + ".npy"))

print("Results " + data)
for j in range(1,num_embeddings):
    score = jaccard_emb(embeddings[0:j])
    print("mean (iter:" + str(j) + ") = " + str(np.mean(score)))
    print("var = " + str(np.var(score)))