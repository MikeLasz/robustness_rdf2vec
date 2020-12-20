import numpy as np
from jaccard_similarity import jaccard_emb, jaccard_of_sets

data = "aifb" # aifb, mutag, or bgs
nlp_model = "cbow" # skip-gram or cbow
num_embeddings = 10
embeddings = []

dim_red = "tsne2d" # empty string "", "pca2d", "pca3d", "tsne2d", or "tsne3d"

# read embeddings:
if dim_red=="":
    for iter in range(num_embeddings):
        embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))
else:
    for iter in range(num_embeddings):
        embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter) + ".npy"))

# compute the score:
score = np.zeros(num_embeddings)
print("Results " + data)
for j in range(1, num_embeddings):
    score_iter = jaccard_emb(embeddings[0:j])
    score[j] = np.mean(score_iter)
    print("mean (iter:" + str(j) + ") = " + str(score[j]))
    print("var = " + str(np.var(score_iter)))


# save the results:
np.save("scores/" + data + "/" + nlp_model + dim_red + "_score.npy", score)