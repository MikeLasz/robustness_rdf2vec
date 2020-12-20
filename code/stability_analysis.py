import numpy as np
from jaccard_similarity import jaccard_emb, jaccard_of_sets

data = "mutag" # aifb, mutag, or bgs
nlp_model = "skip-gram" # skip-gram or cbow
num_embeddings = 10
embeddings = []
for iter in range(num_embeddings):
    embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))

score = np.zeros(num_embeddings)
print("Results " + data)
for j in range(1,num_embeddings):
    score_iter = jaccard_emb(embeddings[0:j])
    score[j] = np.mean(score_iter)
    print("mean (iter:" + str(j) + ") = " + str(score[j]))
    print("var = " + str(np.var(score_iter)))

np.save("scores/" + data + "/" + nlp_model + "_score.npy", score)
