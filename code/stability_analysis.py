import numpy as np
from jaccard_similarity import jaccard_emb, jaccard_of_sets
import matplotlib.pyplot as plt

data = "mutag" # aifb, mutag, or bgs
nlp_model = "cbow" # skip-gram or cbow
num_embeddings = 10
embeddings = []

dim_red = "tsne3d" # empty string "", "pca2d", "pca3d", "tsne2d", or "tsne3d"

# read embeddings:
if dim_red=="":
    for iter in range(num_embeddings):
        embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))
else:
    for iter in range(num_embeddings):
        embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter) + ".npy"))

# compute the score:
num_entities = len(embeddings[0])
data_for_histo = []#np.zeros(num_embeddings * num_entities)

print("Results " + data)
# for non-pairwise jaccard distance. Evolution over number of embeddings.
# non-pairwise jaccard distance converges to 0
#for j in range(1, num_embeddings):
    #score_iter = jaccard_emb(embeddings[0:j], pairwise=True)
    #score[j] = np.mean(score_iter)
    #print("mean (iter:" + str(j) + ") = " + str(score[j]))
    #print("var = " + str(np.var(score_iter)))
    #data_for_histo[(j-1)*num_entities:j*num_entities] = score_iter
    #data_for_histo.append(score_iter)

results_jaccard = jaccard_emb(embeddings, pairwise=True)
print("mean = " + str(np.round(np.mean(results_jaccard), 2)))
print("var = " + str(np.round(np.var(results_jaccard), 2)))


# save the results:
np.save("scores/" + data + "/" + nlp_model + dim_red + "_score.npy", np.round(np.mean(results_jaccard), 2))

# generate histogram and boxplot:
fig1, ax1 = plt.subplots()
ax1.set_title('Jaccard similarities across entities (' + data + "; " + nlp_model + "; " + dim_red + ")")
ax1.hist(results_jaccard, bins=10)
fig1.savefig("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/hist_jaccard.png")
#plt.show()

fig2, ax2 = plt.subplots()
ax2.set_title('Jaccard similarities across entities (' + data + "; " + nlp_model + "; " + dim_red + ")")
ax2.boxplot(results_jaccard)
fig2.savefig("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/boxplot_jaccard.png")
plt.show()