import numpy as np
from jaccard_similarity import jaccard_emb, jaccard_of_sets
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = "aifb" # aifb, mutag, or bgs
num_embeddings = 10
embeddings = []
len_emb = len(np.load("embeddings/" + data + "/cbow/pca2d/embedding1.npy"))

results = np.zeros((1, 2))
#dim_red = "pca2d" # empty string "", "pca2d", "pca3d", "tsne2d", or "tsne3d"
i = 0
for nlp_model in ("cbow", "skip-gram"):
    for dim_red in ("pca2d", "pca3d", "tsne2d", "tsne3d"):
        embeddings = []
        for iter in range(num_embeddings):
            embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter) + ".npy"))

        dist = jaccard_emb(embeddings, pairwise=True)
        if i==0:
            results = pd.DataFrame({"distance":dist, "method":np.repeat(nlp_model, len_emb), "dim_red": np.repeat(dim_red, len_emb)}, columns=("distance", "method", "dim_red") )#np.array([dist, np.repeat(nlp_model + dim_red, len_emb)])
        else:
            results = results.append(pd.DataFrame({"distance":dist, "method":np.repeat(nlp_model, len_emb), "dim_red": np.repeat(dim_red, len_emb)}, columns=("distance", "method", "dim_red") ))
        i += 1

print(results)

# generate boxplot
fig, ax = plt.subplots()
sns.set_theme(style="whitegrid")
color = sns.color_palette("tab10")
ax = sns.boxplot(x="dim_red", y="distance", hue="method", data=results, palette="tab10")
#ax.set_title("Pairwise similarities: " + data, fonzsize=30)
ax.set_xlabel("Dimension Reduction", fontsize=20)
ax.set_ylabel("Pairwise Jaccard distance", fontsize=20)
fig.suptitle("Pairwise similarities: " + data, fontsize=30)
fig.savefig("embeddings/" + data + "/boxplot_jaccard.pdf")
plt.show()
