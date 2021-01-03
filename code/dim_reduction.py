import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# get the embeddings
data = "aifb" # aifb, mutag, or bgs
nlp_model = "cbow" # skip-gram or cbow
num_embeddings = 10
embeddings = []
for iter in range(num_embeddings):
    embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))

# apply dimension reduction
dim_red = "pca2d" # "pca2d", "pca3d", "tsne2d", "tsne3d"
reduced_embeddings = []

import progressbar
with progressbar.ProgressBar(max_value=num_embeddings) as bar:
    bar.update(0)
    for iter in range(num_embeddings):
        if dim_red=="pca2d":
            pca = PCA(n_components=2)
            transformed_embedding = pca.fit_transform(embeddings[iter])
        elif dim_red=="pca3d":
            pca = PCA(n_components=3)
            transformed_embedding = pca.fit_transform(embeddings[iter])
        elif dim_red=="tsne2d":
            tsne = TSNE(n_components=2, random_state=42)
            transformed_embedding = tsne.fit_transform(embeddings[iter])
        elif dim_red=="tsne3d":
            tsne = TSNE(n_components=3, random_state=42)
            transformed_embedding = tsne.fit_transform(embeddings[iter])
        reduced_embeddings.append(transformed_embedding)
        bar.update(iter + 1)


# save the embeddings
for iter in range(num_embeddings):
    np.save("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter), reduced_embeddings[iter])