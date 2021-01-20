import numpy as np
import progressbar
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="aifb", help="Select the data: aifb, mutag, bgs")
parser.add_argument("--nlp_model", type=str, default="skip-gram", help="Select the NLP-model: skip-gram or cbow")
parser.add_argument("--num_embeddings", type=str, default=10, help="Number of embeddings that should be generated")
parser.add_argument("--dim_red", type=str, default="pca2d", help="Select the dimension reduction algorithm: pca2d, pca3d, tsne2d, or tsne3d")
args = parser.parse_args()


if __name__ == "__main__":
    data = args.data
    nlp_model = args.nlp_model
    num_embeddings = args.num_embeddings
    dim_red = args.dim_red

    # read the embeddings
    embeddings = []
    for iter in range(num_embeddings):
        embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))

    # apply dimension reduction
    reduced_embeddings = []

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