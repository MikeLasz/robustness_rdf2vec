import numpy as np
from utils.jaccard_similarity import jaccard_emb
from utils.aligned_cosine_similarity import aligned_cosine_similarity
from utils.second_order_cosine_similarity import sec_cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="aifb", help="Select the data: aifb, mutag, bgs")
parser.add_argument("--nlp_model", type=str, default="skip-gram", help="Select the NLP-model: skip-gram or cbow")
parser.add_argument("--num_embeddings", type=str, default=10, help="Number of embeddings that should be generated")
parser.add_argument("--dim_red", type=str, default="pca2d", help="Select the dimension reduction algorithm: no, pca2d, pca3d, tsne2d, or tsne3d")
parser.add_argument("--criterion", type=str, default="euclidean-knn", help="Select the stability criterion: euclidean-knn, cosine-knn, aligned_cosine, or sec_order_cosine")
args = parser.parse_args()


if __name__ == "__main__":
    data = args.data
    nlp_model = args.nlp_model
    num_embeddings = args.num_embeddings
    dim_red = args.dim_red
    if dim_red == "no":
        dim_red = ""
        print("no dimension reduction")

    criterion = args.criterion


    embeddings = []
    # read reduced embeddings:
    if dim_red=="":
        for iter in range(num_embeddings):
            embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))
    else:
        for iter in range(num_embeddings):
            embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter) + ".npy"))

    # compute the score:
    num_entities = len(embeddings[0])

    print("Results " + data + ":")
    if criterion=="euclidean-knn":
        results = jaccard_emb(embeddings, pairwise=True)
        print("mean = " + str(np.round(np.mean(results), 2)))
        print("var = " + str(np.round(np.var(results), 2)))
    elif criterion=="cosine-knn":
        results = jaccard_emb(embeddings, pairwise=True, distance="cosine")
        print("mean = " + str(np.round(np.mean(results), 2)))
        print("var = " + str(np.round(np.var(results), 2)))
    elif criterion=="sec_order_cosine":
        _, results = sec_cosine_similarity(embeddings)
        print("mean = " + str(np.round(np.mean(results), 2)))
        print("var = " + str(np.round(np.var(results), 2)))
    elif criterion=="aligned_cosine":
        _, results = aligned_cosine_similarity(embeddings)
        print("mean = " + str(np.round(np.mean(results), 2)))
        print("var = " + str(np.round(np.var(results), 2)))


    # save the results:
    PATH = "scores/" + criterion + "/" + data + "/"
    np.save(PATH + nlp_model + dim_red +"_score.npy", np.round(np.mean(results), 2))

    # generate histogram and boxplot:
    fig1, ax1 = plt.subplots()
    ax1.set_title('Jaccard similarities across entities (' + data + "; " + nlp_model + "; " + dim_red + ")")
    ax1 = sns.histplot(results, bins=10)
    ax1.set_xlabel("Jaccard Similarity")

    fig1.savefig(PATH + "plots/hist_" + nlp_model + dim_red + ".pdf")

    fig2, ax2 = plt.subplots()
    ax2.set_title('Jaccard similarities across entities (' + data + "; " + nlp_model + "; " + dim_red + ")")
    ax2 = sns.boxplot(results)
    ax2.set_xlabel("Jaccard Similarity")
    fig2.savefig(PATH + "plots/boxplot_" + nlp_model + dim_red + ".pdf")

    print("Figures are saved in " + PATH + "plots")
    #plt.show()