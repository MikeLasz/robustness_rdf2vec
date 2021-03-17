import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--criterion", type=str, default="euclidean-knn", help="Select the stability criterion: euclidean-knn, cosine-knn, aligned_cosine, or sec_order_cosine")
args = parser.parse_args()

if __name__ == "__main__":
    ls_dim_red =["pca2d", "pca3d", "tsne2d", "tsne3d"]
    ls_data = ["aifb", "bgs", "mutag"]
    criterion = args.criterion

    results_cbow = []
    results_sg = []
    for dim_red in ls_dim_red:
        results_cbow_dimred = np.zeros(3)
        results_sg_dimred = np.zeros(3)

        results_cbow_dimred[0] = np.load("scores/" + criterion + "/aifb/cbow" + dim_red + "_score.npy")
        results_cbow_dimred[1] = np.load("scores/" + criterion + "/bgs/cbow" + dim_red + "_score.npy")
        results_cbow_dimred[2] = np.load("scores/" + criterion + "/mutag/cbow" + dim_red + "_score.npy")

        results_cbow.append(results_cbow_dimred)

        results_sg_dimred[0] = np.load("scores/" + criterion + "/aifb/skip-gram" + dim_red + "_score.npy")
        results_sg_dimred[1] = np.load("scores/" + criterion + "/bgs/skip-gram" + dim_red + "_score.npy")
        results_sg_dimred[2] = np.load("scores/" + criterion + "/mutag/skip-gram" + dim_red + "_score.npy")

        results_sg.append(results_sg_dimred)


    df_cbow = pd.DataFrame(data=results_cbow, index=ls_dim_red, columns=[s + " (cbow)" for s in ls_data])
    df_sg = pd.DataFrame(data=results_sg, index=ls_dim_red, columns=[s + " (sg)" for s in ls_data])

    df_both = [df_sg, df_cbow]
    print("Criterion: " + criterion)
    print(pd.concat(df_both, axis=1))
