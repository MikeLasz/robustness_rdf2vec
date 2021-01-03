import numpy as np
import pandas as pd

ls_dim_red =["pca2d", "pca3d", "tsne2d", "tsne3d"]
ls_data = ["aifb", "bgs", "mutag"]

#results_cbow = pd.DataFrame(columns=("dim_red", "pca2d", "pca3d", "tsne2d", "tsne3d"))
#results_sg = pd.DataFrame(columns=("dim_red", "pca2d", "pca3d", "tsne2d", "tsne3d"))

results_cbow = []
results_sg = []
for dim_red in ls_dim_red:
    results_cbow_dimred = np.zeros(3)
    results_sg_dimred = np.zeros(3)

    print(np.load("scores/aifb/cbow" + dim_red + "_score.npy"))
    results_cbow_dimred[0] = np.load("scores/aifb/cbow" + dim_red + "_score.npy")
    results_cbow_dimred[1] = np.load("scores/bgs/cbow" + dim_red + "_score.npy")
    results_cbow_dimred[2] = np.load("scores/mutag/cbow" + dim_red + "_score.npy")

    results_cbow.append(results_cbow_dimred)

    results_sg_dimred[0] = np.load("scores/aifb/skip-gram" + dim_red + "_score.npy")
    results_sg_dimred[1] = np.load("scores/bgs/skip-gram" + dim_red + "_score.npy")
    results_sg_dimred[2] = np.load("scores/mutag/skip-gram" + dim_red + "_score.npy")

    results_sg.append(results_sg_dimred)


df_cbow = pd.DataFrame(data=results_cbow, index=ls_dim_red, columns=[s + " (cbow)" for s in ls_data])
df_sg = pd.DataFrame(data=results_sg, index=ls_dim_red, columns=[s + " (sg)" for s in ls_data])

df_both = [df_sg, df_cbow]
print(pd.concat(df_both, axis=1))
