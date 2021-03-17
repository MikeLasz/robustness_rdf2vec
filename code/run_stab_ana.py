import subprocess as s

test = True
if test:
    s.run(["python", "stability_analysis.py", "--data", "aifb", "--criterion", "cosine-knn", "--dim_red", "no" ])
else:
    for data in ("aifb", "bgs", "mutag"):
        for nlp_model in ("cbow", "skip-gram"):
            for dim_red in ("pca2d", "pca3d", "tsne2d", "tsne3d"):
                # Run euclidean k-NN only for reduced data
                print("Running: " + data + ", " + nlp_model + ", " + dim_red)
                s.run(["python", "stability_analysis.py", "--data", data, "--nlp_model", nlp_model, "--dim_red", dim_red,
                       "--criterion", "euclidean-knn"])
    for data in ("aifb", "bgs", "mutag"):
        for nlp_model in ("cbow", "skip-gram"):
            for criterion in ("cosine-knn", "aligned_cosine", "sec_order_cosine"):
                # The other criteria do not require to employ a dimension reduction beforehand.
                print("Running: " + data + ", " + nlp_model + ", " + criterion)
                s.run(["python", "stability_analysis.py", "--data", data, "--nlp_model", nlp_model, "--dim_red", "no",
                       "--criterion", criterion])