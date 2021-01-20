import subprocess as s

test = False
if test:
    s.run(["python", "dim_reduction.py", "--data", "aifb"])
else:
    #for data in ("aifb", "bgs", "mutag"):
    for data in ("bgs", "mutag"):
        for nlp_model in ("cbow", "skip-gram"):
            for dim_red in ("pca2d", "pca3d", "tsne2d", "tsne3d"):
                print("Running: " + data + ", " + nlp_model + ", " + dim_red)
                s.run(["python", "dim_reduction.py", "--data", data, "--nlp_model", nlp_model, "--dim_red", dim_red])