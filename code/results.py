import numpy as np
import pandas as pd

ls_results = []
ls_results.append(np.load("scores/aifb/cbow_score.npy"))
ls_results.append(np.load("scores/aifb/skip-gram_score.npy"))
ls_results.append(np.load("scores/bgs/cbow_score.npy"))
ls_results.append(np.load("scores/bgs/skip-gram_score.npy"))
ls_results.append(np.load("scores/mutag/cbow_score.npy"))
ls_results.append(np.load("scores/mutag/skip-gram_score.npy"))

#aifb_cbow = np.load("scores/aifb/cbow_score.npy")
#aifb_sg = np.load("scores/aifb/skip-gram_score.npy")
#bgs_cbow = np.load("scores/bgs/cbow_score.npy")
#bgs_sg = np.load("scores/bgs/skip-gram_score.npy")
#mutag_cbow = np.load("scores/mutag/cbow_score.npy")
#mutag_sg = np.load("scores/mutag/skip-gram_score.npy")

col_names = ["aifb_cbow", "aifb_sg", "bgs_cbow", "bgs_sg", "mutag_cbow", "mutag_sg"]
results = pd.DataFrame(columns=col_names)
i = 0
for col in col_names:
    results[col] = ls_results[i]
    i += 1

print(results)