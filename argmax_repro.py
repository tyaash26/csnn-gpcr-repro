# CSNN reproduction: training-free "labels as features" Argmax prediction.
# For each labeled (compound, receptor), predict the receptor's bioactivity
# class by majority vote of the compound's chemical-space-network neighbours'
# labels for that receptor. Compare to truth -> weighted F1.
import numpy as np, pandas as pd, scipy.sparse as sp
from collections import Counter
from sklearn.metrics import f1_score, classification_report

B = 'data/01_DTU_neighbourhood_reporting/data_standalone/'
VALID = {'Agonist', 'Antagonist', 'Partial Agonist',
         'Inverse Agonist', 'Allosteric binding', 'No Effect'}
ALLOWED = VALID | {'Binding data only'}

df = pd.read_csv(B + 'gpcr_ChEMBL.csv', low_memory=False)
pos = df['Unnamed: 0'].to_numpy()                 # each compound's row in the CSN
rec = [c for c in df.columns
       if c not in ('Unnamed: 0', 'Smiles')
       and df[c].notna().sum() > 0
       and df[c].dropna().astype(str).isin(ALLOWED).mean() > 0.8]
print('receptors detected:', len(rec))

A = sp.load_npz(B + 'weighted_adjacency_matrix_180k_v2.npz').tocsr()
print('adjacency', A.shape, A.nnz, 'nonzeros')

yt, yp = [], []
for R in rec:
    col = df[R]
    rows = np.where(col.isin(VALID).to_numpy())[0]
    posR = {int(pos[i]): col.iloc[i] for i in rows}   # neighbour-position -> label
    for i in rows:
        p = int(pos[i])
        if p >= A.shape[0]:
            continue
        neigh = A.indices[A.indptr[p]:A.indptr[p + 1]]
        votes = [posR[int(q)] for q in neigh if int(q) != p and int(q) in posR]
        if not votes:                                  # no labelled neighbour -> abstain
            continue
        yp.append(Counter(votes).most_common(1)[0][0])
        yt.append(col.iloc[i])

print('scored pairs:', len(yt))
print('weighted F1:', round(f1_score(yt, yp, average='weighted'), 3))
print(classification_report(yt, yp, digits=3))
