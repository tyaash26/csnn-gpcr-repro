# GPCR twist: does chemical homophily (and thus training-free predictability)
# track receptor FAMILY? Per-receptor F1 + neighbourhood homophily, grouped by
# family. The family map below is a STARTER built from name prefixes -- the
# GPCR expert should review and correct it (that correction is the contribution).
import numpy as np, pandas as pd, scipy.sparse as sp
from collections import Counter
from sklearn.metrics import f1_score

B = 'data/01_DTU_neighbourhood_reporting/data_standalone/'
VALID = {'Agonist','Antagonist','Partial Agonist','Inverse Agonist','Allosteric binding','No Effect'}
ALLOWED = VALID | {'Binding data only'}

df = pd.read_csv(B + 'gpcr_ChEMBL.csv', low_memory=False)
pos = df['Unnamed: 0'].to_numpy()
rec = [c for c in df.columns if c not in ('Unnamed: 0','Smiles')
       and df[c].notna().sum() > 0
       and df[c].dropna().astype(str).isin(ALLOWED).mean() > 0.8]
A = sp.load_npz(B + 'weighted_adjacency_matrix_180k_v2.npz').tocsr()

def family(r):
    r = r.upper()
    if (r.startswith(('ADRA','ADRB','ARDA','DRD','HTR','5HT','HRH','CHRM')) or r == 'TAAR1'):
        return 'Aminergic'
    if r.startswith(('ADORA','P2RY')):
        return 'Nucleotide'
    if (r.startswith(('S1PR','PTG','CNR','FFAR')) or r in
        ('TBXA2R','LTB4R','CYSLTR1','PTAFR','GPR84','GPR120','HCAR2','GPR35','GPR55','GPR6')):
        return 'Lipid'
    if (r.startswith(('OPR','CCR','CXCR','CX3','MC','AGTR','AVPR','BDKRB','NK',
                      'NPY','SSTR','CCK','HCRTR','EDNR','FPR')) or r in
        ('OXTR','GHSR','NTSR1','GNRHR','APLN','UTS2R','MCHR1','C5AR1','F2R','F2RL3',
         'NPSR1','NPBWR1','RXFP1','GIPR','TSHR')):
        return 'Peptide'
    return 'Other'

res = []
for R in rec:
    col = df[R]
    rows = np.where(col.isin(VALID).to_numpy())[0]
    posR = {int(pos[i]): col.iloc[i] for i in rows}
    yt, yp, hom = [], [], []
    for i in rows:
        p = int(pos[i])
        if p >= A.shape[0]:
            continue
        neigh = A.indices[A.indptr[p]:A.indptr[p + 1]]
        lab = [posR[int(q)] for q in neigh if int(q) != p and int(q) in posR]
        if not lab:
            continue
        true = col.iloc[i]
        yp.append(Counter(lab).most_common(1)[0][0]); yt.append(true)
        hom.append(np.mean([x == true for x in lab]))
    if len(yt) >= 30:
        res.append((R, family(R), len(yt),
                    round(f1_score(yt, yp, average='weighted'), 3),
                    round(float(np.mean(hom)), 3)))

rdf = pd.DataFrame(res, columns=['receptor','family','n','F1','homophily'])
rdf.to_csv('per_receptor_results.csv', index=False)
print('receptors reported (n>=30):', len(rdf))
print('\n=== BY FAMILY (mean over receptors) ===')
print(rdf.groupby('family').agg(n_receptors=('receptor','size'),
      mean_F1=('F1','mean'), mean_homophily=('homophily','mean'),
      total_pairs=('n','sum')).round(3).to_string())
print('\nF1 vs homophily Pearson r = %.3f (across %d receptors)' %
      (rdf['F1'].corr(rdf['homophily']), len(rdf)))
print('\nTOP 8 receptors by F1:')
print(rdf.sort_values('F1', ascending=False).head(8).to_string(index=False))
print('\nBOTTOM 8 receptors by F1:')
print(rdf.sort_values('F1').head(8).to_string(index=False))
print('\nsaved -> per_receptor_results.csv')
