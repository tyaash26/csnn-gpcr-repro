# Project #4: Is GPCR polypharmacology (promiscuity) structurally predictable?
# Uses gpcr_ChEMBL.csv (labels) + maccs_all_ordered.npy (fingerprints).
# active = functional engagement; "No Effect" = negative; "Binding data only"/NaN = not functionally tested.
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

B = 'data/01_DTU_neighbourhood_reporting/data_standalone/'
ACTIVE = {'Agonist','Antagonist','Partial Agonist','Inverse Agonist','Allosteric binding'}
NEG = {'No Effect'}
TESTED = ACTIVE | NEG
MIN_TESTED = 5     # a compound must be functionally tested on >=5 receptors to count
PROM_THRESH = 5    # promiscuous: active on >=5 receptors
SEL_THRESH = 1     # selective: active on exactly 1
SEEDS = [0,1,2]

df = pd.read_csv(B+'gpcr_ChEMBL.csv', low_memory=False)
ALLOWED = ACTIVE | NEG | {'Binding data only'}
rec = [c for c in df.columns if c not in ('Unnamed: 0','Smiles')
       and df[c].notna().sum()>0 and df[c].dropna().astype(str).isin(ALLOWED).mean()>0.8]
smile_col = 'Smiles' if 'Smiles' in df.columns else max(
    [c for c in df.columns if df[c].dtype==object and c not in rec],
    key=lambda c: df[c].astype(str).str.len().mean())
print('receptors:', len(rec), '| smiles column:', smile_col)

R = df[rec]
active_count = R.isin(ACTIVE).sum(axis=1).to_numpy()
tested_count = R.isin(TESTED).sum(axis=1).to_numpy()

# ---------- A) characterize ----------
print('\n=== promiscuity distribution (compounds tested on >=%d receptors) ===' % MIN_TESTED)
elig = tested_count >= MIN_TESTED
print('eligible compounds:', int(elig.sum()))
ac = active_count[elig]
for lo,hi in [(0,0),(1,1),(2,2),(3,4),(5,9),(10,999)]:
    print('  active on %2d-%-3d receptors: %6d' % (lo,hi,int(((ac>=lo)&(ac<=hi)).sum())))

# which receptors attract promiscuous binders?
rows=[]
for Rc in rec:
    m = df[Rc].isin(ACTIVE).to_numpy()
    if m.sum()>=30:
        rows.append((Rc, int(m.sum()), round(float(active_count[m].mean()),2)))
rr = pd.DataFrame(rows, columns=['receptor','n_actives','mean_promiscuity_of_binders'])
print('\nreceptors whose binders are MOST promiscuous:')
print(rr.sort_values('mean_promiscuity_of_binders',ascending=False).head(8).to_string(index=False))
print('\nreceptors whose binders are MOST selective:')
print(rr.sort_values('mean_promiscuity_of_binders').head(8).to_string(index=False))

pd.DataFrame({'pos':df['Unnamed: 0'],'active_count':active_count,
              'tested_count':tested_count}).to_csv('promiscuity_scores.csv',index=False)

# ---------- B) predict promiscuous vs selective ----------
label = np.full(len(df), -1)
label[(active_count>=PROM_THRESH) & elig] = 1
label[(active_count==SEL_THRESH) & elig] = 0
keep = np.where(label>=0)[0]
print('\n=== predictive task ===')
print('promiscuous (>=%d):'%PROM_THRESH, int((label[keep]==1).sum()),
      '| selective (==1):', int((label[keep]==0).sum()))

pos = df['Unnamed: 0'].to_numpy()[keep].astype(int)
maccs = np.load(B+'maccs_all_ordered.npy')
X = np.unpackbits(maccs[pos], axis=1).astype(np.float32)   # 4096-bit fingerprints
y = label[keep]
print('feature matrix', X.shape)

# scaffold labels (for leakage-safe splitting)
smis = df[smile_col].to_numpy()[keep]
scaf = []
for s in smis:
    try: scaf.append(MurckoScaffold.MurckoScaffoldSmiles(str(s)) or str(s))
    except Exception: scaf.append(str(s))
scaf = np.array(scaf)
uniq = np.array(sorted(set(scaf)))

aucs, mccs, aps = [], [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed); order = uniq.copy(); rng.shuffle(order)
    test_scaf, n = set(), 0
    for s in order:
        test_scaf.add(s); n += int((scaf==s).sum())
        if n >= 0.2*len(keep): break
    te = np.array([s in test_scaf for s in scaf]); tr = ~te
    if len(set(y[tr]))<2 or len(set(y[te]))<2: continue
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:,1]
    aucs.append(roc_auc_score(y[te],p)); aps.append(average_precision_score(y[te],p))
    mccs.append(matthews_corrcoef(y[te], (p>0.5).astype(int)))

print('\nscaffold-split performance over %d seeds:' % len(aucs))
print('  AUROC   %.3f +/- %.3f' % (np.mean(aucs), np.std(aucs)))
print('  AUPRC   %.3f +/- %.3f  (base rate %.3f)' % (np.mean(aps), np.std(aps), (y==1).mean()))
print('  MCC     %.3f +/- %.3f' % (np.mean(mccs), np.std(mccs)))
print('  -> AUROC ~0.5 means structure does NOT predict promiscuity; >0.7 means it does.')

# ---------- C) optional t-SNE map colored by promiscuity ----------
try:
    ts = np.load(B+'tSNE_gpcr_fingerprint.npy')
    if ts.shape[0] >= int(pos.max())+1:
        plt.figure(figsize=(6,5))
        sc = plt.scatter(ts[pos,0], ts[pos,1], c=y, cmap='coolwarm', s=4, alpha=0.5)
        plt.title('Chemical space: promiscuous (red) vs selective (blue)')
        plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2'); plt.colorbar(sc)
        plt.tight_layout(); plt.savefig('promiscuity_tsne.png', dpi=150)
        print('\nsaved promiscuity_tsne.png')
except Exception as e:
    print('t-SNE plot skipped:', e)
print('saved promiscuity_scores.csv')
