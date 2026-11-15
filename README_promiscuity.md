# Is GPCR polypharmacology structurally predictable?

A follow-up analysis on the CSNN GPCR bioactivity dataset (Zenodo 12532113, CC-BY):
does a compound's *promiscuity* — how many GPCRs it functionally engages — follow
from its molecular structure alone?

## Definitions
- **Active** on a receptor = any functional class (Agonist, Antagonist, Partial
  Agonist, Inverse Agonist, Allosteric binding).
- **Negative** = "No Effect". "Binding data only" and missing values = not
  functionally tested.
- **Sparsity control:** only compounds functionally tested on **>= 5 receptors**
  are analysed, so "selective" means genuinely selective, not merely untested.
- **Promiscuity score** = number of receptors a compound is active on.

## Method
1. Compute each compound's promiscuity and tested-count from `gpcr_ChEMBL.csv`.
2. Characterise the distribution and ask which receptors attract promiscuous vs
   selective binders.
3. Predict **promiscuous (active on >=5)** vs **selective (active on exactly 1)**
   from 4096-bit MACCS/Morgan fingerprints, using a gradient-boosted classifier
   with **Murcko-scaffold splits** (no scaffold shared between train and test) over
   3 seeds.

## Results
- **3,436** compounds tested on >=5 receptors. Distribution of receptors engaged:
  0 -> 1211, 1 -> 788, 2 -> 373, 3-4 -> 400, 5-9 -> 530, 10+ -> 134. Most compounds
  are selective; a long tail is highly promiscuous.
- **Promiscuity is partly structural:** AUROC **0.719 +/- 0.032**, AUPRC
  **0.746 +/- 0.036** (base rate 0.457), MCC **0.309 +/- 0.062** on scaffold-split
  test sets. Above chance, but a modest signal — promiscuity is not a clean
  structural property.
- **Recovers known pharmacology:** aminergic receptors attract the most promiscuous
  binders (5-HT5A avg 17.5 receptors/binder, 5-HT1E, ADRA2B, H2, D5, 5-HT1B) — small
  biogenic-amine drugs share a conserved orthosteric pocket. Peptide / protease-
  activated / metabolite receptors (F2R/F2RL3, GnRH-R, HCAR2, GPR84, GPBAR1, CXCR7)
  attract selective binders (~1 receptor each). The modest AUROC reflects this:
  promiscuity rides largely on chemotype.

## Honest limitations
- Modest eligible set (3,436 compounds; 1,452 in the prediction task) and modest MCC
  (0.31) — a real but weak signal; not oversold.
- Promiscuity measured only over GPCRs present in this dataset (off-GPCR targets not
  considered).

## Files
- `promiscuity.py` — full analysis + scaffold-split prediction.
- `promiscuity_scores.csv` — per-compound promiscuity and tested-count (output).
- `promiscuity_tsne.png` — chemical-space map, promiscuous vs selective.

## Reproduce
```
pip install rdkit pandas scikit-learn matplotlib numpy
# download bundle 01 from Zenodo 12532113, unzip into data/
python promiscuity.py
```
