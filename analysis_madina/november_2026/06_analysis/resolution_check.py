"""Do the weaker structures (resolution > 3 A or R-free > 0.30) change the clustering? (Nov 2026)

Units of the 6-descriptor table (clustering_bota/, 2 pairing errors left out, as in the benchmark) merged with
05_metadata_ligands/structure_metadata.csv.
  1. how many units come from weaker structures
  2. Spearman correlation of resolution with each descriptor
  3. share of weaker units among the noise of HDBSCAN, mean shift and grid (benchmark labels) vs overall
  4. HDBSCAN, Spectral and K-means refitted without the weaker units: agreement (ARI) with the full run
     on the remaining units (all units: k = 2; coactivators: Spectral k = 3, K-means k = 2)
Writes resolution_check.txt.  Run: python resolution_check.py
"""
import sys
from pathlib import Path

sys.argv = [sys.argv[0], 'bota']
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import clustering_benchmark as cb

HERE = Path(__file__).resolve().parent


def main():
    meta = pd.read_csv(cb.ROOT / '05_metadata_ligands/structure_metadata.csv')[['pdb_id', 'resolution_A', 'R_free']]
    df, feats = cb.load()
    df = df.merge(meta.drop_duplicates('pdb_id'), on='pdb_id', how='left')
    df['weak'] = (df.resolution_A > 3) | (df.R_free > 0.30)
    out = [f'units {len(df)}, from weaker structures {int(df.weak.sum())} ({df.weak.mean():.0%}) '
           f'in {df[df.weak].pdb_id.nunique()} PDB entries',
           f'resolution median {df.resolution_A.median():.2f} A, range {df.resolution_A.min():.2f}-'
           f'{df.resolution_A.max():.2f} A', '', 'Spearman rho, resolution vs descriptor:']
    for f in feats:
        r, p = spearmanr(df.resolution_A, df[f], nan_policy='omit')
        out.append(f'  {f:16s} {r:+.2f} (p = {p:.3f})')
    for subset, motif in cb.SUBSETS.items():
        sub = df if motif is None else df[df[cb.MOTIF] == motif].reset_index(drop=True)
        lab = pd.read_csv(cb.OUT / subset / 'cluster_labels_all_methods.csv')
        d = lab.merge(sub[['pdb_id', 'receptor_chain', 'peptide_chain', 'weak']],
                      on=['pdb_id', 'receptor_chain', 'peptide_chain'])
        out += ['', f'== {subset}: {len(d)} units, weaker {int(d.weak.sum())}']
        for m in ['HDBSCAN (density)', 'Mean shift (mode-seeking)', 'Grid (grid-based, PC1-PC2)']:
            noise = (d[m] == -1).to_numpy()
            if noise.any():
                out.append(f'  {m}: noise {noise.sum()}, weaker among noise {d.weak[noise].mean():.0%} '
                           f'vs {d.weak.mean():.0%} overall')
        f2, _ = cb.prune(sub, feats)
        X = StandardScaler().fit_transform(sub[f2])
        keep = ~sub.weak.to_numpy()
        for m, k in [('HDBSCAN (density)', None), ('Spectral (graph-based)', 2 if motif is None else 3),
                     ('K-means (partitioning)', 2)]:
            full = cb.run(m, X, k, cb.SEED)[keep]
            alone = cb.run(m, StandardScaler().fit_transform(sub[f2][keep]), k, cb.SEED)
            out.append(f'  {m}: same clusters without the weaker units? ARI {adjusted_rand_score(full, alone):.2f}')
    (HERE / 'resolution_check.txt').write_text('\n'.join(out), encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
