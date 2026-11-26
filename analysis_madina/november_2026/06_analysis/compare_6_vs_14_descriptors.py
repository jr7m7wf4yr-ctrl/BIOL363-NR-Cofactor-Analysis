"""The 4 clustering pictures shown to the team (HDBSCAN, Ward, Spectral, GMM), redone with the SAME method and SAME settings on both descriptor sets.

  6 descriptors   (clustering_bota/clustered_table_all_units.csv)
  14 descriptors  (04_cofactor_geometry/cofactor_geometry_full.csv, 11 after dropping near-duplicates)

Settings copied from the team's figures:
  HDBSCAN             all units, min_cluster_size = max(10, n/100)
  Ward                all units, tree cut into 4 clusters
  Spectral            coactivators only, 3 clusters, 15-nearest-neighbour graph
  Gaussian mixture    all units, 4 components, full covariance
  + Spectral k = 7    coactivators, the k the benchmark picked for the 14 descriptors

Only the descriptors differ, so any difference between left and right comes from the descriptors.

Run:  python compare_6_vs_14_descriptors.py bota   (6 descriptors)
      python compare_6_vs_14_descriptors.py ours   (14 descriptors)
      python compare_6_vs_14_descriptors.py compose      -> compare_6_vs_14_descriptors/compare_*.png + summary
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / 'compare_6_vs_14_descriptors'
# (file name, method in clustering_benchmark, subset, k)
RUNS = [('HDBSCAN', 'HDBSCAN (density)', 'all', None),
        ('Ward', 'Ward (hierarchical)', 'all', 4),
        ('Spectral', 'Spectral (graph-based)', 'coactivator', 3),
        ('Gaussian_mixture', 'Gaussian mixture (model-based)', 'all', 4),
        # extra: the best setting on our descriptors (spectral chose k = 7 by silhouette), for both sets
        ('Spectral_k7', 'Spectral (graph-based)', 'coactivator', 7)]
NAMES = {'bota': '6 descriptors', 'ours': '14 descriptors'}
FILE = {'bota': 'd6', 'ours': 'd14'}       # file-name prefix per descriptor set


def cluster(profile):
    import clustering_benchmark as cb
    import clustering_method_views as mv
    cb.style()
    df, all_feats = cb.load()
    rows = []
    for fname, method, subset, k in RUNS:
        sub = df if subset == 'all' else df[df[cb.MOTIF] != cb.COREPRESSOR].reset_index(drop=True)
        feats, _ = cb.prune(sub, all_feats)
        X = StandardScaler().fit_transform(sub[feats])
        lab = cb.run(method, X, k, cb.SEED)
        pca = PCA(2, random_state=cb.SEED).fit(X)
        fig, ax = plt.subplots(figsize=(15, 8))
        mv.family_map(ax, pca.transform(X), pca.explained_variance_ratio_ * 100, lab, sub, method.split(' (')[0])
        fig.tight_layout()
        fig.savefig(OUT / f'{FILE[profile]}_{fname}.png', dpi=110)
        plt.close(fig)
        ok = lab != -1
        rows.append({'descriptors': NAMES[profile], 'method': fname, 'subset': subset, 'units': len(sub),
                     'clusters': len(set(lab[ok])), 'noise': int((~ok).sum()),
                     'sizes': ' / '.join(map(str, pd.Series(lab[ok]).value_counts().values)),
                     'ARI_receptor': adjusted_rand_score(sub[cb.RECEPTOR][ok], lab[ok]),
                     'ARI_family': adjusted_rand_score(sub.family[ok], lab[ok]),
                     'clusters_pure_in_receptor': homogeneity_score(sub[cb.RECEPTOR][ok], lab[ok])})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'metrics_{FILE[profile]}.csv', index=False)


def compose():
    m = pd.concat([pd.read_csv(OUT / f'metrics_{FILE[p]}.csv') for p in NAMES], ignore_index=True)
    m.to_csv(OUT / 'metrics_both.csv', index=False)
    for i, (fname, *_rest) in enumerate(RUNS, 1):
        fig, axes = plt.subplots(1, 2, figsize=(26, 7.6))
        for ax, p in zip(axes, NAMES):
            ax.imshow(plt.imread(OUT / f'{FILE[p]}_{fname}.png'))
            ax.axis('off')
            r = m[(m.method == fname) & (m.descriptors == NAMES[p])].iloc[0]
            ax.set_title(f'{NAMES[p].upper()}\nclusters {r.clusters} ({r.sizes}), noise {r.noise} · '
                         f'match with receptor ARI {r.ARI_receptor:.2f} · clusters pure in one receptor '
                         f'{r.clusters_pure_in_receptor:.2f}', fontsize=13, loc='left')
        fig.suptitle(f'{fname.replace("_", " ")}: same method and settings, only the descriptors differ '
                     f'(left = 6 descriptors, right = 14 descriptors)', x=0.01, y=0.99, ha='left', fontsize=15, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f'compare_{i}_{fname}.png', dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = np.arange(len(RUNS))
    for ax, col, title in [(axes[0], 'ARI_receptor', 'Do clusters follow the receptors? (ARI)'),
                           (axes[1], 'clusters_pure_in_receptor', 'Is each cluster mostly one receptor?')]:
        for j, (p, c) in enumerate(zip(NAMES, ['#d6d5cf', '#2a78d6'])):
            v = [m[(m.method == r[0]) & (m.descriptors == NAMES[p])][col].iloc[0] for r in RUNS]
            ax.bar(x + (j - 0.5) * 0.36, v, 0.34, color=c, label=NAMES[p])
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 0.5) * 0.36, vi + 0.01, f'{vi:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x, [r[0].replace('_', ' ') for r in RUNS])
        ax.set_ylim(0, 1)
        ax.set_title(title, loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('HDBSCAN, Ward, Spectral, GMM with the same settings + Spectral k = 7; only the descriptors differ '
                 '(0 = random, 1 = perfect)',
                 x=0.01, ha='left', fontsize=10, color='#52514e')
    fig.tight_layout()
    fig.savefig(OUT / 'compare_0_summary.png', dpi=150)
    plt.close(fig)
    print(m.round(2).to_string(index=False))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    if 'compose' in sys.argv[1:]:
        compose()
    else:
        cluster('bota' if 'bota' in sys.argv[1:] else 'ours')
