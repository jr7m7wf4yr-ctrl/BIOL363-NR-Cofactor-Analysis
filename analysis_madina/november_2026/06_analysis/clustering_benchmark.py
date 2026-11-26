"""Clustering benchmark of cofactor-binding geometry (prof's notes, Nov 2026).

One script: loads the descriptor table, describes the descriptors, runs 10 clustering methods (one per family),
scores each with the same metrics, picks the best one, and explains what drives its clusters.

  1. descriptors   how many (distances / angles / other), how many units; ranked from most to least variable
  2. methods       partitioning (k-means), hierarchical (Ward), density (HDBSCAN), model-based (Gaussian mixture),
                   graph-based (spectral), neural (self-organising map), grid-based (dense cells on PC1-PC2),
                   tree/CF (BIRCH), exemplar (affinity propagation), mode-seeking (mean shift)
  3. metrics       internal: silhouette (high), Davies-Bouldin (low), Calinski-Harabasz (high);
                   stability: mean ARI over bootstrap refits on 80% subsamples (high);
                   external: ARI / NMI vs receptor, NR family and motif type (do clusters follow biology?)
  4. best method   mean rank over silhouette, Davies-Bouldin, Calinski-Harabasz and stability; degenerate solutions
                   are ranked last: > 30% noise, > 20 clusters (fragmented), < 1% of units outside the largest
                   cluster (only outliers split off; such a k is also never chosen), or one cluster > 90% plus
                   crumbs (> 2 clusters; a clean 2-cluster split such as coactivator vs corepressor is kept)
  6. top 3         the 3 best methods compared: metrics, pairwise agreement (ARI) and match with receptors
                   (homogeneity = each cluster is one receptor, completeness = each receptor stays in one cluster)
  5. drivers       per descriptor: effect size between the best method's clusters (eta squared) and random-forest
                   importance; the same eta squared between receptors shows which descriptor differs most
  7. classification  clustering only finds groups with empty space between them; a classifier asks instead whether
                   the geometry still tells the receptors apart: random forest, 5-fold cross-validation with whole PDB
                   entries held out, balanced accuracy vs chance for NR family, subfamily and receptor; confusion
                   matrix (which receptors look alike) and descriptor importance (what differs between families)

To run on another descriptor table, change INPUT, FEATURES and the label columns, or add a profile like 'bota'.
Outputs go to 06_analysis/clustering/ (our table) or 06_analysis/clustering_bota/ (`python clustering_benchmark.py bota`).
`python clustering_benchmark.py [bota] classify` runs only step 7.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import (HDBSCAN, AffinityPropagation, AgglomerativeClustering, Birch, KMeans, MeanShift,
                             SpectralClustering, estimate_bandwidth)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, confusion_matrix, calinski_harabasz_score, completeness_score, davies_bouldin_score,
                             homogeneity_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'clustering'
SEED = 0
INPUT = ROOT / '04_cofactor_geometry/cofactor_geometry_full.csv'
# geometry descriptors of cofactor_geometry_full.csv; None = every numeric column ending in _A or _deg
FEATURES = ['centroid_to_motif_A', 'n_contact_residues',
            'h12_center_to_centroid_A', 'h12_center_to_motif_A', 'h12_axisline_to_centroid_A',
            'h12_axisline_to_motif_A', 'h12_vs_centroid_motif_deg', 'h12_vs_motif_axis_deg',
            'h12core_center_to_centroid_A', 'h12core_center_to_motif_A', 'h12core_axisline_to_centroid_A',
            'h12core_axisline_to_motif_A', 'h12core_vs_centroid_motif_deg', 'h12core_vs_motif_axis_deg']
# 'all' = every unit; the second run keeps coactivators only, since the corepressor split dominates the first
SUBSETS = {'all': None, 'coactivator_LXXLL': 'LXXLL'}
RECEPTOR, CODE, MOTIF = 'receptor_display', 'nr_code', 'motif_type'
SUBCODE, COREPRESSOR = 'nr_code', 'CoRNR'
EXCLUDE_UNITS = []          # (pdb_id, receptor_chain, peptide_chain) left out of the clustering      # column whose first 4 letters give the subfamily (NR1C); corepressor value
# Bota's table (1,105 units, 6 descriptors in the H3-H12-motif frame): run with `python clustering_benchmark.py bota`
PROFILE = next((a for a in sys.argv[1:] if a in ('ours', 'bota')), 'ours')
if PROFILE == 'bota':
    OUT = HERE / 'clustering_bota'
    INPUT = OUT / 'clustered_table_all_units.csv'
    FEATURES = ['angle_H3_H12', 'angle_H12_motif', 'angle_H3_motif', 'dist_H12_motif', 'dist_H3_motif', 'dist_H3_H12']
    SUBSETS = {'all': None, 'coactivator': 'COACTIVATOR'}
    RECEPTOR, CODE, MOTIF = 'receptor', 'subfamily', 'cofactor_type'
    SUBCODE, COREPRESSOR = 'family', 'COREPRESSOR'
    # 2 peptides outside the AF-2 groove (QC 'exclude'; dist_H3_motif 23 and 31 A vs median 10 A): they only split
    # off as their own 'cluster' and stretch the PCA; removed, as in Bota's spectral run (the units stay in the dataset)
    EXCLUDE_UNITS = [('1ZDU', 'A', 'Q'), ('7B9O', 'A', 'C')]
# gene-style names in Bota's table -> the display names used in the figures
DISPLAY = {'ESR1': 'ERα', 'ESR2': 'ERβ', 'PPARG': 'PPARγ', 'PPARA': 'PPARα', 'PPARD': 'PPARδ', 'RXRA': 'RXRα',
           'RXRB': 'RXRβ', 'RXRG': 'RXRγ', 'RORC': 'RORγ', 'RORA': 'RORα', 'LRH1': 'LRH-1', 'THRB': 'TRβ',
           'ESRRG': 'ERRγ', 'ESRRB': 'ERRβ', 'RARA': 'RARα', 'RARB': 'RARβ', 'LXR_ALPHA': 'LXRα', 'LXR_BETA': 'LXRβ',
           'HNF4A': 'HNF4α', 'SF1': 'SF-1'}
K_RANGE = range(2, 11)
CORR_DROP = 0.95          # of two descriptors correlated above this, keep one (else that geometry counts twice)
N_BOOT = 10
MIN_CLASS = 10            # classification: receptors with fewer units are left out (too few to learn)
MIN_OUTSIDE = 0.01        # a solution must put >= 1% of units outside its largest cluster (else = outliers only)

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#d6d5cf'
FAMILY_NAME = {'NR1': 'NR1 (thyroid-like)', 'NR2': 'NR2 (RXR-like)', 'NR3': 'NR3 (steroid)',
               'NR4': 'NR4 (NGFI-B)', 'NR5': 'NR5 (SF-1/LRH-1)', 'NR6': 'NR6 (GCNF)'}


def style():
    plt.rcParams.update({
        'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9,
        'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    })


def kind(col):
    if col.endswith('_deg') or col.startswith('angle'):
        return 'angle'
    return 'distance' if col.endswith('_A') or col.startswith('dist') else 'other'


# ---------- 1. data and descriptors ----------

def load():
    df = pd.read_csv(INPUT)
    if 'qc_flag' in df:
        df = df[(df.qc_flag != 'exclude') & df.h12_status.fillna('').str.startswith('PASS')]
    feats = FEATURES or [c for c in df.select_dtypes('number').columns if c.endswith(('_A', '_deg'))]
    drop = set(EXCLUDE_UNITS)
    df = df[[(p, r, c) not in drop for p, r, c in zip(df.pdb_id, df.receptor_chain, df.peptide_chain)]]
    df = df.dropna(subset=feats).reset_index(drop=True)
    df[RECEPTOR] = df[RECEPTOR].replace(DISPLAY)
    df['nr_subfamily'] = df[SUBCODE].str[:4]
    df['family'] = df[CODE].str[:3].map(FAMILY_NAME)
    return df, feats


def prune(df, feats):
    """Drop one of each near-duplicate pair, keeping the earlier column."""
    corr = df[feats].corr().abs()
    keep, dropped = [], []
    for f in feats:
        twin = next((k for k in keep if corr.loc[f, k] > CORR_DROP), None)
        (dropped.append((f, twin, corr.loc[f, twin])) if twin else keep.append(f))
    return keep, dropped


def eta_squared(values, groups):
    """Share of a descriptor's variance explained by the groups (0 = no difference, 1 = groups fully separate)."""
    v = pd.Series(values)
    g = pd.Series(np.asarray(groups))
    total = ((v - v.mean()) ** 2).sum()
    between = sum(len(x) * (x.mean() - v.mean()) ** 2 for _, x in v.groupby(g.values))
    return between / total if total else 0.0


def describe(df, feats):
    big = df[RECEPTOR].value_counts()
    big = big[big >= 10].index      # receptors with too few units would inflate eta squared
    sub = df[df[RECEPTOR].isin(big)]
    rows = []
    for f in feats:
        x = df[f]
        q5, q25, q50, q75, q95 = x.quantile([.05, .25, .5, .75, .95])
        rows.append({'descriptor': f, 'type': kind(f), 'median': q50, 'IQR': q75 - q25, 'range_5_95': q95 - q5,
                     'eta2_receptor': eta_squared(sub[f].values, sub[RECEPTOR].values),
                     'eta2_family': eta_squared(df[f].values, df.family.values),
                     'eta2_motif': eta_squared(df[f].values, df[MOTIF].values)})
    d = pd.DataFrame(rows)
    # flexibility can only be compared within one unit (Å with Å, ° with °): rank by IQR inside each type
    d['flexibility_rank_in_type'] = d.groupby('type').IQR.rank(ascending=False).astype(int)
    return d.sort_values(['type', 'flexibility_rank_in_type'])


# ---------- 2. methods ----------

def som(X, rows=8, cols=8, iters=6000, seed=SEED):
    """Minimal self-organising map; returns the codebook (one prototype vector per map node)."""
    rng = np.random.default_rng(seed)
    w = X[rng.choice(len(X), rows * cols, replace=False)].copy()
    grid = np.array([(i, j) for i in range(rows) for j in range(cols)], float)
    for t in range(iters):
        lr, sigma = 0.5 * (1 - t / iters) + 0.01, max(rows, cols) / 2 * (1 - t / iters) + 0.5
        x = X[rng.integers(len(X))]
        bmu = np.argmin(((w - x) ** 2).sum(1))
        h = np.exp(-((grid - grid[bmu]) ** 2).sum(1) / (2 * sigma ** 2))
        w += lr * h[:, None] * (x - w)
    return w


def som_cluster(X, k, seed=SEED):
    """Two-stage SOM clustering: train the map, cluster its prototypes (Ward), give each unit its node's cluster."""
    w = som(X, seed=seed)
    node_label = fcluster(linkage(w, 'ward'), k, 'maxclust')
    bmu = np.argmin(((X[:, None, :] - w[None]) ** 2).sum(2), 1)
    return node_label[bmu]


def grid_cluster(X, seed=SEED, bins=20, min_pts=5):
    """Grid-based (STING/CLIQUE idea) on PC1-PC2: dense cells joined with their neighbours; sparse cells = noise."""
    p = PCA(2, random_state=seed).fit_transform(X)
    edges = [np.linspace(p[:, i].min(), p[:, i].max() + 1e-9, bins + 1) for i in range(2)]
    ix = [np.clip(np.digitize(p[:, i], edges[i]) - 1, 0, bins - 1) for i in range(2)]
    counts = np.zeros((bins, bins), int)
    np.add.at(counts, (ix[0], ix[1]), 1)
    cells, _ = ndimage.label(counts >= min_pts, structure=np.ones((3, 3)))
    return np.where(cells[ix[0], ix[1]] > 0, cells[ix[0], ix[1]], -1)


K_METHODS = {
    'K-means (partitioning)': lambda X, k, s: KMeans(k, n_init=10, random_state=s).fit_predict(X),
    'Ward (hierarchical)': lambda X, k, s: AgglomerativeClustering(k, linkage='ward').fit_predict(X),
    'Spectral (graph-based)': lambda X, k, s: SpectralClustering(
        k, affinity='nearest_neighbors', n_neighbors=15, random_state=s).fit_predict(X),
    'SOM (neural)': lambda X, k, s: som_cluster(X, k, s),
    'BIRCH (tree / CF)': lambda X, k, s: Birch(n_clusters=k).fit_predict(X),
}
AUTO_METHODS = {
    'HDBSCAN (density)': lambda X, s: HDBSCAN(min_cluster_size=max(10, len(X) // 100)).fit_predict(X),
    'Affinity propagation (exemplar)': lambda X, s: AffinityPropagation(
        preference=-50, damping=0.9, max_iter=500, random_state=s).fit_predict(X),
    'Mean shift (mode-seeking)': lambda X, s: MeanShift(
        bandwidth=estimate_bandwidth(X, quantile=0.2, random_state=s), bin_seeding=True).fit_predict(X),
    'Grid (grid-based, PC1-PC2)': lambda X, s: grid_cluster(X, s),
}


def gmm(X, k, s):
    return GaussianMixture(k, covariance_type='full', n_init=3, random_state=s).fit(X)


def internal(X, labels):
    """Metrics on the non-noise units; NaN when fewer than 2 clusters."""
    m = labels != -1
    if len(set(labels[m])) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(silhouette=silhouette_score(X[m], labels[m]), davies_bouldin=davies_bouldin_score(X[m], labels[m]),
                calinski_harabasz=calinski_harabasz_score(X[m], labels[m]))


def outside_largest(labels):
    """Share of the units outside the largest cluster (noise not counted)."""
    sizes = pd.Series(labels[labels != -1]).value_counts()
    return (sizes.sum() - sizes.max()) / len(labels) if len(sizes) else 0.0


def scan_k(X):
    """Silhouette and BIC over k for the methods that need k; chosen k = best silhouette (BIC for the mixture).
    A k whose clusters other than the largest hold < 1% of the units only splits off outliers: it is not chosen."""
    rows = []
    for name, fit in K_METHODS.items():
        for k in K_RANGE:
            lab = np.asarray(fit(X, k, SEED))
            rows.append({'method': name, 'k': k, 'outside_largest': outside_largest(lab), **internal(X, lab)})
    for k in K_RANGE:
        g = gmm(X, k, SEED)
        lab = g.predict(X)
        rows.append({'method': 'Gaussian mixture (model-based)', 'k': k, 'bic': g.bic(X),
                     'outside_largest': outside_largest(lab), **internal(X, lab)})
    scan = pd.DataFrame(rows)
    ok = scan[scan.outside_largest >= MIN_OUTSIDE]
    ok = ok if len(ok) else scan
    best_k = {m: int(g.loc[g.silhouette.idxmax(), 'k']) for m, g in ok.groupby('method')}
    mix = ok[ok.method.str.startswith('Gaussian')]
    best_k['Gaussian mixture (model-based)'] = int(mix.loc[mix.bic.idxmin(), 'k'])
    return scan, best_k


def run(name, X, k, seed):
    if name in K_METHODS:
        return np.asarray(K_METHODS[name](X, k, seed))
    if name.startswith('Gaussian'):
        return gmm(X, k, seed).predict(X)
    return np.asarray(AUTO_METHODS[name](X, seed))


def stability(name, X, k):
    """Refit on 80% subsamples; ARI with the full-data labels on the same units (label names do not matter)."""
    rng = np.random.default_rng(SEED)
    full = run(name, X, k, SEED)
    aris = []
    for b in range(N_BOOT):
        idx = np.sort(rng.choice(len(X), int(0.8 * len(X)), replace=False))
        aris.append(adjusted_rand_score(full[idx], run(name, X[idx], k, SEED + b + 1)))
    return float(np.mean(aris))


def benchmark(df, X, best_k):
    rows, labels = [], {}
    names = list(K_METHODS) + ['Gaussian mixture (model-based)'] + list(AUTO_METHODS)
    for name in names:
        k = best_k.get(name)
        lab = run(name, X, k, SEED)
        labels[name] = lab
        sizes = pd.Series(lab[lab != -1]).value_counts()
        m = lab != -1
        rows.append({
            'method': name, 'clusters': len(sizes), 'k_chosen_by': ('BIC' if name.startswith('Gaussian') else
                                                                     'silhouette' if k else 'method itself'),
            'noise_fraction': 1 - m.mean(), 'largest_cluster_fraction': sizes.max() / len(lab) if len(sizes) else 1,
            'smallest_cluster_fraction': sizes.min() / len(lab) if len(sizes) else 0,
            **internal(X, lab), 'stability_ARI': stability(name, X, k),
            **{f'ARI_{t}': adjusted_rand_score(df.loc[m, c], lab[m]) for t, c in
               [('receptor', RECEPTOR), ('family', 'family'), ('motif', MOTIF)]},
            **{f'NMI_{t}': normalized_mutual_info_score(df.loc[m, c], lab[m]) for t, c in
               [('receptor', RECEPTOR), ('family', 'family'), ('motif', MOTIF)]},
            'homogeneity_receptor': homogeneity_score(df.loc[m, RECEPTOR], lab[m]),
            'completeness_receptor': completeness_score(df.loc[m, RECEPTOR], lab[m]),
        })
        print(f'  {name}: {len(sizes)} clusters', flush=True)
    res = pd.DataFrame(rows)
    ranks = pd.concat([res.silhouette.rank(ascending=False), res.davies_bouldin.rank(),
                       res.calinski_harabasz.rank(ascending=False), res.stability_ARI.rank(ascending=False)], axis=1)
    res['mean_rank'] = ranks.mean(axis=1)
    outside = 1 - res.noise_fraction - res.largest_cluster_fraction
    res['degenerate'] = ((res.noise_fraction > 0.3) | (res.clusters < 2) | (res.clusters > 20)
                         | (outside < MIN_OUTSIDE) | ((res.largest_cluster_fraction > 0.9) & (res.clusters > 2)))
    # ties in mean rank are broken by stability (a result that reproduces is worth more)
    res = res.sort_values(['degenerate', 'mean_rank', 'stability_ARI'], ascending=[True, True, False]).reset_index(drop=True)
    return res, labels


def drivers(df, X, feats, lab):
    m = lab != -1
    rf = RandomForestClassifier(300, random_state=SEED).fit(X[m], lab[m])
    d = pd.DataFrame({'descriptor': feats, 'type': [kind(f) for f in feats],
                      'eta2_between_clusters': [eta_squared(df.loc[m, f].values, lab[m]) for f in feats],
                      'random_forest_importance': rf.feature_importances_})
    means = df[m].groupby(lab[m])[feats].median().T
    means.columns = [f'median_cluster_{c}' for c in means.columns]
    return d.merge(means, left_on='descriptor', right_index=True).sort_values('eta2_between_clusters',
                                                                               ascending=False)


# ---------- plots ----------

def colours(values, top=7):
    """Top categories get the fixed palette order; the rest fold into grey 'Other' (never a 9th hue)."""
    counts = pd.Series(values).value_counts()
    keep = list(counts.index[:top])
    cmap = {c: SERIES[i] for i, c in enumerate(keep)}
    return [cmap.get(v, OTHER) for v in values], cmap


def scatter(ax, xy, values, title, top=7, order_back=None):
    cols, cmap = colours(values, top)
    cols = np.array(cols)
    back = np.isin(cols, [OTHER]) if order_back is None else order_back
    ax.scatter(*xy[back].T, s=10, c=OTHER, linewidths=0, label='Other' if back.any() else None)
    ax.scatter(*xy[~back].T, s=12, c=cols[~back], linewidths=0.3, edgecolors=SURFACE)
    for name, c in cmap.items():
        ax.scatter([], [], s=30, c=c, label=f'{name} ({(pd.Series(values) == name).sum()})')
    ax.legend(fontsize=7, frameon=False, loc='best', markerscale=0.9)
    ax.set_title(title)


def subtype_panel(ax, xy, df, members, title):
    vals = df[RECEPTOR].where(df[RECEPTOR].isin(members), 'Other')
    back = (vals == 'Other').values
    ax.scatter(*xy[back].T, s=8, c=OTHER, linewidths=0, label='Other receptors')
    for i, r in enumerate(members):
        m = (vals == r).values
        if not m.any():
            continue
        ax.scatter(*xy[m].T, s=14, c=SERIES[i], linewidths=0.3, edgecolors=SURFACE, label=f'{r} ({m.sum()})')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(title)


def embedding_figure(df, xy, labels_best, best, axis_names, fname, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    scatter(axes[0, 0], xy, df[RECEPTOR].values, 'Individual receptors (top 7)')
    scatter(axes[0, 1], xy, df.family.values, 'Evolutionary families')
    scatter(axes[0, 2], xy, df[MOTIF].values, 'Motif type (coactivator LXXLL / corepressor CoRNR)')
    subtype_panel(axes[1, 0], xy, df, ['PPARα', 'PPARγ', 'PPARδ'], 'Subtypes: PPARα / γ / δ')
    subtype_panel(axes[1, 1], xy, df, ['ERα', 'ERβ'], 'Subtypes: ERα / ERβ')
    lab = np.array([f'cluster {c}' if c != -1 else 'noise' for c in labels_best])
    scatter(axes[1, 2], xy, lab, f'Best method: {best}', top=8)
    for ax in axes.flat:
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
    fig.tight_layout()
    fig.savefig(out / fname, dpi=170)
    plt.close(fig)


def metrics_figure(res, out):
    cols = [('silhouette', 'Silhouette  (higher = better)'), ('davies_bouldin', 'Davies-Bouldin  (lower = better)'),
            ('stability_ARI', 'Stability, bootstrap ARI  (higher = better)'),
            ('ARI_receptor', 'Agreement with receptor, ARI')]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), sharey=True)
    y = np.arange(len(res))
    for ax, (c, t) in zip(axes, cols):
        colour = np.where(res.degenerate, OTHER, SERIES[0])
        colour[0] = SERIES[1]
        ax.barh(y, res[c], color=colour, height=0.66)
        for i, v in enumerate(res[c]):
            ax.text(v, i, f' {v:.2f}', va='center', fontsize=8, color=INK2)
        ax.set_title(t)
    axes[0].invert_yaxis()      # shared y: invert once, best method on top
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'{m}  ({n} cl.)' for m, n in zip(res.method, res.clusters)])
    fig.suptitle('Clustering benchmark: orange = best by mean rank, grey = degenerate (> 30% noise, > 20 clusters, '
                 'or one cluster > 90% plus crumbs)',
                 x=0.01, ha='left', fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'benchmark_metrics.png', dpi=170)
    plt.close(fig)


def k_scan_figure(scan, out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (m, g) in enumerate(scan.groupby('method', sort=False)):
        ax.plot(g.k, g.silhouette, marker='o', ms=4, lw=2, color=SERIES[i], label=m)
    ax.set_xlabel('number of clusters k')
    ax.set_ylabel('silhouette')
    ax.set_title('How many clusters? Silhouette for each k')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'k_scan_silhouette.png', dpi=170)
    plt.close(fig)


def descriptor_figures(df, desc, feats, out):
    n = len(feats)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(14, 2.6 * np.ceil(n / cols)))
    order = desc.descriptor.tolist()
    for ax, f in zip(axes.flat, order):
        ax.hist(df[f], bins=40, color=SERIES[0] if kind(f) == 'distance' else SERIES[1] if kind(f) == 'angle'
                else SERIES[2], linewidth=0)
        r = desc.set_index('descriptor').loc[f]
        ax.set_title(f'{f}\nIQR {r.IQR:.1f} · flex. rank {r.flexibility_rank_in_type} ({r.type})', fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle('Descriptor distributions (blue = distance Å, orange = angle °, aqua = count)', x=0.01, ha='left',
                 color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'descriptor_distributions.png', dpi=150)
    plt.close(fig)

    d = desc.sort_values('eta2_receptor')
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.2))
    ax.barh(d.descriptor, d.eta2_receptor, color=SERIES[0], height=0.66)
    for i, v in enumerate(d.eta2_receptor):
        ax.text(v, i, f' {v:.2f}', va='center', fontsize=8, color=INK2)
    ax.set_xlabel('η²: share of the variance explained by the receptor')
    ax.set_title('Which descriptor differs most between receptors')
    fig.tight_layout()
    fig.savefig(out / 'descriptor_difference_between_receptors.png', dpi=170)
    plt.close(fig)


def drivers_figure(drv, best, out):
    d = drv.sort_values('eta2_between_clusters')
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.2))
    ax.barh(d.descriptor, d.eta2_between_clusters, color=SERIES[1], height=0.66)
    for i, v in enumerate(d.eta2_between_clusters):
        ax.text(v, i, f' {v:.2f}', va='center', fontsize=8, color=INK2)
    ax.set_xlabel('η²: share of the variance explained by the clusters')
    ax.set_title(f'What drives the separation ({best})')
    fig.tight_layout()
    fig.savefig(out / 'cluster_drivers.png', dpi=170)
    plt.close(fig)


def composition_figure(df, lab, best, out):
    m = lab != -1
    tab = pd.crosstab(lab[m], df.loc[m, RECEPTOR])
    top = df[RECEPTOR].value_counts().index[:7]
    tab = tab.reindex(columns=list(top), fill_value=0).assign(Other=tab.drop(columns=top, errors='ignore').sum(1))
    share = tab.div(tab.sum(1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(share) + 1.6))
    left = np.zeros(len(share))
    for i, c in enumerate(share.columns):
        ax.barh([f'cluster {k} (n={n})' for k, n in zip(share.index, tab.sum(1))], share[c], left=left,
                color=SERIES[i] if c != 'Other' else OTHER, height=0.66, edgecolor=SURFACE, linewidth=1, label=c)
        left += share[c].values
    ax.set_xlabel('% of the cluster')
    ax.set_title(f'Which receptors are in each cluster ({best})')
    ax.legend(fontsize=8, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / 'cluster_composition.png', dpi=170)
    plt.close(fig)


def methods_figure(df, xy, labels, res, axis_names, fname, out):
    """One map per method, same points, coloured by that method's clusters; first panel = receptors for reference."""
    fig, axes = plt.subplots(3, 4, figsize=(19, 13.5))
    scatter(axes[0, 0], xy, df[RECEPTOR].values, 'Reference: coloured by receptor')
    info = res.set_index('method')
    for ax, name in zip(axes.flat[1:], res.method):     # benchmark order: best first, degenerate last
        lab, r = labels[name], info.loc[name]
        names = np.array([f'cluster {c}' if c != -1 else 'noise' for c in lab])
        sizes = pd.Series(names[lab != -1]).value_counts()
        cmap = {c: SERIES[i] for i, c in enumerate(sizes.index[:7])}     # 7 largest clusters, rest grey
        cols = np.array([cmap.get(n, OTHER) for n in names])
        back = cols == OTHER
        ax.scatter(*xy[back].T, s=7, c=OTHER, linewidths=0)
        ax.scatter(*xy[~back].T, s=9, c=cols[~back], linewidths=0.2, edgecolors=SURFACE)
        flag = '  (degenerate)' if r.degenerate else '  ★ best' if name == res.method.iloc[0] else ''
        ax.set_title(f'{name}{flag}', fontsize=9.5)
        extra = f', {r.noise_fraction:.0%} noise' if r.noise_fraction else ''
        ax.text(0.02, 0.98, f'{r.clusters} clusters{extra}\nsilhouette {r.silhouette:.2f} · stability {r.stability_ARI:.2f}'
                f'\nmatch with receptor (ARI) {r.ARI_receptor:.2f}', transform=ax.transAxes, va='top', fontsize=7.5,
                color=INK2, bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.85, pad=2))
    axes.flat[-1].set_visible(False)
    for ax in axes.flat:
        ax.set_xlabel(axis_names[0], fontsize=8)
        ax.set_ylabel(axis_names[1], fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle('Same units in every panel, coloured by the clusters of each method (7 largest in colour, rest and noise '
                 'grey); best first, degenerate last', x=0.01, ha='left', color=INK2)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def majority_tags(df, lab):
    """'cluster c: ERα 80%' for each cluster of one method."""
    tags = {}
    for c in sorted(set(lab) - {-1}):
        counts = df.loc[lab == c, RECEPTOR].value_counts()
        tags[c] = f'{counts.index[0]} {counts.iloc[0] / counts.sum():.0%} (n={counts.sum()})'
    return tags


def top3_figure(df, res, labels, out):
    """Top 3 methods: metrics side by side, agreement between methods, and which clusters correspond."""
    good = res[~res.degenerate].method.tolist()
    top = good[:3]
    short = {m: m.split(' (')[0] for m in res.method}
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.55], hspace=0.45, wspace=0.55)
    metrics = [('silhouette', 'Silhouette ↑'), ('davies_bouldin', 'Davies-Bouldin ↓'),
               ('stability_ARI', 'Stability ↑'), ('ARI_receptor', 'Match receptor (ARI) ↑'),
               ('completeness_receptor', 'Receptor kept together ↑'), ('homogeneity_receptor', 'Cluster = one receptor ↑')]
    info = res.set_index('method')
    for i, (c, t) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        vals = [info.loc[m, c] for m in top]
        ax.barh(range(3), vals, color=SERIES[0], height=0.6)
        for j, v in enumerate(vals):
            ax.text(v, j, f' {v:.2f}', va='center', fontsize=9, color=INK2)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f'{short[m]} ({info.loc[m, "clusters"]})' if i == 0 else '' for m in top])
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_title(t, fontsize=9)

    ax = fig.add_subplot(gs[1, :3])
    names = good[:6]
    agree = np.array([[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names])
    blues = matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#cde2fb', '#3987e5', '#0d366b'])
    ax.imshow(agree, cmap=blues, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{agree[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='#ffffff' if agree[i, j] > 0.55 else INK2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short[m] for m in names], rotation=30, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short[m] for m in names])
    ax.grid(False)
    ax.set_title('Do the methods find the same clusters? (ARI between methods, 1 = identical)')

    ax = fig.add_subplot(gs[1, 3:])
    a, b = top[0], top[1]
    tab = pd.crosstab(labels[a], labels[b])
    tab = tab.loc[[c for c in tab.index if c != -1], [c for c in tab.columns if c != -1]]
    order = tab.idxmax(axis=0).sort_values(kind='stable').index      # line columns up with their best row
    tab = tab[order]
    share = tab.div(tab.sum(1), axis=0)
    ax.imshow(share.values, cmap=blues, vmin=0, vmax=1, aspect='auto')
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            if tab.values[i, j]:
                ax.text(j, i, tab.values[i, j], ha='center', va='center', fontsize=8,
                        color='#ffffff' if share.values[i, j] > 0.55 else INK2)
    ta, tb = majority_tags(df, labels[a]), majority_tags(df, labels[b])
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels([ta[c] for c in tab.index], fontsize=8)
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels([tb[c] for c in tab.columns], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(f'{short[a]} clusters')
    ax.set_xlabel(f'{short[b]} clusters')
    ax.grid(False)
    ax.set_title(f'Which clusters correspond: {short[a]} vs {short[b]} (units; dark = most of the row)')
    fig.suptitle(f'Top 3 methods: {", ".join(short[m] for m in top)}', x=0.01, ha='left', fontsize=12, color=INK2)
    fig.savefig(out / 'top3_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(agree, index=names, columns=names).round(3).to_csv(out / 'method_agreement_ARI.csv')


def receptor_panels(df, xy, axis_names, fname, out, cols=6):
    """Small multiples: one panel per receptor, that receptor highlighted on the shared map, the rest grey."""
    counts = df[RECEPTOR].value_counts()
    rows = int(np.ceil(len(counts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.9 * rows), sharex=True, sharey=True)
    family = df.groupby(RECEPTOR).family.first()
    for ax, (r, n) in zip(axes.flat, counts.items()):
        m = (df[RECEPTOR] == r).values
        ax.scatter(*xy[~m].T, s=3, c=OTHER, linewidths=0)
        ax.scatter(*xy[m].T, s=9, c=SERIES[0], linewidths=0.2, edgecolors=SURFACE)
        ax.set_title(f'{r}  (n={n})', fontsize=9)
        ax.text(0.03, 0.04, str(family[r]).split(' ')[0], transform=ax.transAxes, fontsize=7, color=MUTED)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(counts):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(axis_names[1], fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(axis_names[0], fontsize=7)
    fig.suptitle('Where is each receptor? Same map in every panel; the named receptor in blue, all others grey '
                 '(NR family bottom left)', x=0.01, ha='left', color=INK2)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)



def classification(df, feats, out):
    """Can the geometry name the receptor / subfamily / family of a structure it has never seen?"""
    X = df[feats].to_numpy()
    targets = [('NR family', 'family'), ('NR subfamily', 'nr_subfamily'), ('receptor', RECEPTOR)]
    if df[MOTIF].nunique() > 1:
        targets.insert(0, ('motif type', MOTIF))
    rows, preds = [], {}
    for name, col in targets:
        y = df[col].astype(str)
        keep = (y.map(y.value_counts()) >= MIN_CLASS).to_numpy()
        rf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
        p = cross_val_predict(rf, X[keep], y[keep], groups=df.pdb_id[keep], cv=GroupKFold(5))
        preds[name] = (y[keep].to_numpy(), p)
        rows.append({'target': name, 'classes': y[keep].nunique(), 'units': int(keep.sum()),
                     'left_out_units': int((~keep).sum()),
                     'balanced_accuracy': balanced_accuracy_score(y[keep], p), 'chance': 1 / y[keep].nunique()})
    scores = pd.DataFrame(rows)
    scores['times_chance'] = scores.balanced_accuracy / scores.chance
    scores.to_csv(out / 'classification_scores.csv', index=False)

    # importance for the NR-family classifier, measured on held-out PDB entries (permutation: accuracy lost when
    # one descriptor is shuffled)
    y = df['family'].astype(str)
    keep = (y.map(y.value_counts()) >= MIN_CLASS).to_numpy()
    Xk, yk, gk = X[keep], y[keep].reset_index(drop=True), df.pdb_id[keep]
    imp = np.zeros(len(feats))
    for tr, te in GroupKFold(5).split(Xk, yk, gk):
        rf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1).fit(Xk[tr], yk.iloc[tr])
        imp += permutation_importance(rf, Xk[te], yk.iloc[te], scoring='balanced_accuracy',
                                      n_repeats=5, random_state=SEED).importances_mean / 5
    importance = pd.DataFrame({'descriptor': feats, 'family_accuracy_lost_when_shuffled': imp}
                              ).sort_values('family_accuracy_lost_when_shuffled', ascending=False)
    importance.to_csv(out / 'classification_importance.csv', index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={'width_ratios': [1, 1.1]})
    yy = np.arange(len(scores))[::-1]
    a1.barh(yy + 0.14, scores.balanced_accuracy, height=0.28, color=SERIES[0], label='geometry (held-out PDB entries)')
    a1.barh(yy - 0.14, scores.chance, height=0.28, color=OTHER, label='chance (random guess)')
    for yi, (_, r) in zip(yy, scores.iterrows()):
        a1.text(r.balanced_accuracy + 0.015, yi + 0.14, f'{r.balanced_accuracy:.2f}  ({r.times_chance:.0f}x chance)',
                va='center', fontsize=8, color=INK)
    a1.set_yticks(yy, [f'{t}\n({c} classes)' for t, c in zip(scores.target, scores.classes)])
    a1.set_xlim(0, 1.3)
    a1.set_xticks([0, .25, .5, .75, 1])
    a1.set_xlabel('balanced accuracy (1 = always right)')
    a1.legend(loc='lower right', fontsize=7, frameon=False)
    a1.set_title('Does the geometry tell them apart?', loc='left')
    imp_plot = importance.iloc[::-1]
    a2.barh(imp_plot.descriptor, imp_plot.family_accuracy_lost_when_shuffled, height=0.5, color=SERIES[0])
    a2.set_xlabel('accuracy lost when this descriptor is shuffled')
    a2.set_title('Which descriptors separate the NR families', loc='left')
    a2.tick_params(axis='y', labelsize=8)
    fig.suptitle(f'Classification ({len(df)} units, random forest, 5-fold CV with whole PDB entries held out)',
                 x=0.01, ha='left', color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'classification_summary.png', dpi=150)
    plt.close(fig)

    true, pred = preds['receptor']
    order = (df[[RECEPTOR, 'nr_subfamily']].astype(str).drop_duplicates(RECEPTOR)
             .sort_values(['nr_subfamily', RECEPTOR]))
    names = [r for r in order[RECEPTOR] if r in set(true)]
    cm = confusion_matrix(true, pred, labels=names)
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                        color='white' if share[i, j] > 0.6 else INK)
    sub = order.set_index(RECEPTOR).nr_subfamily
    labels = [f'{n} ({sub[n]})' for n in names]
    ax.set_xticks(range(len(names)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), [f'{l}  {share[i, i]:.0%}' for i, l in enumerate(labels)], fontsize=8)
    for b in np.flatnonzero(np.array([sub[a] != sub[c] for a, c in zip(names[:-1], names[1:])])):
        ax.axhline(b + 0.5, color=INK2, lw=0.6)
        ax.axvline(b + 0.5, color=INK2, lw=0.6)
    ax.grid(False)
    ax.set_xlabel('predicted receptor')
    ax.set_ylabel('true receptor (share predicted correctly)')
    ax.set_title('Which receptors look alike? Counts of held-out units; lines separate NR subfamilies',
                 loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'classification_confusion.png', dpi=150)
    plt.close(fig)

    print(scores.round(3).to_string(index=False))
    print(importance.round(3).to_string(index=False))


def analyse(df, all_feats, out):
    out.mkdir(parents=True, exist_ok=True)
    feats, dropped = prune(df, all_feats)
    types = pd.Series([kind(f) for f in all_feats]).value_counts().to_dict()
    print(f'units: {len(df)} | receptors: {df[RECEPTOR].nunique()} | descriptors: {len(all_feats)} {types}')
    for f, twin, r in dropped:
        print(f'  dropped {f} (r = {r:.2f} with {twin})')
    print(f'descriptors used for clustering: {len(feats)}')

    desc = describe(df, all_feats)
    desc.to_csv(out / 'descriptor_ranking.csv', index=False)
    descriptor_figures(df, desc, all_feats, out)

    X = StandardScaler().fit_transform(df[feats])
    scan, best_k = scan_k(X)
    scan.to_csv(out / 'k_scan.csv', index=False)
    k_scan_figure(scan[~scan.method.str.startswith('Gaussian')], out)
    print('chosen k:', best_k)
    res, labels = benchmark(df, X, best_k)
    res.to_csv(out / 'benchmark_metrics.csv', index=False)
    metrics_figure(res, out)
    top3_figure(df, res, labels, out)
    best = res.method.iloc[0]
    print(f'best: {best}')

    lab = labels[best]
    table = df[['pdb_id', 'receptor_chain', 'peptide_chain', RECEPTOR, 'family', MOTIF]].copy()
    for name, l in labels.items():
        table[name] = l
    table.to_csv(out / 'cluster_labels_all_methods.csv', index=False)
    drv = drivers(df, X, feats, lab)
    drv.to_csv(out / 'cluster_drivers.csv', index=False)
    drivers_figure(drv, best, out)
    composition_figure(df, lab, best, out)

    pca = PCA(2, random_state=SEED).fit(X)
    var = pca.explained_variance_ratio_ * 100
    embedding_figure(df, pca.transform(X), lab, best, (f'PC1 ({var[0]:.0f}%)', f'PC2 ({var[1]:.0f}%)'),
                     'map_pca.png', out)
    receptor_panels(df, pca.transform(X), (f'PC1 ({var[0]:.0f}%)', f'PC2 ({var[1]:.0f}%)'), 'receptors_map_pca.png', out)
    methods_figure(df, pca.transform(X), labels, res, (f'PC1 ({var[0]:.0f}%)', f'PC2 ({var[1]:.0f}%)'),
                   'methods_map_pca.png', out)
    tsne = TSNE(2, perplexity=30, random_state=SEED).fit_transform(X)
    embedding_figure(df, tsne, lab, best, ('t-SNE 1', 't-SNE 2'), 'map_tsne.png', out)
    receptor_panels(df, tsne, ('t-SNE 1', 't-SNE 2'), 'receptors_map_tsne.png', out)
    methods_figure(df, tsne, labels, res, ('t-SNE 1', 't-SNE 2'), 'methods_map_tsne.png', out)

    pd.set_option('display.width', 250)
    print(res[['method', 'clusters', 'noise_fraction', 'largest_cluster_fraction', 'smallest_cluster_fraction',
               'silhouette', 'davies_bouldin', 'stability_ARI', 'ARI_receptor', 'ARI_family', 'ARI_motif',
               'mean_rank', 'degenerate']].round(3).to_string(index=False))
    print(drv[['descriptor', 'eta2_between_clusters', 'random_forest_importance']].round(3).to_string(index=False))
    classification(df, feats, out)


def main():
    style()
    df, feats = load()
    for name, motif in SUBSETS.items():
        print(f'\n===== {name} =====')
        sub = df if motif is None else df[df[MOTIF] == motif].reset_index(drop=True)
        if 'classify' in sys.argv[1:]:
            (OUT / name).mkdir(parents=True, exist_ok=True)
            classification(sub, prune(sub, feats)[0], OUT / name)
        else:
            analyse(sub, feats, OUT / name)


if __name__ == '__main__':
    main()
