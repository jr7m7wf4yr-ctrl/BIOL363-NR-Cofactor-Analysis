"""Each clustering method drawn in its own view (run clustering_benchmark.py first; add `bota` for Bota's table).

A clustering method only assigns units to groups, so on a shared PCA/t-SNE map every method looks the same apart from
colour. Here each method is shown with what it works with: spectral = its graph embedding, SOM = its neuron grid,
Ward = dendrogram, Gaussian mixture = fitted ellipses, HDBSCAN = density valleys, grid = dense cells, K-means = centroid
profiles, BIRCH = subcluster centres, affinity propagation = exemplars, mean shift = density peaks.

Outputs per subset:
  method_views.png       all 10 views in one figure; cluster colour = its majority receptor, so panels compare
  per_method/NN_*.png    one figure per method: its view + what is inside each cluster (receptor make-up);
                         cluster colour = the cluster itself (7 largest in colour, smaller clusters and noise grey)
  per_method/NN_*_map.png  the same clusters on the PCA map in Bota's style: points coloured by NR subfamily
                         (hue = family, shade = subfamily), dashed outline = cluster, × = noise, ring = corepressor
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch, Rectangle
from scipy import ndimage
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, to_tree
from scipy.stats import gaussian_kde
from sklearn.cluster import (HDBSCAN, OPTICS, AffinityPropagation, Birch, KMeans, MeanShift, SpectralClustering,
                             estimate_bandwidth)
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from clustering_benchmark import (COREPRESSOR, INK2, MOTIF, MUTED, OTHER, OUT, RECEPTOR, SEED, SERIES, SUBSETS,
                                  SURFACE, load, prune, som, style)

SHORT = {'centroid_to_motif_A': 'pocket→motif', 'n_contact_residues': 'contacts',
         'h12_center_to_centroid_A': 'H12→pocket', 'h12_center_to_motif_A': 'H12→motif',
         'h12_axisline_to_centroid_A': 'H12 axis→pocket', 'h12_axisline_to_motif_A': 'H12 axis→motif',
         'h12_vs_centroid_motif_deg': 'H12 angle (pocket)', 'h12_vs_motif_axis_deg': 'H12 angle (motif)',
         'h12core_center_to_centroid_A': 'core→pocket', 'h12core_center_to_motif_A': 'core→motif',
         'h12core_axisline_to_centroid_A': 'core axis→pocket', 'h12core_axisline_to_motif_A': 'core axis→motif',
         'h12core_vs_centroid_motif_deg': 'core angle (pocket)', 'h12core_vs_motif_axis_deg': 'core angle (motif)'}
METHODS = ['Spectral (graph-based)', 'SOM (neural)', 'Ward (hierarchical)', 'K-means (partitioning)',
           'Gaussian mixture (model-based)', 'HDBSCAN (density)', 'Grid (grid-based, PC1-PC2)', 'BIRCH (tree / CF)',
           'Affinity propagation (exemplar)', 'Mean shift (mode-seeking)']


class Colours:
    """mode 'majority': cluster colour = its majority receptor's colour (> 50%, top 7 receptors; else grey).
    mode 'cluster': each cluster its own colour, by size (7 largest; smaller clusters and noise grey)."""

    def __init__(self, receptors, mode='majority'):
        self.rec = np.asarray(receptors)
        self.mode = mode
        self.palette = {r: SERIES[i] for i, r in enumerate(pd.Series(self.rec).value_counts().index[:7])}

    def majority(self, members):
        counts = pd.Series(self.rec[members]).value_counts()
        if counts.empty:
            return None, 0
        return counts.index[0], counts.iloc[0] / counts.sum()

    def of_cluster(self, labels, c):
        if c == -1:
            return OTHER
        if self.mode == 'cluster':
            order = list(pd.Series(labels[labels != -1]).value_counts().index)
            return SERIES[order.index(c)] if order.index(c) < 7 else OTHER
        r, share = self.majority(labels == c)
        return self.palette.get(r, OTHER) if share > 0.5 else OTHER

    def per_unit(self, labels):
        cmap = {c: self.of_cluster(labels, c) for c in set(labels)}
        return np.array([cmap[c] for c in labels])

    def tag(self, members):
        r, share = self.majority(members)
        return f'{r} {share:.0%}'


def faint(ax, p):
    ax.scatter(*p.T, s=5, c=OTHER, linewidths=0, alpha=0.6)


def pc_axes(ax, var):
    ax.set_xlabel(f'PC1 ({var[0]:.0f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.0f}%)')


# ---------- one view per method; each returns the cluster label of every unit ----------

def kmeans_view(ax, X, k, col, feats, **_):
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    x = np.arange(len(feats))
    for c in range(k):
        ax.plot(x, km.cluster_centers_[c], lw=2, marker='o', ms=4, color=col.of_cluster(km.labels_, c),
                label=f'{c}: {col.tag(km.labels_ == c)} (n={np.sum(km.labels_ == c)})')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(f, f) for f in feats], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('centroid (z-score)')
    ax.legend(fontsize=6, frameon=False, ncol=2, loc='lower left')
    ax.set_title(f'K-means (partitioning): centroid profiles, k = {k}')
    return km.labels_


def ward_view(ax, X, k, col, **_):
    Z = linkage(X, 'ward')
    lab = fcluster(Z, k, 'maxclust')
    nodes = to_tree(Z, rd=True)[1]
    leaf_cluster = {}

    def label(i):
        members = nodes[i].pre_order()
        leaf_cluster[i] = lab[members[0]]
        mask = np.zeros(len(X), bool)
        mask[members] = True
        return f'{col.tag(mask)}\nn={mask.sum()}'

    leaves = max(k, 16)       # show the tree deeper than the cut; every leaf lies inside one cluster
    d = dendrogram(Z, truncate_mode='lastp', p=leaves, leaf_label_func=label, ax=ax, link_color_func=lambda _: INK2,
                   leaf_font_size=6)
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2 if k > 1 else Z[-1, 2]
    ax.axhline(cut, color=MUTED, ls='--', lw=1)
    ax.text(ax.get_xlim()[1], cut, f' cut → {k} clusters', va='bottom', ha='right', fontsize=7, color=MUTED)
    for tick, i in zip(ax.get_xticklabels(), d['leaves']):
        tick.set_color(INK2)
        ax.plot(tick.get_position()[0], 0, 's', ms=9, color=col.of_cluster(lab, leaf_cluster[i]), clip_on=False)
    ax.set_ylabel('merge distance')
    ax.grid(False)
    ax.set_title(f'Ward (hierarchical): top {leaves} branches; square = cluster after the cut (k = {k})')
    return lab


def spectral_view(ax, X, k, col, **_):
    lab = SpectralClustering(k, affinity='nearest_neighbors', n_neighbors=15, random_state=SEED).fit_predict(X)
    emb = SpectralEmbedding(2, affinity='nearest_neighbors', n_neighbors=15, random_state=SEED).fit_transform(X)
    ax.scatter(*emb.T, s=9, c=col.per_unit(lab), linewidths=0.2, edgecolors=SURFACE)
    ax.set_xlabel('graph eigenvector 1')
    ax.set_ylabel('graph eigenvector 2')
    ax.set_title(f'Spectral (graph-based): its own graph map, k = {k}\n(the first 2 eigenvectors: the most '
                 f'separate groups form arms)', fontsize=9)
    return lab


def som_view(ax, X, k, col, rows=8, cols=8, **_):
    w = som(X, rows, cols)
    node_lab = fcluster(linkage(w, 'ward'), k, 'maxclust')
    bmu = np.argmin(((X[:, None, :] - w[None]) ** 2).sum(2), 1)
    unit_lab = node_lab[bmu]
    hits = np.bincount(bmu, minlength=rows * cols)
    for n in range(rows * cols):
        i, j = divmod(n, cols)
        colour = col.of_cluster(unit_lab, node_lab[n]) if hits[n] else OTHER
        ax.add_patch(Rectangle((j, rows - 1 - i), 0.94, 0.94, color=colour,
                               alpha=0.25 + 0.75 * hits[n] / hits.max()))
        if hits[n]:
            r, _ = col.majority(bmu == n)
            ax.text(j + 0.47, rows - 1 - i + 0.47, f'{r}\n{hits[n]}', ha='center', va='center', fontsize=5.5,
                    color='#0b0b0b')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'SOM (neural): its own 8×8 neuron map, k = {k}\n(cell = neuron: majority receptor, units; darker = more)',
                 fontsize=9)
    return unit_lab


def gmm_view(ax, X, k, col, pca, p, var, **_):
    g = GaussianMixture(k, covariance_type='full', n_init=3, random_state=SEED).fit(X)
    lab = g.predict(X)
    ax.scatter(*p.T, s=6, c=col.per_unit(lab), linewidths=0, alpha=0.45)
    W = pca.components_
    for c in range(k):
        mean = pca.transform(g.means_[c:c + 1])[0]
        vals, vecs = np.linalg.eigh(W @ g.covariances_[c] @ W.T)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        colour = col.of_cluster(lab, c)
        ax.add_patch(Ellipse(mean, *(2 * 2 * np.sqrt(vals[::-1])), angle=angle, fill=False, lw=2, color=colour))
        ax.plot(*mean, 'o', color=colour, ms=5)
    pc_axes(ax, var)
    ax.set_title(f'Gaussian mixture (distribution-based): {k} fitted Gaussians (2σ)')
    return lab


def hdbscan_view(ax, X, col, **_):
    lab = HDBSCAN(min_cluster_size=max(10, len(X) // 100)).fit_predict(X)
    op = OPTICS(min_samples=max(10, len(X) // 100)).fit(X)
    reach = op.reachability_[op.ordering_]
    finite = np.isfinite(reach)
    reach[~finite] = np.nanmax(reach[finite])
    ax.bar(np.arange(len(reach)), reach, width=1.0, color=col.per_unit(lab)[op.ordering_], linewidth=0)
    ax.set_xlabel('units in density order')
    ax.set_ylabel('reachability distance')
    ax.grid(False)
    ax.set_title(f'HDBSCAN (density): valleys = dense clusters\n{(lab == -1).mean():.0%} noise (grey)', fontsize=9)
    return lab


def grid_view(ax, col, p, var, bins=20, min_pts=5, **_):
    edges = [np.linspace(p[:, i].min(), p[:, i].max() + 1e-9, bins + 1) for i in range(2)]
    counts, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=edges)
    cells, _ = ndimage.label(counts >= min_pts, structure=np.ones((3, 3)))
    ax.imshow(counts.T, origin='lower', extent=(*edges[0][[0, -1]], *edges[1][[0, -1]]), aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#fcfcfb', '#86b6ef', '#1c5cab']))
    ix = [np.clip(np.digitize(p[:, i], edges[i]) - 1, 0, bins - 1) for i in range(2)]
    unit_lab = np.where(cells[ix[0], ix[1]] > 0, cells[ix[0], ix[1]], -1)
    for i in range(bins):
        for j in range(bins):
            if cells[i, j]:
                ax.add_patch(Rectangle((edges[0][i], edges[1][j]), edges[0][1] - edges[0][0],
                                       edges[1][1] - edges[1][0], fill=False, lw=1.4,
                                       color=col.of_cluster(unit_lab, cells[i, j])))
    ax.grid(False)
    pc_axes(ax, var)
    ax.set_title(f'Grid-based: units per cell; outlined = dense cells\n{cells.max()} connected regions', fontsize=9)
    return unit_lab


def birch_view(ax, X, k, col, pca, p, var, **_):
    b = Birch(n_clusters=k).fit(X)
    lab = b.predict(X)
    nearest = np.argmin(((X[:, None, :] - b.subcluster_centers_[None]) ** 2).sum(2), 1)
    size = np.bincount(nearest, minlength=len(b.subcluster_centers_))
    centres = pca.transform(b.subcluster_centers_)
    faint(ax, p)
    ax.scatter(*centres.T, s=8 + size * 3, c=[col.of_cluster(lab, c) for c in b.subcluster_labels_], alpha=0.75,
               linewidths=0.8, edgecolors=SURFACE)
    pc_axes(ax, var)
    ax.set_title(f'BIRCH (tree / CF): {len(centres)} subcluster centres → {k} clusters\n(circle size = units)',
                 fontsize=9)
    return lab


def affinity_view(ax, X, col, p, var, **_):
    ap = AffinityPropagation(preference=-50, damping=0.9, max_iter=500, random_state=SEED).fit(X)
    lab = ap.labels_
    ex = ap.cluster_centers_indices_
    colours = col.per_unit(lab)
    for i in range(len(X)):
        e = ex[lab[i]]
        ax.plot([p[i, 0], p[e, 0]], [p[i, 1], p[e, 1]], lw=0.4, color=colours[i], alpha=0.35)
    ax.scatter(*p[ex].T, s=60, marker='*', c=colours[ex], linewidths=0.6, edgecolors='#0b0b0b', zorder=3)
    pc_axes(ax, var)
    ax.set_title(f'Affinity propagation (exemplar): {len(ex)} exemplars (★)\nlines = unit → its exemplar',
                 fontsize=9)
    return lab


def meanshift_view(ax, X, col, pca, p, var, **_):
    ms = MeanShift(bandwidth=estimate_bandwidth(X, quantile=0.2, random_state=SEED), bin_seeding=True).fit(X)
    lab = ms.labels_
    kde = gaussian_kde(p.T)
    gx, gy = np.meshgrid(np.linspace(p[:, 0].min(), p[:, 0].max(), 120), np.linspace(p[:, 1].min(), p[:, 1].max(), 120))
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=10, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'b', ['#fcfcfb', '#b7d3f6', '#3987e5', '#104281']))
    centres = pca.transform(ms.cluster_centers_)
    sizes = np.bincount(lab)
    ax.scatter(*centres.T, s=30 + sizes / sizes.max() * 300, c=[col.of_cluster(lab, c) for c in range(len(centres))],
               marker='^', linewidths=1, edgecolors='#0b0b0b', zorder=3)
    ax.grid(False)
    pc_axes(ax, var)
    ax.set_title(f'Mean shift (mode-seeking): density peaks (▲)\n{len(centres)} modes; size = units', fontsize=9)
    return lab


VIEWS = {'Spectral (graph-based)': spectral_view, 'SOM (neural)': som_view, 'Ward (hierarchical)': ward_view,
         'K-means (partitioning)': kmeans_view, 'Gaussian mixture (model-based)': gmm_view,
         'HDBSCAN (density)': hdbscan_view, 'Grid (grid-based, PC1-PC2)': grid_view, 'BIRCH (tree / CF)': birch_view,
         'Affinity propagation (exemplar)': affinity_view, 'Mean shift (mode-seeking)': meanshift_view}


def composition(ax, lab, col, receptors, top=14):
    """What is inside each cluster: for each receptor, the share of its units in each cluster (as in Bota's plot)."""
    rec = pd.Series(receptors)
    order = rec.value_counts().index[:top]
    clusters = [c for c in pd.Series(lab[lab != -1]).value_counts().index] + ([-1] if (lab == -1).any() else [])
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for c in clusters:
        share = np.array([np.mean(lab[(rec == r).values] == c) * 100 for r in order])
        ax.barh(y, share, left=left, color=col.of_cluster(lab, c), height=0.7, edgecolor=SURFACE, linewidth=1)
        left += share
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r} (n={(rec == r).sum()})' for r in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the receptor's units in each cluster")
    ax.set_title(f'What is inside each cluster ({top} largest receptors)')
    shown = [c for c in clusters if col.of_cluster(lab, c) != OTHER]
    handles = [Patch(color=col.of_cluster(lab, c), label=f'cluster {c}: n={np.sum(lab == c)}, {col.tag(lab == c)}')
               for c in shown]
    if len(shown) < len(clusters):
        handles.append(Patch(color=OTHER, label='smaller clusters / noise'))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)


FAMILY_RAMP = {'NR1': 'YlOrRd', 'NR2': 'Blues', 'NR3': 'Greens', 'NR4': 'Greys', 'NR5': 'Purples', 'NR6': 'Greys'}


# hand-picked so subfamilies of one family stay distinguishable (NR1 warm, NR2 blue, NR3 green, NR5 purple)
SUBFAMILY_COLOUR = {'NR1A': '#8c2d04', 'NR1B': '#d6604d', 'NR1C': '#f4a582', 'NR1D': '#b35806', 'NR1F': '#f0a30a',
                    'NR1H': '#e7298a', 'NR1I': '#a6761d', 'NR2A': '#6baed6', 'NR2B': '#2166ac', 'NR2C': '#08306b',
                    'NR2F': '#4292c6', 'NR3A': '#4daf4a', 'NR3B': '#a6d96a', 'NR3C': '#1b5e20', 'NR4A': '#737373',
                    'NR5A': '#984ea3', 'NR6A': '#525252'}


def subfamily_colours(codes):
    """Hue by NR family, shade by subfamily (as in Bota's plots); unknown subfamilies get a shade of their family ramp."""
    out = {c: SUBFAMILY_COLOUR[c] for c in set(codes) if c in SUBFAMILY_COLOUR}
    subs = sorted(set(codes))
    for fam, cmap in FAMILY_RAMP.items():
        mine = [c for c in subs if c.startswith(fam) and c not in out]
        for i, c in enumerate(mine):
            out[c] = plt.get_cmap(cmap)(0.5 + 0.4 * i / max(len(mine) - 1, 1))
    return out


def label_point(pts, lo, hi):
    """Top point of the cluster that is inside the zoomed view (else its median)."""
    inside = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    return inside[inside[:, 1].argmax()] if len(inside) else np.median(pts, axis=0)


def family_map(ax, p, var, lab, sub, method, max_boxes=8):
    """Bota-style map: PCA points by NR subfamily, dashed outline per cluster with its main receptors, × = noise."""
    from scipy.spatial import ConvexHull
    codes = sub.nr_subfamily.values
    pal = subfamily_colours(codes)
    corep = (sub[MOTIF] == COREPRESSOR).values
    noise = lab == -1
    sizes = pd.Series(lab[~noise]).value_counts()
    lo, hi = np.quantile(p, [0.003, 0.997], axis=0)
    for rank, (c, n) in enumerate(sizes.items()):
        pts = p[lab == c]
        if n >= 3:
            try:
                hull = pts[ConvexHull(pts).vertices]
            except Exception:       # all points on a line
                hull = pts
            ax.fill(*np.vstack([hull, hull[:1]]).T, facecolor='#eeeeec', edgecolor='#3a3a38', ls='--', lw=1,
                    zorder=0, alpha=0.7)
        if rank < max_boxes:
            top = sub.loc[lab == c, RECEPTOR].value_counts().head(3)
            text = f'cluster {c} (n={n})\n' + ', '.join(f'{r} {v}' for r, v in top.items())
            ax.annotate(text, label_point(pts, lo, hi), xytext=(0, 8), textcoords='offset points', ha='center',
                        fontsize=7.5, fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3a3a38', lw=0.8))
    colours = np.array([pal[c] for c in codes])
    ok = ~noise
    ax.scatter(*p[ok & ~corep].T, s=14, c=colours[ok & ~corep], linewidths=0.3, edgecolors='white', zorder=2)
    ax.scatter(*p[ok & corep].T, s=40, c=colours[ok & corep], linewidths=1.2, edgecolors='#0b0b0b', zorder=3)
    ax.scatter(*p[noise].T, s=60, c=colours[noise], marker='X', linewidths=0.8, edgecolors='#0b0b0b', zorder=4)
    handles = []
    for code in sorted(pal):
        names = ', '.join(sub.loc[codes == code, RECEPTOR].value_counts().index[:3])
        handles.append(plt.Line2D([], [], marker='o', ls='', color=pal[code], ms=6,
                                  label=f'{code}  {names}  (n={np.sum(codes == code)})'))
    handles.append(plt.Line2D([], [], marker='X', ls='', color='#d6d5cf', mec='#0b0b0b', ms=8,
                              label=f'× = noise (n={noise.sum()})'))
    handles.append(plt.Line2D([], [], marker='o', ls='', mfc='white', mec='#0b0b0b', ms=8,
                              label='black ring = corepressor-bound'))
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1),
              title='NR subfamily (hue = family: NR1 red/orange, NR2 blue, NR3 green, NR5 purple)', title_fontsize=8)
    pc_axes(ax, var)
    # a few far-away units squeeze the cloud into a line: zoom to the central 99.4% of units on each axis
    lo, hi = np.quantile(p, [0.003, 0.997], axis=0)
    pad = (hi - lo) * 0.06
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    outside = int(np.sum((p < lo - pad) | (p > hi + pad), axis=1).astype(bool).sum())
    more = f'; boxes for the {max_boxes} largest of {len(sizes)} clusters' if len(sizes) > max_boxes else ''
    more += f'; {outside} far-away units outside the zoomed view' if outside else ''
    ax.set_title(f'{method}: {len(sizes)} clusters of {len(sub)} units, coloured by NR subfamily\n'
                 f'dashed outline = cluster; × = noise{more}', fontsize=10)


def main():
    style()
    df, all_feats = load()
    for name, motif in SUBSETS.items():
        sub = df if motif is None else df[df[MOTIF] == motif].reset_index(drop=True)
        feats, _ = prune(sub, all_feats)
        X = StandardScaler().fit_transform(sub[feats])
        res = pd.read_csv(OUT / name / 'benchmark_metrics.csv').set_index('method')
        pca = PCA(2, random_state=SEED).fit(X)
        common = dict(X=X, pca=pca, p=pca.transform(X), var=pca.explained_variance_ratio_ * 100, feats=feats)

        def args(m):
            return dict(common, k=int(res.loc[m, 'clusters']))

        col = Colours(sub[RECEPTOR])
        fig, axes = plt.subplots(3, 4, figsize=(21, 15))
        for ax, m in zip(axes.flat, METHODS):
            VIEWS[m](ax, col=col, **args(m))
        ax = axes.flat[10]
        for r, c in col.palette.items():
            ax.scatter([], [], s=60, c=c, label=f'{r} ({np.sum(col.rec == r)})')
        ax.scatter([], [], s=60, c=OTHER, label='other receptor or mixed cluster')
        ax.legend(loc='center', frameon=False, fontsize=10, title='Cluster colour = majority receptor',
                  title_fontsize=10)
        ax.axis('off')
        axes.flat[11].axis('off')
        fig.suptitle(f'Each clustering method in its own view ({name}, {len(sub)} units)', x=0.01, ha='left',
                     fontsize=13, color=INK2)
        fig.tight_layout()
        fig.savefig(OUT / name / 'method_views.png', dpi=140)
        plt.close(fig)

        per = OUT / name / 'per_method'
        per.mkdir(exist_ok=True)
        for old in per.glob('*.png'):       # ranks change between runs, so old file names would linger
            old.unlink()
        own = Colours(sub[RECEPTOR], mode='cluster')
        ranked = [m for m in res.index if m in VIEWS]
        for m in METHODS:
            fig, (a, b) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={'width_ratios': [1.15, 1]})
            lab = np.asarray(VIEWS[m](a, col=own, **args(m)))
            composition(b, lab, own, sub[RECEPTOR].values)
            r = res.loc[m]
            flag = ' · degenerate' if r.degenerate else ''
            fig.suptitle(f'{m} · {len(sub)} units ({name}) · rank {ranked.index(m) + 1} of {len(ranked)}{flag}\n'
                         f'{int(r.clusters)} clusters · silhouette {r.silhouette:.2f} · stability {r.stability_ARI:.2f}'
                         f' · match with receptor (ARI) {r.ARI_receptor:.2f}', x=0.01, ha='left', fontsize=11,
                         color=INK2)
            fig.tight_layout()
            fig.savefig(per / f'{ranked.index(m) + 1:02d}_{m.split(" (")[0].replace(" ", "_")}.png', dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(15, 8))
            family_map(ax, common['p'], common['var'], lab, sub, m)
            fig.tight_layout()
            fig.savefig(per / f'{ranked.index(m) + 1:02d}_{m.split(" (")[0].replace(" ", "_")}_map.png', dpi=150)
            plt.close(fig)
        print('saved', OUT / name, flush=True)


if __name__ == '__main__':
    main()
