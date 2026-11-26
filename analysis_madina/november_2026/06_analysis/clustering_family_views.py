"""The 5 extra clustering methods (SOM, grid, BIRCH, affinity propagation, mean shift), each in its own view but with
units coloured by NR subfamily (as in Bota's plots: hue = family, shade = subfamily).
Run clustering_benchmark.py first (add `bota` for Bota's table); k is the one the benchmark chose.

Left: the method's own view, coloured by subfamily; cluster borders drawn in black.
Right: what each cluster is made of (share of each NR subfamily).
Output: <OUT>/<subset>/family_colour/<method>.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.cluster import AffinityPropagation, Birch, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clustering_benchmark import INK2, MOTIF, MUTED, OUT, RECEPTOR, SEED, SUBSETS, load, prune, som, style
from clustering_method_views import subfamily_colours

INK = '#0b0b0b'


def zoom(ax, p, var):
    """Central 99.4% of units, so a few far-away units do not squeeze the cloud."""
    lo, hi = np.quantile(p, [0.003, 0.997], axis=0)
    pad = (hi - lo) * 0.06
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel(f'PC1 ({var[0]:.0f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.0f}%)')
    return int(np.any((p < lo - pad) | (p > hi + pad), axis=1).sum())


def points(ax, p, codes, pal, s=10, alpha=0.85):
    ax.scatter(*p.T, s=s, c=[pal[c] for c in codes], linewidths=0.2, edgecolors='white', alpha=alpha, zorder=2)


def hull(ax, pts, label=None):
    if len(pts) >= 3:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:       # all points on a line
            pass
        ax.plot(*np.vstack([pts, pts[:1]]).T, ls='--', lw=1, color=INK, zorder=3)
    if label is not None:
        ax.annotate(label, pts[pts[:, 1].argmax()], xytext=(0, 6), textcoords='offset points', ha='center',
                    fontsize=7, fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=INK, lw=0.6))


def major(codes):
    return pd.Series(codes).value_counts().index[0]


# ---------- the 5 views; each returns the cluster label of every unit ----------

def som_family(ax, X, k, codes, pal, rows=8, cols=8, **_):
    w = som(X, rows, cols)
    node_lab = fcluster(linkage(w, 'ward'), k, 'maxclust')
    bmu = np.argmin(((X[:, None, :] - w[None]) ** 2).sum(2), 1)
    hits = np.bincount(bmu, minlength=rows * cols)
    rec = _['receptors']
    for n in range(rows * cols):
        i, j = divmod(n, cols)
        x, y = j, rows - 1 - i
        if hits[n]:
            ax.add_patch(Rectangle((x, y), 1, 1, color=pal[major(codes[bmu == n])],
                                   alpha=0.35 + 0.65 * hits[n] / hits.max(), linewidth=0))
            ax.text(x + 0.5, y + 0.5, f'{pd.Series(rec[bmu == n]).value_counts().index[0]}\n{hits[n]}', ha='center',
                    va='center', fontsize=5.5, color=INK)
        else:
            ax.add_patch(Rectangle((x, y), 1, 1, color='#f1f0ec', linewidth=0))
    for n in range(rows * cols):          # thick black line between neighbouring neurons of different clusters
        i, j = divmod(n, cols)
        if j + 1 < cols and node_lab[n] != node_lab[n + 1]:
            ax.plot([j + 1, j + 1], [rows - 1 - i, rows - i], color=INK, lw=2.2)
        if i + 1 < rows and node_lab[n] != node_lab[n + cols]:
            ax.plot([j, j + 1], [rows - 1 - i, rows - 1 - i], color=INK, lw=2.2)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'SOM (neural): 8×8 neuron map, {k} clusters (black lines = cluster borders)\n'
                 'cell colour = main NR subfamily of its units; label = main receptor, units; darker = more units',
                 fontsize=9)
    return node_lab[bmu]


def grid_family(ax, p, codes, pal, var, bins=20, min_pts=5, **_):
    edges = [np.linspace(p[:, i].min(), p[:, i].max() + 1e-9, bins + 1) for i in range(2)]
    counts, _e, _f = np.histogram2d(p[:, 0], p[:, 1], bins=edges)
    cells, n = ndimage.label(counts >= min_pts, structure=np.ones((3, 3)))
    ix = [np.clip(np.digitize(p[:, i], edges[i]) - 1, 0, bins - 1) for i in range(2)]
    lab = np.where(cells[ix[0], ix[1]] > 0, cells[ix[0], ix[1]], -1)
    dx, dy = edges[0][1] - edges[0][0], edges[1][1] - edges[1][0]
    for i in range(bins):
        for j in range(bins):
            if cells[i, j]:
                ax.add_patch(Rectangle((edges[0][i], edges[1][j]), dx, dy, facecolor='#eeeeec', edgecolor='#b5b4ad',
                                       lw=0.6, zorder=0))
    points(ax, p, codes, pal)
    for c in range(1, n + 1):
        members = lab == c
        hull(ax, p[members], f'cluster {c} (n={members.sum()})')
    noise = lab == -1
    ax.scatter(*p[noise].T, s=30, marker='x', color=INK, lw=0.8, zorder=4)
    out = zoom(ax, p, var)
    ax.set_title(f'Grid-based: {n} dense regions (grey cells = dense, ≥ {min_pts} units)\n× = units in sparse cells '
                 f'(noise, {noise.sum()}); {out} far-away units outside the view', fontsize=9)
    return lab


def birch_family(ax, X, k, codes, pal, pca, p, var, **_):
    b = Birch(n_clusters=k).fit(X)
    lab = b.predict(X)
    nearest = np.argmin(((X[:, None, :] - b.subcluster_centers_[None]) ** 2).sum(2), 1)
    size = np.bincount(nearest, minlength=len(b.subcluster_centers_))
    centres = pca.transform(b.subcluster_centers_)
    used = size > 0
    ax.scatter(*p.T, s=4, c='#d6d5cf', linewidths=0, zorder=1)
    ax.scatter(*centres[used].T, s=10 + size[used] * 4,
               c=[pal[major(codes[nearest == s])] for s in np.flatnonzero(used)], alpha=0.8, linewidths=0.6,
               edgecolors='white', zorder=2)
    for c in np.unique(lab):
        members = lab == c
        hull(ax, p[members], f'cluster {c} (n={members.sum()})')
    out = zoom(ax, p, var)
    ax.set_title(f'BIRCH (tree / CF): {used.sum()} subclusters (circles) merged into {k} clusters (dashed)\n'
                 f'circle colour = main NR subfamily, size = units; {out} far-away units outside the view', fontsize=9)
    return lab


def affinity_family(ax, X, codes, pal, p, var, **_):
    ap = AffinityPropagation(preference=-50, damping=0.9, max_iter=500, random_state=SEED).fit(X)
    lab, ex = ap.labels_, ap.cluster_centers_indices_
    for i in range(len(X)):
        e = ex[lab[i]]
        ax.plot([p[i, 0], p[e, 0]], [p[i, 1], p[e, 1]], lw=0.35, color='#b5b4ad', alpha=0.6, zorder=1)
    points(ax, p, codes, pal, s=9)
    ax.scatter(*p[ex].T, s=150, marker='*', c=[pal[codes[e]] for e in ex], linewidths=1, edgecolors=INK, zorder=4)
    out = zoom(ax, p, var)
    ax.set_title(f'Affinity propagation (exemplar): {len(ex)} clusters, each around one exemplar unit (★)\n'
                 f'grey lines = unit → its exemplar; {out} far-away units outside the view', fontsize=9)
    return lab


def meanshift_family(ax, X, codes, pal, pca, p, var, **_):
    ms = MeanShift(bandwidth=estimate_bandwidth(X, quantile=0.2, random_state=SEED), bin_seeding=True).fit(X)
    lab = ms.labels_
    kde = gaussian_kde(p.T)
    lo, hi = np.quantile(p, [0.003, 0.997], axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 150), np.linspace(lo[1], hi[1], 150))
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, z, levels=8, colors='#8a8984', linewidths=0.7, zorder=1)
    points(ax, p, codes, pal, s=9)
    centres = pca.transform(ms.cluster_centers_)
    sizes = np.bincount(lab)
    ax.scatter(*centres.T, s=40 + sizes / sizes.max() * 300, marker='^', facecolor='white', edgecolor=INK, lw=1.4,
               zorder=4)
    for c in np.argsort(sizes)[::-1][:6]:
        ax.annotate(f'peak {c} (n={sizes[c]})', centres[c], xytext=(8, 6), textcoords='offset points', fontsize=7,
                    fontweight='bold', zorder=5, bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                                           edgecolor=INK, lw=0.6))
    out = zoom(ax, p, var)
    ax.set_title(f'Mean shift (mode-seeking): {len(centres)} density peaks (▲, size = units); lines = density\n'
                 f'each unit climbs to its nearest peak; {out} far-away units outside the view', fontsize=9)
    return lab


VIEWS = {'SOM (neural)': som_family, 'Grid (grid-based, PC1-PC2)': grid_family, 'BIRCH (tree / CF)': birch_family,
         'Affinity propagation (exemplar)': affinity_family, 'Mean shift (mode-seeking)': meanshift_family}


def make_up(ax, lab, codes, pal, receptors, top=10):
    """For each cluster (largest first): share of each NR subfamily; label = size and main receptors."""
    sizes = pd.Series(lab[lab != -1]).value_counts()
    rows = list(sizes.index[:top])
    names = [f'cluster {c} (n={sizes[c]})' for c in rows]
    groups = [lab == c for c in rows]
    if len(sizes) > top:
        groups.append(np.isin(lab, sizes.index[top:]))
        names.append(f'{len(sizes) - top} smaller clusters (n={groups[-1].sum()})')
    if (lab == -1).any():
        groups.append(lab == -1)
        names.append(f'noise (n={(lab == -1).sum()})')
    order = sorted(pal)
    left = np.zeros(len(groups))
    for code in order:
        share = np.array([np.mean(codes[g] == code) * 100 for g in groups])
        ax.barh(range(len(groups)), share, left=left, color=pal[code], height=0.7, edgecolor='white', linewidth=1)
        left += share
    for i, g in enumerate(groups):
        top3 = pd.Series(receptors[g]).value_counts().head(3)
        ax.text(101, i, ', '.join(f'{r} {v}' for r, v in top3.items()), va='center', fontsize=7, color=INK2)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of the cluster')
    ax.set_title('What each cluster is made of (NR subfamily; right: main receptors)')
    ax.spines['right'].set_visible(False)


def legend(fig, codes, pal, receptors):
    handles = []
    for code in sorted(pal):
        names = ', '.join(pd.Series(receptors[codes == code]).value_counts().index[:3])
        handles.append(plt.Line2D([], [], marker='s', ls='', color=pal[code], ms=8,
                                  label=f'{code}  {names}  (n={np.sum(codes == code)})'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=False,
               title='NR subfamily (NR1 warm, NR2 blue, NR3 green, NR5 purple)', title_fontsize=8.5)


def main():
    style()
    df, all_feats = load()
    for name, motif in SUBSETS.items():
        sub = df if motif is None else df[df[MOTIF] == motif].reset_index(drop=True)
        feats, _ = prune(sub, all_feats)
        X = StandardScaler().fit_transform(sub[feats])
        res = pd.read_csv(OUT / name / 'benchmark_metrics.csv').set_index('method')
        pca = PCA(2, random_state=SEED).fit(X)
        codes = sub.nr_subfamily.values
        receptors = sub[RECEPTOR].values
        pal = subfamily_colours(codes)
        common = dict(X=X, pca=pca, p=pca.transform(X), var=pca.explained_variance_ratio_ * 100, codes=codes,
                      pal=pal, receptors=receptors)
        folder = OUT / name / 'family_colour'
        folder.mkdir(exist_ok=True)
        for m, view in VIEWS.items():
            fig = plt.figure(figsize=(18, 9.2))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.05, right=0.87, top=0.86, bottom=0.2,
                                  wspace=0.38)
            a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            lab = np.asarray(view(a, k=int(res.loc[m, 'clusters']), **common))
            make_up(b, lab, codes, pal, receptors)
            legend(fig, codes, pal, receptors)
            r = res.loc[m]
            fig.suptitle(f'{m} · {len(sub)} units ({name}) · coloured by NR subfamily\n'
                         f'{int(r.clusters)} clusters · silhouette {r.silhouette:.2f} · stability {r.stability_ARI:.2f}'
                         f' · match with NR family (ARI) {r.ARI_family:.2f}', x=0.01, ha='left', fontsize=11,
                         color=INK2)
            fig.savefig(folder / f'{m.split(" (")[0].replace(" ", "_")}.png', dpi=150)
            plt.close(fig)
        print('saved', folder, flush=True)


if __name__ == '__main__':
    main()
