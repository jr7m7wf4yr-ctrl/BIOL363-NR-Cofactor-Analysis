"""Is the cofactor or the receptor the driving force of the binding geometry? (prof's question, Nov 2026)

Cofactor = the bound peptide, labelled by its motif (LHRLL = SRC-2 box 2, LRYLL = SRC-1 box 2, LEDII / LEAII = NCoR /
SMRT CoRNR boxes, ...); cofactor type = coactivator or corepressor. Receptor = the NR the peptide is bound to.

  fig A  variance partitioning: share of the geometry (all descriptors, standardised) explained by the receptor alone,
         by the cofactor alone, and by both together (adjusted R², so factors with many levels are not favoured);
         unique receptor = both - cofactor, unique cofactor = both - receptor, shared = the overlap (the two are
         confounded: most receptors were crystallised with SRC-2).  For all units and for coactivators only,
         on the 6-descriptor and the 14-descriptor sets.
  fig B  the same thing as pictures on one PCA map: one receptor with different cofactors vs one cofactor on
         different receptors, plus the two receptors where the coactivator peptide matters most
  fig C  inside each receptor: share of its geometry explained by which coactivator peptide is bound

Only receptor x motif combinations with >= 3 units are used.  Outputs in 06_analysis/cofactor_vs_receptor/.
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cofactor_vs_receptor'
MIN_CELL = 3
INK, INK2, MUTED, OTHER = '#0b0b0b', '#52514e', '#898781', '#d6d5cf'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
C_REC, C_SHARED, C_COF, C_NONE = '#2a78d6', '#a3a29c', '#eb6834', '#efeee9'
NAMES = {'bota': '6 descriptors', 'ours': '14 descriptors'}


def load(profile):
    """Unit table with columns receptor, motif, cofactor_type and the (pruned) descriptors."""
    sys.argv = ['x', profile]
    sys.modules.pop('clustering_benchmark', None)
    cb = importlib.import_module('clustering_benchmark')
    df, feats = cb.load()
    feats, _ = cb.prune(df, feats)
    df = df.rename(columns={cb.RECEPTOR: 'receptor_name'})
    if profile == 'ours':
        df['motif'] = df.motif_seq
        df['cofactor_type'] = np.where(df[cb.MOTIF] == cb.COREPRESSOR, 'COREPRESSOR', 'COACTIVATOR')
    df = df[df.groupby(['receptor_name', 'motif']).pdb_id.transform('size') >= MIN_CELL].reset_index(drop=True)
    return df, feats


def r2(Z, factors):
    """Adjusted R² of a linear model (one-hot factors) summed over all standardised descriptors."""
    G = np.hstack([pd.get_dummies(f).to_numpy(float)[:, (0 if i == 0 else 1):] for i, f in enumerate(factors)])
    n, p = len(Z), G.shape[1] - 1
    b, *_ = np.linalg.lstsq(G, Z, rcond=None)
    return 1 - (((Z - G @ b) ** 2).sum() / (n - p - 1)) / (((Z - Z.mean(0)) ** 2).sum() / (n - 1))


def partition(d, feats):
    Z = StandardScaler().fit_transform(d[feats])
    R, C, A = r2(Z, [d.receptor_name]), r2(Z, [d.motif]), r2(Z, [d.receptor_name, d.motif])
    return {'units': len(d), 'receptors': d.receptor_name.nunique(), 'cofactor_motifs': d.motif.nunique(),
            'receptor_alone': R, 'cofactor_alone': C, 'both': A,
            'unique_receptor': A - C, 'unique_cofactor': A - R, 'shared': R + C - A, 'unexplained': 1 - A}


def within_receptor(d, feats):
    co = d[d.cofactor_type == 'COACTIVATOR']
    Z = StandardScaler().fit_transform(co[feats])
    rows = []
    for r, g in co.groupby('receptor_name'):
        if g.motif.nunique() >= 2:
            rows.append({'receptor': r, 'units': len(g), 'peptide_motifs': g.motif.nunique(),
                         'motifs': ', '.join(f'{m} {n}' for m, n in g.motif.value_counts().items()),
                         'share_explained_by_peptide': max(r2(Z[g.index.map(co.index.get_loc)], [g.motif]), 0)})
    return pd.DataFrame(rows)


def fig_a(part):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    labels = []
    for i, (_, r) in enumerate(part.iloc[::-1].iterrows()):
        left = 0
        for col, c in [('unique_receptor', C_REC), ('shared', C_SHARED), ('unique_cofactor', C_COF),
                       ('unexplained', C_NONE)]:
            v = max(r[col], 0)
            ax.barh(i, v, left=left, color=c, height=0.6, edgecolor='white', linewidth=2)
            if v >= 0.06 and col != 'unexplained':
                ax.text(left + v / 2, i, f'{v:.0%}', ha='center', va='center', fontsize=9,
                        color='white' if col != 'shared' else INK)
            left += v
        labels.append(f'{r.subset}\n{r.descriptors} ({r.units} units)')
    ax.set_yticks(range(len(part)), labels, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('share of the binding geometry explained')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_REC, C_SHARED, C_COF, C_NONE)]
    ax.legend(handles, ['receptor only', 'receptor and cofactor together (confounded)', 'cofactor only',
                        'unexplained (ligand, crystal, noise)'], ncol=4, fontsize=8, frameon=False,
              loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.set_title('With corepressors included the cofactor explains more than the receptor; among coactivators the '
                 'receptor explains more', loc='left', fontsize=10, pad=24)
    fig.tight_layout()
    fig.savefig(OUT / 'A_variance_partitioning.png', dpi=160)
    plt.close(fig)


def panel(ax, p, mask, values, title, lim, order=None):
    ax.scatter(*p[~mask].T, s=6, c=OTHER, linewidths=0, zorder=1)
    vals = pd.Series(values[mask])
    cats = order or list(vals.value_counts().index)
    for i, v in enumerate(cats[:len(SERIES)]):
        m = np.flatnonzero(mask)[(vals == v).to_numpy()]
        ax.scatter(*p[m].T, s=18, c=SERIES[i], linewidths=0.4, edgecolors='white', zorder=2, label=f'{v} ({len(m)})')
    rest = np.flatnonzero(mask)[~vals.isin(cats[:len(SERIES)]).to_numpy()]
    if len(rest):
        ax.scatter(*p[rest].T, s=12, c=INK2, linewidths=0, zorder=2, label=f'other ({len(rest)})')
    ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title(title, loc='left', fontsize=9.5)
    if lim is not None:
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
    ax.set_xticks([])
    ax.set_yticks([])


def fig_b(d, feats, wr):
    Z = StandardScaler().fit_transform(d[feats])
    pca = PCA(2, random_state=0).fit(Z)
    p = pca.transform(Z)
    var = pca.explained_variance_ratio_ * 100
    lo, hi = np.quantile(p[(d.cofactor_type == 'COACTIVATOR').to_numpy()], [0.005, 0.995], axis=0)
    zoom = ((lo[0] - 0.3, hi[0] + 0.3), (lo[1] - 0.3, hi[1] + 0.3))
    cof = (d.motif + np.where(d.cofactor_type == 'COREPRESSOR', ' (corepressor)', '')).to_numpy()
    rec = d.receptor_name.to_numpy()
    top_motif = d[d.cofactor_type == 'COACTIVATOR'].motif.value_counts().index[0]
    best = wr.sort_values('share_explained_by_peptide', ascending=False).receptor.tolist()[:2]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel(axes[0, 0], p, (rec == 'PPARγ'), cof, 'ONE RECEPTOR, different cofactors: PPARγ\n'
          'corepressor-bound units sit far from the coactivator-bound ones', None)
    panel(axes[0, 1], p, (d.motif == top_motif).to_numpy() & (d.cofactor_type == 'COACTIVATOR').to_numpy(), rec,
          f'ONE COFACTOR, different receptors: {top_motif} (SRC-2 box 2)\nall receptors fall into the same cloud '
          f'(zoomed), with some shift by receptor', zoom)
    for ax, r in zip(axes[1], best):
        row = wr.set_index('receptor').loc[r]
        panel(ax, p, (rec == r) & (d.cofactor_type == 'COACTIVATOR').to_numpy(), d.motif.to_numpy(),
              f'ONE RECEPTOR, different coactivator peptides: {r}\npeptide explains '
              f'{row.share_explained_by_peptide:.0%} of its geometry (zoomed)', zoom)
    fig.suptitle(f"Same PCA map in every panel (6 descriptors; PC1 {var[0]:.0f}%, PC2 {var[1]:.0f}%); "
                 'grey = all other units', x=0.01, ha='left', color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / 'B_examples_on_map.png', dpi=150)
    plt.close(fig)


def fig_c(wr_all):
    w = wr_all.pivot_table(index='receptor', columns='descriptors', values='share_explained_by_peptide')
    w = w.loc[w.mean(1).sort_values().index]
    units = wr_all.groupby('receptor').units.max()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(w))
    for j, (col, c) in enumerate(zip(NAMES.values(), [OTHER, SERIES[1]])):
        if col in w:
            ax.barh(y + (j - 0.5) * 0.38, w[col].fillna(0), height=0.36, color=c, label=col)
    ax.set_yticks(y, [f'{r}  (n={units[r]})' for r in w.index], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('share of the receptor\'s geometry explained by which coactivator peptide is bound')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title('Inside one receptor: does the coactivator peptide change the geometry?\n'
                 'receptors with >= 2 peptide motifs (>= 3 units each); small n = less reliable', loc='left',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'C_peptide_effect_within_receptor.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    parts, wrs, data = [], [], {}
    for profile, name in NAMES.items():
        d, feats = load(profile)
        data[profile] = (d, feats)
        for subset, sub in [('All units', d), ('Coactivators only', d[d.cofactor_type == 'COACTIVATOR'])]:
            parts.append({'subset': subset, 'descriptors': name, **partition(sub.reset_index(drop=True), feats)})
        wrs.append(within_receptor(d, feats).assign(descriptors=name))
    part = pd.DataFrame(parts).sort_values(['subset', 'descriptors']).reset_index(drop=True)
    wr = pd.concat(wrs, ignore_index=True)
    part.to_csv(OUT / 'variance_partitioning.csv', index=False)
    wr.to_csv(OUT / 'peptide_effect_within_receptor.csv', index=False)
    fig_a(part)
    fig_b(*data['bota'], wr[wr.descriptors == NAMES['bota']])
    fig_c(wr)
    pd.set_option('display.width', 220)
    print(part.round(2).to_string(index=False))
    print(wr.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
