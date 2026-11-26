"""Rank descriptors by importance, then add the 10 pairwise landmark distances and check what changes (prof, Nov 2026).

Descriptor sets (21 in total)
  6 helix descriptors      the 6-descriptor table (H3, H12 and motif axes: 3 angles, 3 distances)
  + 15 pairwise distances  between Lys clamp (H3), Glu AF-2 (H12), motif L1, L4, L5 (CA) and the ligand-pocket centroid
                           (pairwise_landmarks.py); L4-L5 is one helix step and constant, kept as a check
Units: clustering units (pairing errors 1ZDU-Q, 7B9O-C left out) with all 21 values (LRH-1: no Lys clamp; 8 stapled ERα: no centroid).

Importance, one ranking per question
  receptors       Kruskal-Wallis epsilon² between receptors, on one median per PDB entry and receptor
                  (receptors with >= 5 PDB), as on the "what drives the separation" slide
  cofactor type   coactivator vs corepressor: |Cliff's delta| on the same per-PDB medians (0 = same, 1 = no overlap)
  NR family       random forest, 5-fold CV with whole PDB entries held out: balanced accuracy lost when one
                  descriptor is shuffled (permutation importance), for 6 and for 21 descriptors
What changes
  - where the new distances enter each ranking; correlation of each with the closest helix descriptor
  - NR-family accuracy with 6 vs 21 descriptors
  - HDBSCAN on all units with 6 vs 21 descriptors (same settings as the benchmark): clusters and agreement (ARI)
Writes descriptor_importance.csv, descriptor_importance_changes.txt and figures/fig18_descriptor_importance.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
GREEN, GOLD, INK, MUTED, GRID, BG = '#467a26', '#b8860b', '#263b1f', '#6d726b', '#e3e6dc', '#f9f9f3'
HELIX = {'dist_H3_H12': 'H3–H12 distance', 'angle_H3_H12': 'H3–H12 angle', 'dist_H3_motif': 'H3–motif distance',
         'angle_H12_motif': 'H12–motif angle', 'dist_H12_motif': 'H12–motif distance',
         'angle_H3_motif': 'H3–motif angle'}
PAIR = {'d_K_E': 'Lys clamp – Glu AF-2', 'd_K_L1': 'Lys clamp – L1', 'd_K_L4': 'Lys clamp – L4',
        'd_K_L5': 'Lys clamp – L5', 'd_E_L1': 'Glu AF-2 – L1', 'd_E_L4': 'Glu AF-2 – L4', 'd_E_L5': 'Glu AF-2 – L5',
        'd_L1_L4': 'L1 – L4', 'd_L1_L5': 'L1 – L5', 'd_L4_L5': 'L4 – L5',
        'd_K_C': 'centroid – Lys clamp', 'd_E_C': 'centroid – Glu AF-2', 'd_L1_C': 'centroid – L1',
        'd_L4_C': 'centroid – L4', 'd_L5_C': 'centroid – L5'}
NAME = {**HELIX, **PAIR}
EXCLUDE = {('1ZDU', 'A', 'Q'), ('7B9O', 'A', 'C')}


def load():
    u = pd.read_csv(HERE / 'clustering_bota/clustered_table_all_units.csv')
    p = pd.read_csv(HERE / 'pairwise_landmarks.csv').drop(columns=['receptor', 'status'])
    d = u.merge(p, on=['pdb_id', 'receptor_chain', 'peptide_chain'])
    d = d[[(a, b, c) not in EXCLUDE for a, b, c in zip(d.pdb_id, d.receptor_chain, d.peptide_chain)]]
    return d.dropna(subset=list(NAME)).reset_index(drop=True)


def epsilon2(d, f):
    m = d.groupby(['pdb_id', 'receptor'])[f].median().reset_index()
    keep = m.receptor.map(m.receptor.value_counts()) >= 5
    m = m[keep]
    groups = [g[f].values for _, g in m.groupby('receptor')]
    h = kruskal(*groups).statistic
    n, k = len(m), len(groups)
    return (h - k + 1) / (n - k)


def cliffs(d, f):
    m = d.groupby(['pdb_id', 'cofactor_type'])[f].median().reset_index()
    a = m[m.cofactor_type == 'COACTIVATOR'][f].values
    b = m[m.cofactor_type == 'COREPRESSOR'][f].values
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return abs(gt - lt)


def family_importance(d, feats):
    X, y, g = d[feats].to_numpy(), d.family.astype(str), d.pdb_id
    keep = (y.map(y.value_counts()) >= 10).to_numpy()
    X, y, g = X[keep], y[keep].reset_index(drop=True), g[keep]
    imp, pred = np.zeros(len(feats)), np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, g):
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr], y.iloc[tr])
        pred[te] = rf.predict(X[te])
        imp += permutation_importance(rf, X[te], y.iloc[te], scoring='balanced_accuracy', n_repeats=5,
                                      random_state=0).importances_mean / 5
    return pd.Series(imp, index=feats), balanced_accuracy_score(y, pred)


def hdbscan(d, feats):
    X = StandardScaler().fit_transform(d[feats])
    return HDBSCAN(min_cluster_size=max(10, len(X) // 100)).fit_predict(X)


def barh(ax, s, title, xlabel):
    s = s.sort_values()
    colours = [GREEN if f in HELIX else GOLD for f in s.index]
    ax.barh(range(len(s)), s.values, color=colours, height=0.62)
    ax.set_yticks(range(len(s)), [NAME[f] for f in s.index], fontsize=11, color=INK)
    for i, v in enumerate(s.values):
        ax.text(v + s.max() * 0.015, i, f'{v:.2f}', va='center', fontsize=10, color=INK)
    rank = {f: i + 1 for i, f in enumerate(s.index[::-1])}
    ax.set_xlim(0, s.max() * 1.18)
    ax.set_title(title, loc='left', fontsize=13.5, fontweight='bold', color=INK)
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED)
    ax.set_facecolor(BG)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#c9cdc2')
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis='x', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return rank


def main():
    d = load()
    helix, allf = list(HELIX), list(NAME)
    eps = pd.Series({f: epsilon2(d, f) for f in allf})
    cd = pd.Series({f: cliffs(d, f) for f in allf})
    imp6, acc6 = family_importance(d, helix)
    imp16, acc16 = family_importance(d, allf)
    corr = d[allf].corr(method='spearman')
    closest = {f: corr.loc[f, helix].abs().idxmax() for f in PAIR}

    table = pd.DataFrame({'descriptor': allf, 'label': [NAME[f] for f in allf],
                          'set': ['helix (6)' if f in HELIX else 'pairwise (new)' for f in allf],
                          'epsilon2_receptors': eps.values, 'cliffs_delta_coact_vs_corep': cd.values,
                          'family_importance_6': [imp6.get(f, np.nan) for f in allf],
                          'family_importance_16': imp16.values,
                          'closest_helix_descriptor': [closest.get(f, '') for f in allf],
                          'spearman_with_closest': [corr.loc[f, closest[f]] if f in PAIR else np.nan for f in allf]})
    for col in ['epsilon2_receptors', 'cliffs_delta_coact_vs_corep', 'family_importance_16']:
        table[f'rank_{col}'] = table[col].rank(ascending=False).astype(int)
    table['rank_family_importance_6'] = table.family_importance_6.rank(ascending=False)
    table.round(3).to_csv(HERE / 'descriptor_importance.csv', index=False)

    l6, l16 = hdbscan(d, helix), hdbscan(d, allf)
    lines = [f'units: {len(d)}',
             f'NR family predicted (balanced accuracy): 6 descriptors {acc6:.2f}, 21 descriptors {acc16:.2f}',
             f'HDBSCAN all units, 6 descriptors: clusters {pd.Series(l6[l6 >= 0]).value_counts().tolist()}, '
             f'noise {(l6 < 0).sum()}',
             f'HDBSCAN all units, 21 descriptors: clusters {pd.Series(l16[l16 >= 0]).value_counts().tolist()}, '
             f'noise {(l16 < 0).sum()}',
             f'agreement 6 vs 21 (ARI, all units incl. noise as a group): {adjusted_rand_score(l6, l16):.2f}']
    for f in PAIR:
        lines.append(f'{NAME[f]}: Spearman {corr.loc[f, closest[f]]:+.2f} with {NAME[closest[f]]}')
    (HERE / 'descriptor_importance_changes.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print(table.drop(columns=['descriptor']).round(2).to_string(index=False))

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.6))
    fig.patch.set_facecolor(BG)
    barh(axes[0], eps, 'Differs between receptors', 'ε² (Kruskal–Wallis), share of spread explained by receptor')
    barh(axes[1], cd, 'Differs coactivator vs corepressor', "|Cliff's δ| (0 = same, 1 = no overlap)")
    barh(axes[2], imp16.clip(lower=0), f'Predicts NR family (accuracy {acc16:.2f}; 6 only: {acc6:.2f})',
         'accuracy lost when shuffled (random forest)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, GOLD)]
    fig.legend(handles, ['6 helix descriptors (H3, H12, motif axes)', 'new pairwise distances (Lys, Glu, L1, L4, L5, pocket centroid)'],
               ncol=2, loc='lower center', fontsize=12.5, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=3)
    fig.savefig(HERE / 'figures' / 'fig18_descriptor_importance.png', dpi=110, facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    main()
