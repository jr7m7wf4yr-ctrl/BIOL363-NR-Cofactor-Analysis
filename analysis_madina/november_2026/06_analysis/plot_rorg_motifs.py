"""fig13: do all RORγ units in each HDBSCAN cluster carry the LHRLL motif? (prof's request, Nov 2026)

Same style as fig12_stoichiometry. RORγ units of the 6-descriptor table (69 units, 44 PDB), descriptors
standardised within RORγ, HDBSCAN(min_cluster_size=4, min_samples=4). This reproduces the RORγ panel shown to the
team (clusters of 23 and 31 units identical; the small cluster has 6 units here vs 5, noise 9 vs 10).
Clusters are numbered as in that panel: 0 = the 23-unit group, 1 = the small group, 2 = the 31-unit group.
Cofactor names come from 01_dataset/cofactor_identity/cofactor_identity_v2.csv.
Writes figures/fig13_rorg_motifs.png and rorg_cluster_motifs.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GREEN, GOLD, INK, MUTED, GRID, BG = '#467a26', '#b8860b', '#263b1f', '#6d726b', '#e3e6dc', '#f9f9f3'
F = ['angle_H3_H12', 'angle_H12_motif', 'angle_H3_motif', 'dist_H12_motif', 'dist_H3_motif', 'dist_H3_H12']
NAME = {'LLQLL': 'RIP140', 'LKKLL': 'SRC-1', 'LQQLL': 'SRC-1', 'LLSLL': 'synthetic peptide'}


def main():
    d = pd.read_csv(HERE / 'clustering_bota/clustered_table_all_units.csv')
    d = d[d.receptor == 'RORC'].reset_index(drop=True)
    v = pd.read_csv(ROOT / '01_dataset/cofactor_identity/cofactor_identity_v2.csv')
    d = d.merge(v[['pdb_id', 'peptide_chain', 'cofactor_v2']], on=['pdb_id', 'peptide_chain'], how='left')
    lab = HDBSCAN(min_cluster_size=4, min_samples=4).fit_predict(StandardScaler().fit_transform(d[F]))
    # number clusters like the team's panel: 23-unit group = 0, smallest = 1, 31-unit group = 2
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    rename = {sizes.index[sizes == 23][0]: 0, sizes.index[sizes == 31][0]: 2}
    rename.update({c: 1 for c in sizes.index if c not in rename})
    d['cluster'] = [rename.get(c, -1) for c in lab]
    d.groupby(['cluster', 'motif', 'cofactor_v2']).size().rename('units').reset_index().to_csv(
        HERE / 'rorg_cluster_motifs.csv', index=False)

    groups = [(0, 'cluster 0'), (2, 'cluster 2'), (1, 'cluster 1'), (-1, 'noise')]
    lhrll = [int(((d.cluster == c) & (d.motif == 'LHRLL')).sum()) for c, _ in groups]
    other = [int(((d.cluster == c) & (d.motif != 'LHRLL')).sum()) for c, _ in groups]
    other_text = []
    for c, _ in groups:
        o = d[(d.cluster == c) & (d.motif != 'LHRLL')].motif.value_counts()
        other_text.append('\n'.join(f'{n} × {m} ({NAME.get(m, "?")})' for m, n in o.items()))

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c9cdc2')
    ax.tick_params(colors=MUTED, labelsize=14)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    x = np.arange(len(groups))
    ax.bar(x, lhrll, color=GREEN, width=0.62, label='LHRLL (SRC-1 / SRC-2)')
    ax.bar(x, other, bottom=lhrll, color=GOLD, width=0.62, label='other motif')
    for i in x:
        total = lhrll[i] + other[i]
        ax.text(i, total + 0.6, str(total), ha='center', fontsize=17, fontweight='bold', color=INK)
        if lhrll[i] >= 3:
            ax.text(i, lhrll[i] / 2, str(lhrll[i]), ha='center', va='center', fontsize=15, fontweight='bold',
                    color='white')
        if other[i]:
            if other[i] >= 3:
                ax.text(i, lhrll[i] + other[i] / 2, f'{other[i]}', ha='center', va='center', fontsize=15,
                        fontweight='bold', color='white')
    share = [f'{100 * a / (a + b):.0f}% LHRLL' if a + b else '' for a, b in zip(lhrll, other)]
    ax.set_xticks(x, [f'{name}  ·  {s}' for (_, name), s in zip(groups, share)], fontsize=15, color=INK)
    for i in x:   # what the non-LHRLL units are, under each bar
        ax.annotate(other_text[i], (i, 0), xytext=(0, -36), textcoords='offset points', ha='center', va='top',
                    fontsize=12.5, color=GOLD, fontweight='bold', annotation_clip=False)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, max(a + b for a, b in zip(lhrll, other)) * 1.15)
    ax.set_ylabel('RORγ receptor–peptide units', fontsize=15, color=MUTED)
    ax.set_xlabel('HDBSCAN groups of RORγ (69 units, 44 PDB); gold text = the other motifs', fontsize=15, color=MUTED, labelpad=30)
    ax.legend(loc='upper right', fontsize=14, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'fig13_rorg_motifs.png', dpi=110, facecolor=BG)
    plt.close(fig)
    print(pd.DataFrame({'group': [g for _, g in groups], 'LHRLL': lhrll, 'other': other, 'other_motifs': other_text}))


if __name__ == '__main__':
    main()
