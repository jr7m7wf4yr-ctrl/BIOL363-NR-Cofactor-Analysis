"""fig16: is the RORγ split by peptide motif unique, or do the other top receptors split by motif too? (prof, Nov 2026)

For each receptor: HDBSCAN on its own units (6 descriptors, standardised within the receptor), one rule for all:
min_cluster_size = min_samples = max(4, 5% of the receptor's units). For RORγ this is 4 and gives the clusters of
fig13. Colours as in fig13: green = LHRLL, gold = another coactivator motif, dark = corepressor. Under each bar: what
it contains (motif, cofactor name from cofactor_identity_v2.csv). Groups are ordered by size; 'not grouped' = noise.
The panel headline is the verdict; the match with motif and with crystal form (adjusted Rand index, noise left out)
is written to motif_by_receptor_summary.csv.
Writes figures/fig16_motif_by_receptor.png, motif_by_receptor.csv and motif_by_receptor_summary.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GREEN, GOLD, DARK, INK, MUTED, GRID, BG = '#467a26', '#b8860b', '#3d3d3a', '#263b1f', '#6d726b', '#e3e6dc', '#f9f9f3'
RED, GREY = '#b03a2e', '#8a8d85'
F = ['angle_H3_H12', 'angle_H12_motif', 'angle_H3_motif', 'dist_H12_motif', 'dist_H3_motif', 'dist_H3_H12']
SHORT = {'SRC-2 (TIF2/GRIP1)': 'SRC-2', 'unresolved': '?'}
# receptor, name, verdict colour, headline, one-line reason; ordered from 'yes' to 'cannot test'
PANELS = [
    ('PPARA', 'PPARα', GREEN, 'YES: splits by peptide', 'PGC-1α apart from SRC-1; same crystal forms in both'),
    ('RORC', 'RORγ', GREEN, 'YES: splits by peptide', 'RIP140 apart from SRC; but each in its own crystal form'),
    ('RXRA', 'RXRα', GREEN, 'YES, weak: splits by peptide', 'the LRYLL group is only one structure (1XLS)'),
    ('PPARG', 'PPARγ', GOLD, 'ONLY coactivator vs corepressor', 'PGC-1α and SRC-1 stay mixed'),
    ('FXR', 'FXR', RED, 'NO split by peptide', 'small group = one crystal form (P6₅)'),
    ('ESR1', 'ERα', GREY, 'CANNOT TEST', '95% have the same peptide (LHRLL); no groups found'),
]


def main():
    d = pd.read_csv(HERE / 'clustering_bota/clustered_table_all_units.csv')
    d = d.merge(pd.read_csv(ROOT / '05_metadata_ligands/structure_metadata.csv')[['pdb_id', 'space_group']],
                on='pdb_id', how='left')
    v = pd.read_csv(ROOT / '01_dataset/cofactor_identity/cofactor_identity_v2.csv')
    v = v.drop_duplicates(['pdb_id', 'peptide_chain'])[['pdb_id', 'peptide_chain', 'cofactor_v2']]
    d = d.merge(v, on=['pdb_id', 'peptide_chain'], how='left')
    d['kind'] = np.where(d.cofactor_type == 'COREPRESSOR', 'corepressor',
                         np.where(d.motif == 'LHRLL', 'LHRLL', 'other'))

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.4))
    fig.patch.set_facecolor(BG)
    rows, summary = [], []
    for ax, (code, name, vcol, headline, reason) in zip(axes.flat, PANELS):
        r = d[d.receptor == code].reset_index(drop=True)
        k = max(4, round(0.05 * len(r)))
        lab = HDBSCAN(min_cluster_size=k, min_samples=k).fit_predict(StandardScaler().fit_transform(r[F]))
        ok = lab >= 0
        if len(set(lab[ok])) > 1:
            summary.append({'receptor': name, 'groups': len(set(lab[ok])), 'not_grouped': int((~ok).sum()),
                            'match_with_motif_ARI': adjusted_rand_score(r.motif[ok], lab[ok]),
                            'match_with_crystal_form_ARI': adjusted_rand_score(r.space_group[ok], lab[ok])})
        groups = sorted(set(lab[ok]), key=lambda c: -np.sum(lab == c)) + ([-1] if (~ok).any() else [])
        names = []
        for i, c in enumerate(groups):
            g = r[lab == c]
            bottom = 0
            for kind, colour in [('LHRLL', GREEN), ('other', GOLD), ('corepressor', DARK)]:
                n = int((g.kind == kind).sum())
                if n:
                    ax.bar(i, n, bottom=bottom, color=colour, width=0.62, edgecolor=BG, linewidth=1)
                    bottom += n
            ax.text(i, bottom + len(r) * 0.015, str(len(g)), ha='center', va='bottom', fontsize=14,
                    fontweight='bold', color=INK)
            # what is in the group: top 2 motif + cofactor combinations
            combo = g.assign(c=g.cofactor_v2.map(lambda x: SHORT.get(x, x))).groupby(['motif', 'c']).size()
            combo = combo.sort_values(ascending=False)
            lines = [f'{n} {m} {c}' if c != '?' else f'{n} {m}' for (m, c), n in combo.head(2).items()]
            if len(combo) > 2:
                lines.append(f'+{int(combo.iloc[2:].sum())} other')
            if (g.cofactor_type == 'COREPRESSOR').all():
                lines = [f'{len(g)} corepressor']
            title = ('all units, no groups' if len(groups) == 1 else 'not grouped') if c == -1 else f'group {i + 1}'
            names.append(title + '\n' + '\n'.join(lines))
            for (m, cname), n in combo.items():
                rows.append({'receptor': name, 'group': title, 'motif': m, 'cofactor': cname, 'units': n})
        ax.set_xticks(range(len(groups)), names, fontsize=10.5, color=INK)
        ax.set_xlim(-0.6, max(len(groups), 3) - 0.4)
        ax.set_ylim(0, len(r) * 1.13)
        ax.set_facecolor(BG)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color('#c9cdc2')
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis='y', color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(f'{name}  ({len(r)} units)', loc='left', fontsize=15, fontweight='bold', color=INK, pad=52)
        ax.text(0, 1.11, headline, transform=ax.transAxes, fontsize=14, fontweight='bold', color=vcol,
                va='bottom', ha='left')
        ax.text(0, 1.03, reason, transform=ax.transAxes, fontsize=11, color=MUTED, va='bottom', ha='left')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, GOLD, DARK)]
    fig.legend(handles, ['LHRLL motif', 'other coactivator motif', 'corepressor'], ncol=3, loc='lower center',
               fontsize=13, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=4, w_pad=2.5)
    fig.savefig(HERE / 'figures' / 'fig16_motif_by_receptor.png', dpi=110, facecolor=BG)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(HERE / 'motif_by_receptor.csv', index=False)
    pd.DataFrame(summary).round(2).to_csv(HERE / 'motif_by_receptor_summary.csv', index=False)
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
