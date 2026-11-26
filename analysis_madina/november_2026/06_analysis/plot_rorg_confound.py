"""fig14: is the RORγ RIP140 / SRC split caused by the ligand, the crystal form or the peptide? (Nov 2026)

Same RORγ units and HDBSCAN groups as plot_rorg_motifs.py (fig13).
  left   crystal form (space group) of the PDB entries in the RIP140 group and in the SRC group
  right  what differs between the groups: median RIP140 minus median SRC for each descriptor, divided by the
         usual spread of that descriptor (IQR over all coactivator units); 1 = as large as the usual spread
Ligand check (printed): ligand states from 05_metadata_ligands/ligand_states_auto.csv per group.
Writes figures/fig14_rorg_confound.png and rorg_confound_check.csv.
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
PALE_GREEN, PALE_GOLD = '#a9bf95', '#dcc27a'
F = ['angle_H3_H12', 'angle_H12_motif', 'angle_H3_motif', 'dist_H12_motif', 'dist_H3_motif', 'dist_H3_H12']
SHOW = [('angle_H12_motif', 'H12–motif angle', '°'), ('dist_H12_motif', 'H12–motif distance', ' Å'),
        ('angle_H3_H12', 'H3–H12 angle', '°'), ('dist_H3_H12', 'H3–H12 distance', ' Å')]
SG = {'P 1 21 1': 'P2₁', 'P 63 2 2': 'P6₃22', 'P 41 21 2': 'P4₁2₁2', 'P 41': 'P4₁', 'P 21 21 21': 'P2₁2₁2₁'}


def groups():
    all_units = pd.read_csv(HERE / 'clustering_bota/clustered_table_all_units.csv')
    d = all_units[all_units.receptor == 'RORC'].reset_index(drop=True)
    lab = HDBSCAN(min_cluster_size=4, min_samples=4).fit_predict(StandardScaler().fit_transform(d[F]))
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    name = {sizes.index[sizes == 23][0]: 'RIP140', sizes.index[sizes == 31][0]: 'SRC'}
    d['group'] = [name.get(c, 'other') for c in lab]
    meta = pd.read_csv(ROOT / '05_metadata_ligands/structure_metadata.csv')[['pdb_id', 'space_group']]
    lig = pd.read_csv(ROOT / '05_metadata_ligands/ligand_states_auto.csv')[['pdb_id', 'ligand_state']]
    d = d.merge(meta, on='pdb_id', how='left').merge(lig, on='pdb_id', how='left')
    return d, all_units[all_units.cofactor_type == 'COACTIVATOR']


def main():
    d, co = groups()
    pdb = d[d.group != 'other'].drop_duplicates(['pdb_id', 'group'])
    print(pd.crosstab(pdb.group, pdb.ligand_state.fillna('unknown')))
    iqr = co[F].quantile(0.75) - co[F].quantile(0.25)
    med = d.groupby('group')[F].median()
    diff = med.loc['RIP140'] - med.loc['SRC']
    pd.DataFrame({'RIP140_median': med.loc['RIP140'], 'SRC_median': med.loc['SRC'], 'difference': diff,
                  'IQR_all_coactivators': iqr, 'difference_in_IQR': diff.abs() / iqr}).round(2).to_csv(
        HERE / 'rorg_confound_check.csv')

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={'width_ratios': [1, 1.15]})
    fig.patch.set_facecolor(BG)
    for ax in (a, b):
        ax.set_facecolor(BG)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color('#c9cdc2')
        ax.tick_params(colors=MUTED, labelsize=13)

    # left: crystal forms per group
    rows = [('RIP140', 'RIP140 group'), ('SRC', 'SRC group')]
    for y, (g, label) in enumerate(rows[::-1]):
        counts = pdb[pdb.group == g].space_group.value_counts()
        left = 0
        for k, (sg, n) in enumerate(counts.items()):
            colour = (GREEN if k == 0 else PALE_GREEN) if g == 'RIP140' else (GOLD if k == 0 else PALE_GOLD)
            share = 100 * n / counts.sum()
            a.barh(y, share, left=left, color=colour, height=0.5, edgecolor=BG, linewidth=2)
            if k == 0:
                a.text(left + share / 2, y, f'{SG.get(sg, sg)}  {n}', ha='center', va='center', fontsize=14,
                       fontweight='bold', color='white')
            left += share
        rest = ', '.join(f'{SG.get(sg, sg)} {n}' for sg, n in list(counts.items())[1:])
        if rest:
            a.text(0, y - 0.42, f'other forms (light): {rest}', ha='left', va='center', fontsize=12, color=MUTED)
    a.set_yticks([0, 1], [f'{label}\n{(pdb.group == g).sum()} PDB' for g, label in rows[::-1]], fontsize=14,
                 color=INK)
    a.set_xlim(0, 100)
    a.set_ylim(-0.7, 1.45)
    a.set_xlabel('% of PDB entries in each crystal form (space group)', fontsize=13, color=MUTED)
    a.set_title('Crystal form: the two groups never share one', loc='left', fontsize=15, color=INK,
                fontweight='bold')

    # right: what differs, in units of the usual spread
    y = np.arange(len(SHOW))[::-1]
    vals = [abs(diff[c]) / iqr[c] for c, _, _ in SHOW]
    colours = [GREEN if c.startswith(('angle_H12_motif', 'dist_H12_motif')) else PALE_GREEN for c, _, _ in SHOW]
    b.barh(y, vals, color=colours, height=0.55)
    b.axvline(1, color=MUTED, linestyle='--', linewidth=1)
    b.text(1.03, len(SHOW) - 0.45, 'usual spread', fontsize=12, color=MUTED, va='center')
    for yi, v, (c, label, unit) in zip(y, vals, SHOW):
        b.text(v + 0.06, yi, f'{diff[c]:+.1f}{unit}', va='center', fontsize=13, fontweight='bold', color=INK)
    b.set_yticks(y, [label for _, label, _ in SHOW], fontsize=14, color=INK)
    b.set_xlim(0, max(vals) * 1.25)
    b.set_ylim(-0.5, len(SHOW) - 0.2)
    b.set_xlabel('RIP140 minus SRC, in units of the usual spread (IQR)', fontsize=13, color=MUTED)
    b.set_title('What differs: the peptide moves, H12 barely', loc='left', fontsize=15, color=INK, fontweight='bold')
    b.grid(axis='x', color=GRID, linewidth=1)
    b.set_axisbelow(True)
    fig.tight_layout(w_pad=4)
    fig.savefig(HERE / 'figures' / 'fig14_rorg_confound.png', dpi=110, facecolor=BG)
    plt.close(fig)
    print((diff.abs() / iqr).round(2))


if __name__ == '__main__':
    main()
