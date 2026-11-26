"""fig15: what is inside the green RORγ cluster 2 (31 units)? Are all of them LHRLL? (prof's question, Nov 2026)

Same RORγ HDBSCAN groups as fig13 / fig14 (plot_rorg_confound.groups); cluster 2 = the 31-unit SRC group, identical
to the green cluster on the clustering slide. Cofactor names from cofactor_identity_v2.csv; the 2 units of 4WPF have
no name there, but their peptide (…HKILHRLLQE…, PDB description RHKILHRLLQEGSPS) is the SRC-1 NR box 2.
Writes figures/fig15_rorg_cluster2.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from plot_rorg_confound import groups

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GREEN, GOLD, INK, MUTED, GRID, BG = '#467a26', '#b8860b', '#263b1f', '#6d726b', '#e3e6dc', '#f9f9f3'


def main():
    d, _ = groups()
    v = pd.read_csv(ROOT / '01_dataset/cofactor_identity/cofactor_identity_v2.csv')
    c = d[d.group == 'SRC'].merge(v[['pdb_id', 'peptide_chain', 'cofactor_v2']], on=['pdb_id', 'peptide_chain'])
    c.loc[c.pdb_id == '4WPF', 'cofactor_v2'] = 'SRC-1 (by sequence)'
    c['cof'] = c.cofactor_v2.str.replace(' (TIF2/GRIP1)', '', regex=False)
    bars = c.groupby(['motif', 'cof']).size().sort_values(ascending=False)
    bars = pd.concat([bars[bars.index.get_level_values(0) == 'LHRLL'], bars[bars.index.get_level_values(0) != 'LHRLL']])
    lhrll = int((c.motif == 'LHRLL').sum())

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, ax = plt.subplots(figsize=(11.5, 6.1))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c9cdc2')
    ax.tick_params(colors=MUTED, labelsize=14)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    colours = [GREEN if m == 'LHRLL' else GOLD for m, _ in bars.index]
    ax.bar(range(len(bars)), bars.values, color=colours, width=0.62)
    for i, v in enumerate(bars.values):
        ax.text(i, v + 0.4, str(v), ha='center', fontsize=17, fontweight='bold', color=INK)
    labels = []
    for m, cof in bars.index:
        extra = '\n9VZQ' if m != 'LHRLL' else ''
        labels.append(f'{m}\n{cof}{extra}')
    ax.set_xticks(range(len(bars)), labels, fontsize=15, color=INK)
    for t, (m, _) in zip(ax.get_xticklabels(), bars.index):
        if m != 'LHRLL':
            t.set_color(GOLD)
            t.set_fontweight('bold')
    ax.set_ylim(0, bars.max() * 1.18)
    ax.text(len(bars) - 0.6, bars.max() * 1.08, f'{lhrll} of {len(c)} units LHRLL\nall {len(c)} are SRC coactivators',
            ha='right', va='top', fontsize=16, color=INK, fontweight='bold')
    ax.set_ylabel('units in RORγ cluster 2', fontsize=15, color=MUTED)
    ax.set_xlabel(f'Motif and cofactor of the {len(c)} units ({c.pdb_id.nunique()} PDB) in the green cluster 2',
                  fontsize=15, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'fig15_rorg_cluster2.png', dpi=110, facecolor=BG)
    plt.close(fig)
    print(bars)


if __name__ == '__main__':
    main()
