"""Slide plots of the experimental metadata (716 curated structures), one message per plot:
  fig10_quality.png         resolution vs R-free; the few weaker structures highlighted
  fig11_crystal_form.png    share of each receptor's structures in its most common space group
  fig12_stoichiometry.png   receptor-peptide copies per crystal; the 3 structures with extra peptides
Reads 05_metadata_ligands/structure_metadata.csv (run extract_metadata.py first).
Colours follow the BIOL 363 progress-update slides (green/gold pair checked with the dataviz validator).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GREEN, GOLD, PALE, INK, MUTED, GRID, BG = '#467a26', '#b8860b', '#c9cfc0', '#263b1f', '#6d726b', '#e3e6dc', '#f9f9f3'
NAMES = {'ESR1': 'ERα', 'PPARG': 'PPARγ', 'FXR': 'FXR', 'RXRA': 'RXRα', 'RORC': 'RORγ', 'PPARA': 'PPARα',
         'ESR2': 'ERβ', 'GR': 'GR', 'LRH1': 'LRH-1'}
SG = {'P 1 21 1': 'P2₁', 'P 21 21 21': 'P2₁2₁2₁', 'C 2 2 21': 'C222₁', 'P 43 21 2': 'P4₃2₁2',
      'P 41 21 2': 'P4₁2₁2', 'P 32 2 1': 'P3₂21'}


def axes(figsize=(11.5, 6.1)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c9cdc2')
    ax.tick_params(colors=MUTED, labelsize=14)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / name, facecolor=BG)
    plt.close(fig)


def quality(m):
    weak = (m.resolution_A > 3.0) | (m.R_free > 0.30)
    fig, ax = axes()
    ax.grid(color=GRID, linewidth=1)
    ax.scatter(m.resolution_A[~weak], m.R_free[~weak], s=34, color=GREEN, alpha=0.55, linewidths=0,
               label=f'{(~weak).sum()} structures')
    ax.scatter(m.resolution_A[weak], m.R_free[weak], s=60, color=GOLD, edgecolors=BG, linewidths=1.2,
               label=f'{weak.sum()} weaker: resolution > 3 Å or R-free > 0.30')
    ax.axvline(3.0, color=MUTED, linestyle='--', linewidth=1)
    ax.axhline(0.30, color=MUTED, linestyle='--', linewidth=1)
    ax.set_xlabel('Resolution (Å)   ← sharper', fontsize=15, color=MUTED)
    ax.set_ylabel('R-free   ← better fit', fontsize=15, color=MUTED)
    ax.legend(loc='upper left', fontsize=15, frameon=False, labelcolor=INK)
    save(fig, 'fig10_quality.png')
    return weak.sum()


def crystal_form(m):
    m = m.assign(rec=m.receptors.str.split(';').str[0])
    m = m[m.rec.isin(NAMES)]
    top = m.groupby('rec').space_group.agg(lambda s: s.value_counts().index[0])
    share = m.groupby('rec').apply(lambda g: (g.space_group == top[g.name]).mean() * 100, include_groups=False)
    n = m.groupby('rec').size()
    order = share.sort_values().index
    fig, ax = axes()
    y = range(len(order))
    ax.barh(y, [share[r] for r in order], color=GREEN, height=0.66)
    ax.barh(y, [100 - share[r] for r in order], left=[share[r] for r in order], color=PALE, height=0.66)
    for i, r in enumerate(order):
        ax.text(share[r] - 1.5, i, f'{share[r]:.0f}%', va='center', ha='right', fontsize=15, fontweight='bold',
                color='white')
        ax.text(101.5, i, f'{SG.get(top[r], top[r])}  ·  n = {n[r]}', va='center', fontsize=13, color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels([NAMES[r] for r in order], fontsize=16, color=INK)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('% of the receptor\'s structures in its most common crystal form', fontsize=15, color=MUTED)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    save(fig, 'fig11_crystal_form.png')
    return share.round().to_dict()


def stoichiometry(m):
    one = m.peptides_per_receptor_chain == 1
    copies = m[one].receptor_chains.astype(int).value_counts()
    rows = [('1', copies.get(1, 0)), ('2', copies.get(2, 0)), ('3', copies.get(3, 0)), ('4', copies.get(4, 0)),
            ('6 or 8', copies.get(6, 0) + copies.get(8, 0))]
    labels = [r[0] for r in rows] + ['extra peptide']
    values = [r[1] for r in rows] + [(~one).sum()]
    fig, ax = axes()
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.bar(range(len(values)), values, color=[GREEN] * len(rows) + [GOLD], width=0.62)
    for i, v in enumerate(values):
        ax.text(i, v + 6, str(v), ha='center', fontsize=17, fontweight='bold', color=INK)
    ax.text(len(values) - 1, values[-1] + 40, '1YOK, 1ZDU, 9H66\noutside the groove', ha='center', fontsize=13,
            color=MUTED)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=16, color=INK)
    ax.set_xlabel('Receptor–peptide copies in the crystal (one peptide per receptor chain)', fontsize=15, color=MUTED)
    ax.set_ylabel('PDB structures', fontsize=15, color=MUTED)
    ax.set_ylim(0, max(values) * 1.15)
    save(fig, 'fig12_stoichiometry.png')
    return dict(zip(labels, values))


def main():
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    m = pd.read_csv(ROOT / '05_metadata_ligands/structure_metadata.csv')
    m = m[m.in_geometry_set]
    print('structures', len(m))
    print('weaker', quality(m))
    print('crystal form share', crystal_form(m))
    print('stoichiometry', stoichiometry(m))


if __name__ == '__main__':
    main()
