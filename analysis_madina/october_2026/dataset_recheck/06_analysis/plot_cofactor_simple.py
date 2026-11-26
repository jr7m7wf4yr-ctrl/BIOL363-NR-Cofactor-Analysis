"""Simple slide version of the cofactor distribution: PDB entries per cofactor, large labels,
small cofactors pooled into 'Other'. Reads cofactor_counts_overall.csv (run cofactor_counts.py first).
Colours follow the BIOL 363 progress-update slides.
Output: figures/fig8b_cofactor_simple.png (slides) and fig8b_cofactor_simple_report.png (white, report)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
TOP = 6
DARK, LIGHT, INK, MUTED, BG = '#466b2f', '#a9b89c', '#263b1f', '#6d726b', '#f9f9f3'


def main():
    o = pd.read_csv(HERE / 'cofactor_counts_overall.csv', index_col=0)
    named = o.drop('unresolved', errors='ignore').sort_values('pdbs', ascending=False)
    top = named.pdbs.head(TOP)
    rest = named.iloc[TOP:]
    rows = list(top.items()) + [(f'Other ({len(rest)} cofactors)', int(rest.pdbs.sum()))]
    if 'unresolved' in o.index:
        rows.append(('Name unknown', int(o.loc['unresolved', 'pdbs'])))
    labels, values = zip(*rows[::-1])
    colors = [DARK if v >= 200 else LIGHT for v in values]

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, ax = plt.subplots(figsize=(11.5, 6.1), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.barh(range(len(values)), values, color=colors, height=0.7)
    for i, v in enumerate(values):
        ax.text(v + 6, i, str(v), va='center', fontsize=17, fontweight='bold', color=INK)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=17, color=INK)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel('PDB structures', fontsize=15, color=MUTED)
    ax.tick_params(axis='x', labelsize=13, colors=MUTED)
    ax.tick_params(axis='y', length=0)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9cdc2')
    ax.grid(axis='x', color='#e3e6dc', linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / 'figures/fig8b_cofactor_simple.png', facecolor=BG)
    # white copy for the written report
    ax.set_facecolor('white')
    fig.savefig(HERE / 'figures/fig8b_cofactor_simple_report.png', facecolor='white')
    print(rows)


if __name__ == '__main__':
    main()
