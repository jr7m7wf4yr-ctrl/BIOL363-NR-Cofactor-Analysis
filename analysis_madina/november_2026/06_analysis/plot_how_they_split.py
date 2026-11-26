"""fig17: how PPARγ, FXR and ERα really split, when not by peptide (Nov 2026).

Each panel: every unit of the receptor on the two descriptors that separate its groups, the H3–H12 angle and the
H3–H12 distance (6-descriptor table). Groups as in fig16 (HDBSCAN within the receptor, min size max(4, 5%)).
  PPARγ  coactivator vs corepressor; coactivators coloured by peptide (PGC-1α LKKLL vs the rest)
  FXR    the small HDBSCAN group (6 structures of one ligand series, crystal form P6₅) vs the rest;
         rings = all P6₅ units, to show that only half of that crystal form moves
  ERα    one cloud, no groups
Writes figures/fig17_how_they_split.png.
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
GREEN, GOLD, DARK, INK, MUTED, GRID, BG = '#467a26', '#b8860b', '#3d3d3a', '#263b1f', '#6d726b', '#e3e6dc', '#f9f9f3'
F = ['angle_H3_H12', 'angle_H12_motif', 'angle_H3_motif', 'dist_H12_motif', 'dist_H3_motif', 'dist_H3_H12']
X, Y = 'angle_H3_H12', 'dist_H3_H12'


def groups(r):
    k = max(4, round(0.05 * len(r)))
    return HDBSCAN(min_cluster_size=k, min_samples=k).fit_predict(StandardScaler().fit_transform(r[F]))


def style(ax, title, sub):
    ax.set_facecolor(BG)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c9cdc2')
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel('H3–H12 angle (°)', fontsize=13, color=MUTED)
    ax.set_ylabel('H3–H12 distance (Å)', fontsize=13, color=MUTED)
    ax.set_title(title, loc='left', fontsize=15, fontweight='bold', color=INK, pad=30)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=11.5, color=MUTED, va='bottom')


def label(ax, pts, text, colour, dx=0, dy=0):
    x, y = np.median(pts[X]), np.median(pts[Y])
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=12.5, fontweight='bold',
                color=colour, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=colour, lw=1.2))


def main():
    d = pd.read_csv(HERE / 'clustering_bota/clustered_table_all_units.csv')
    meta = pd.read_csv(ROOT / '05_metadata_ligands/structure_metadata.csv')[['pdb_id', 'space_group']]
    d = d.merge(meta.drop_duplicates('pdb_id'), on='pdb_id', how='left')
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans']})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor(BG)

    # PPARγ
    ax = axes[0]
    r = d[d.receptor == 'PPARG']
    co = r[r.cofactor_type == 'COACTIVATOR']
    parts = [(co[co.motif != 'LKKLL'], GREEN, f'coactivator, other peptides ({(co.motif != "LKKLL").sum()})'),
             (co[co.motif == 'LKKLL'], GOLD, f'coactivator PGC-1α, LKKLL ({(co.motif == "LKKLL").sum()})'),
             (r[r.cofactor_type == 'COREPRESSOR'], DARK, f'corepressor ({(r.cofactor_type == "COREPRESSOR").sum()})')]
    for p, c, lab in parts:
        ax.scatter(p[X], p[Y], s=30, color=c, alpha=0.8, linewidths=0.4, edgecolors=BG, label=lab)
    cor = r[r.cofactor_type == 'COREPRESSOR']
    label(ax, co, f'{np.median(co[X]):.0f}° · {np.median(co[Y]):.0f} Å', GREEN, dx=-70, dy=30)
    label(ax, cor, f'{np.median(cor[X]):.0f}° · {np.median(cor[Y]):.0f} Å', DARK, dx=120, dy=-6)
    style(ax, 'PPARγ: the corepressor moves H12 away', 'H12 turns and moves 11 Å from H3; PGC-1α sits among the rest')
    ax.legend(fontsize=10.5, frameon=False, loc='center', labelcolor=INK)

    # FXR
    ax = axes[1]
    r = d[d.receptor == 'FXR'].reset_index(drop=True)
    lab = groups(r)
    small = min(set(lab[lab >= 0]), key=lambda c: np.sum(lab == c))
    main_g, small_g = r[lab != small], r[lab == small]
    ax.scatter(main_g[X], main_g[Y], s=30, color=GREEN, alpha=0.75, linewidths=0.4, edgecolors=BG,
               label=f'main group ({len(main_g)})')
    ax.scatter(small_g[X], small_g[Y], s=46, color=GOLD, linewidths=0.4, edgecolors=BG,
               label=f'small group: one ligand series ({len(small_g)})')
    p65 = r[r.space_group == 'P 65']
    ax.scatter(p65[X], p65[Y], s=110, facecolors='none', edgecolors=INK, linewidths=1.1,
               label=f'ring = crystal form P6₅ ({len(p65)})')
    label(ax, main_g, f'{np.median(main_g[X]):.0f}°', GREEN, dx=0, dy=-46)
    label(ax, small_g, f'{np.median(small_g[X]):.0f}°', GOLD, dx=0, dy=34)
    style(ax, 'FXR: one ligand series tilts H12', 'all have the same peptide (LRYLL); 6 of the 12 P6₅ units tilt')
    ax.legend(fontsize=10.5, frameon=False, loc='center right', labelcolor=INK)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # ERα
    ax = axes[2]
    r = d[d.receptor == 'ESR1']
    lo, hi = r[[X, Y]].quantile(0.005), r[[X, Y]].quantile(0.995)
    inside = r[(r[X].between(lo[X], hi[X])) & (r[Y].between(lo[Y], hi[Y]))]
    ax.scatter(inside[X], inside[Y], s=22, color=GREEN, alpha=0.55, linewidths=0, label=f'all ERα units ({len(r)})')
    label(ax, inside, f'{np.median(r[X]):.0f}° · {np.median(r[Y]):.0f} Å', GREEN, dx=-60, dy=-45)
    style(ax, 'ERα: one cloud, no groups', f'95% one peptide, 97% one crystal form; same axes as FXR')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=10.5, frameon=False, loc='upper left', labelcolor=INK)

    fig.tight_layout(w_pad=3)
    fig.savefig(HERE / 'figures' / 'fig17_how_they_split.png', dpi=110, facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    main()
