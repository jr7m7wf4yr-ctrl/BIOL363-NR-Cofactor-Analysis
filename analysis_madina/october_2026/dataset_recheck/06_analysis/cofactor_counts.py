"""Cofactor counts (prof's notes, 28 Oct): how many unique cofactors, cofactors per receptor,
receptor-to-cofactor ratio, the whole distribution of cofactors.

Pairs: curated receptor-peptide pairs (01_dataset/cofactor_complexes_handoff_clean.csv, physical
receptor chain verified); pairs that geometry QC placed outside the AF-2 groove are dropped only with DROP_OUTSIDE_GROOVE.
The 6 recovered RORγ pairs (01_dataset/added_structures) are added only with INCLUDE_ADDED = True. Cofactor name: cofactor_v2 from
01_dataset/cofactor_identity/cofactor_identity_v2.csv. Counts are per PDB entry, so chain copies of
one crystal count once; pair counts are given alongside.

Outputs (this folder): cofactor_counts_overall.csv, cofactor_by_receptor.csv (PDB counts),
cofactors_per_receptor.csv, figures/fig8_cofactor_distribution.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from pca_clustering import FIG, HERE, INK, INK2, MUTED, ROOT, SURFACE, style

MOTIF_COLOR = {'LXXLL': '#2a78d6', 'CoRNR': '#eb6834'}
# The 3 RORγ structures found by sequence search (5X8U, 5X8W, 5X8X) are linked to a gibbon UniProt entry;
# the professor decided (Nov) not to include them.
INCLUDE_ADDED = False
# The 8 ERα structures with stapled SRC-2 peptides (lost at the motif step because modified residues read as X).
# They have no UniProt link, so the cofactor name is set by hand.
INCLUDE_STAPLED = True
# The team's October dataset (716 PDB, 1,181 units) keeps the 7 peptides that geometry QC placed outside the
# AF-2 groove (they are flagged for review), so they are kept here too. True drops them (715 PDB).
DROP_OUTSIDE_GROOVE = False
SEQUENTIAL = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])


def load_pairs():
    clean = pd.read_csv(ROOT / '01_dataset/cofactor_complexes_handoff_clean.csv')
    pairs = clean[['pdb_id', 'peptide_chain', 'receptor_chain', 'receptor', 'has_LXXLL', 'has_CoRNR']].copy()
    if INCLUDE_ADDED:
        added = pd.read_csv(ROOT / '01_dataset/added_structures/added_confirmed_cofactor_complexes_FIXED.csv')
        added = added.assign(receptor='RORC', receptor_chain='')[pairs.columns]
        pairs = pd.concat([pairs, added], ignore_index=True)
    if INCLUDE_STAPLED:
        lost = pd.read_csv(ROOT / '01_dataset/peptide_contact_check/lost_peptides_contact_check.csv')
        stapled = lost[lost.group.str.startswith('A ')].rename(columns={'contacted_chain': 'receptor_chain'})
        stapled = stapled.assign(receptor='ESR1', has_LXXLL=True, has_CoRNR=False)[pairs.columns]
        pairs = pd.concat([pairs, stapled], ignore_index=True)
    ident = pd.read_csv(ROOT / '01_dataset/cofactor_identity/cofactor_identity_v2.csv')
    ident = ident.drop_duplicates(['pdb_id', 'peptide_chain'])[['pdb_id', 'peptide_chain', 'cofactor_v2', 'identity_review']]
    pairs = pairs.merge(ident, on=['pdb_id', 'peptide_chain'], how='left')
    geo = pd.read_csv(ROOT / '04_cofactor_geometry/cofactor_geometry_full.csv')
    if DROP_OUTSIDE_GROOVE:
        excluded = set(map(tuple, geo.loc[geo.qc_flag == 'exclude', ['pdb_id', 'peptide_chain']].values))
        pairs = pairs[[(p, c) not in excluded for p, c in zip(pairs.pdb_id, pairs.peptide_chain)]]
    names = geo.drop_duplicates('receptor').set_index('receptor').receptor_display.to_dict()
    names['RORC'] = 'RORγ'
    pairs['receptor_display'] = pairs.receptor.map(names)
    pairs['motif'] = np.where(pairs.has_CoRNR & ~pairs.has_LXXLL, 'CoRNR', 'LXXLL')
    if INCLUDE_STAPLED:
        pairs.loc[pairs.pdb_id.isin(stapled.pdb_id) & pairs.cofactor_v2.isna(), 'cofactor_v2'] = 'SRC-2 (TIF2/GRIP1)'
    pairs['cofactor_v2'] = pairs.cofactor_v2.fillna('unresolved')
    return pairs


def main():
    style()
    pairs = load_pairs()
    named = pairs[pairs.cofactor_v2 != 'unresolved']

    overall = (pairs.groupby('cofactor_v2')
               .agg(pdbs=('pdb_id', 'nunique'), pairs=('pdb_id', 'size'), receptors=('receptor_display', 'nunique'),
                    receptor_list=('receptor_display', lambda s: ', '.join(s.value_counts().index)),
                    motif=('motif', lambda s: s.value_counts().index[0]),
                    identity_review_pairs=('identity_review', lambda s: int(s.fillna(False).sum())))
               .sort_values('pdbs', ascending=False))
    overall.to_csv(HERE / 'cofactor_counts_overall.csv')

    by_rec = pd.crosstab(pairs.receptor_display, pairs.cofactor_v2, values=pairs.pdb_id, aggfunc='nunique').fillna(0).astype(int)
    by_rec = by_rec.loc[by_rec.sum(axis=1).sort_values(ascending=False).index, overall.index]
    by_rec.to_csv(HERE / 'cofactor_by_receptor.csv')

    per_rec = (pairs.groupby('receptor_display')
               .agg(pdbs=('pdb_id', 'nunique'), pairs=('pdb_id', 'size'),
                    unique_cofactors=('cofactor_v2', lambda s: s[s != 'unresolved'].nunique()),
                    most_common=('cofactor_v2', lambda s: s[s != 'unresolved'].value_counts().index[0]
                                 if (s != 'unresolved').any() else '')))
    per_rec['pdbs_per_unique_cofactor'] = (per_rec.pdbs / per_rec.unique_cofactors.replace(0, np.nan)).round(1)
    per_rec = per_rec.sort_values('pdbs', ascending=False)
    per_rec.to_csv(HERE / 'cofactors_per_receptor.csv')

    plot(overall, by_rec, pairs)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 70)
    print('pairs:', len(pairs), '| PDB:', pairs.pdb_id.nunique(), '| receptors:', pairs.receptor_display.nunique(),
          '| unique named cofactors:', named.cofactor_v2.nunique(), '| unresolved pairs:', (pairs.cofactor_v2 == 'unresolved').sum())
    print(overall.drop(columns='receptor_list').to_string())
    print(per_rec.to_string())
    print('receptors per cofactor (mean):', round(overall.drop('unresolved', errors='ignore').receptors.mean(), 2),
          '| cofactors per receptor (mean):', round(per_rec.unique_cofactors.mean(), 2))


def plot(overall, by_rec, pairs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.4), gridspec_kw={'width_ratios': [1, 1.9]})
    o = overall.iloc[::-1]
    colors = [MUTED if n == 'unresolved' else MOTIF_COLOR[m] for n, m in zip(o.index, o.motif)]
    ax1.barh(range(len(o)), o.pdbs, color=colors, height=0.72, edgecolor=SURFACE, linewidth=1)
    for i, (n, v) in enumerate(zip(o.index, o.pdbs)):
        ax1.text(v + 4, i, f'{v}', va='center', fontsize=8, color=INK2)
    ax1.set_yticks(range(len(o)))
    ax1.set_yticklabels(o.index, fontsize=9, color=INK)
    ax1.set_xlabel('PDB entries')
    ax1.grid(axis='y', visible=False)
    ax1.set_xlim(0, o.pdbs.max() * 1.15)
    ax1.set_title('A  How often each cofactor appears')
    handles = [plt.Rectangle((0, 0), 1, 1, color=MOTIF_COLOR['LXXLL'], label='binds via LXXLL'),
               plt.Rectangle((0, 0), 1, 1, color=MOTIF_COLOR['CoRNR'], label='binds via CoRNR'),
               plt.Rectangle((0, 0), 1, 1, color=MUTED, label='name unknown')]
    ax1.legend(handles=handles, loc='lower right', frameon=False, fontsize=8)

    m = by_rec.drop(columns='unresolved', errors='ignore')
    m = m.loc[:, m.sum() > 0]
    im = ax2.imshow(np.where(m.values > 0, m.values, np.nan), cmap=SEQUENTIAL, aspect='auto',
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(1, m.values.max())))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if v:
                ax2.text(j, i, str(v), ha='center', va='center', fontsize=6.5,
                         color='#ffffff' if v >= 20 else INK)
    ax2.set_xticks(range(m.shape[1]))
    ax2.set_xticklabels(m.columns, rotation=55, ha='right', fontsize=8, color=INK)
    ax2.set_yticks(range(m.shape[0]))
    ax2.set_yticklabels([f'{r}  ({by_rec.loc[r].sum()})' for r in m.index], fontsize=8, color=INK)
    ax2.grid(False)
    ax2.set_xticks(np.arange(-.5, m.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-.5, m.shape[0], 1), minor=True)
    ax2.grid(which='minor', color=SURFACE, linewidth=1.5)
    ax2.tick_params(which='both', length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title('B  Which receptor binds which cofactor (PDB entries)')
    fig.suptitle(f'Cofactors in the dataset: {pairs.pdb_id.nunique()} PDB entries, {len(pairs)} receptor–peptide pairs',
                 x=0.01, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.text(0.01, 0.01, 'Counts per PDB entry (chain copies count once). Cofactor named from the peptide\'s UniProt link, '
             'else its PDB description. Blank cell = no structure.', fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(FIG / 'fig8_cofactor_distribution.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
