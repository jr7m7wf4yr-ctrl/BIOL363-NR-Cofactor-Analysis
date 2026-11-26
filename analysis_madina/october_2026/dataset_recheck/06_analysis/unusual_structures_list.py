"""List of unusual structures in the 716-PDB base, kept for now (professor's decision, Nov 2026):
  large crystal     the 2 structures with 6 or 8 receptor-peptide copies in the crystal (fig12 '6 or 8')
  extra peptide     the 4 extra SRC-2 peptides (in 3 PDB) that sit outside the AF-2 groove (fig12 'extra peptide')
  weaker data       the 29 structures with resolution > 3 Å or R-free > 0.30 (fig10 gold points)
Writes unusual_structures_list.csv, one row per peptide for 'extra peptide' and one per PDB otherwise.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# the extra copy in each structure: few contacts or far from H12 (see fig12)
EXTRA = {('1YOK', 'C'), ('1ZDU', 'Q'), ('9H66', 'F'), ('9H66', 'I')}


def main():
    m = pd.read_csv(ROOT / '05_metadata_ligands/structure_metadata.csv')
    m = m[m.in_geometry_set]
    geo = pd.read_csv(ROOT / '04_cofactor_geometry/cofactor_geometry_full.csv')
    cols = ['pdb_id', 'receptors', 'receptor_chains', 'cofactor_peptides', 'resolution_A', 'R_free', 'space_group']

    big = m[m.receptor_chains >= 6][cols].assign(
        group='large crystal', reason=lambda d: d.receptor_chains.astype(int).astype(str) + ' receptor chains, '
        + d.cofactor_peptides.astype(int).astype(str) + ' peptides (one per chain)')

    g = geo[[(p, c) in EXTRA for p, c in zip(geo.pdb_id, geo.peptide_chain)]]
    extra = g[['pdb_id', 'peptide_chain', 'receptor_chain', 'n_contact_residues', 'h12_center_to_motif_A',
               'qc_flag', 'qc_reason']].merge(m[cols], on='pdb_id').assign(
        group='extra peptide', reason=lambda d: 'second SRC-2 copy on receptor chain ' + d.receptor_chain
        + ', outside the AF-2 groove')

    weak_res, weak_rfree = m.resolution_A > 3.0, m.R_free > 0.30
    weak = m[weak_res | weak_rfree][cols].assign(group='weaker data')
    weak['reason'] = [' + '.join(r for r, hit in [('resolution > 3 Å', a), ('R-free > 0.30', b)] if hit)
                      for a, b in zip(weak_res[weak.index], weak_rfree[weak.index])]

    out = pd.concat([big, extra, weak], ignore_index=True)
    groups = out.groupby('pdb_id').group.agg(lambda s: '; '.join(dict.fromkeys(s)))
    out['also_in'] = [';'.join(x for x in groups[p].split('; ') if x != grp) for p, grp in zip(out.pdb_id, out.group)]
    out['decision'] = 'kept for now (professor, Nov 2026)'
    out = out[['group', 'pdb_id', 'receptors', 'peptide_chain', 'receptor_chain', 'reason', 'resolution_A', 'R_free',
               'space_group', 'receptor_chains', 'cofactor_peptides', 'n_contact_residues', 'h12_center_to_motif_A',
               'qc_flag', 'also_in', 'decision']]
    out = out.sort_values(['group', 'pdb_id', 'peptide_chain'], key=lambda s: s.map(
        {'large crystal': 0, 'extra peptide': 1, 'weaker data': 2}) if s.name == 'group' else s)
    out = out.round({'resolution_A': 2, 'R_free': 3, 'h12_center_to_motif_A': 1})
    # utf-8-sig so Excel shows Å correctly
    out.to_csv(HERE / 'unusual_structures_list.csv', index=False, encoding='utf-8-sig')
    print(out.group.value_counts().to_string())
    print(out.drop(columns=['decision']).to_string(index=False))


if __name__ == '__main__':
    main()
