"""Experimental metadata for every structure in the cofactor dataset (prof's notes, 28 Oct:
"metadata", "experimental procedures", "space group symmetry", "crystal packing parameters",
"stoichiometry").

Read from the "header" of each mmCIF file (not the coordinates). One file at a time, low memory.
Stoichiometry comes from the curated handoff: receptor chains and cofactor peptides per PDB.

Output: structure_metadata.csv (one row per PDB) and a printed summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CIF = ROOT / 'raw/structures_718_cofactor'
# The 8 ERα structures with stapled SRC-2 peptides, recovered in the 924 -> 718 re-check. Their pairs come from
# the contact check (peptide chain + contacted ERα chain); their mmCIF files are in the full download.
INCLUDE_STAPLED = True
CIF_ALL = ROOT / 'raw/structures_all_1994'

FIELDS = {
    'title': '_struct.title',
    'method': '_exptl.method',
    'deposition_date': '_pdbx_database_status.recvd_initial_deposition_date',
    'resolution_A': '_refine.ls_d_res_high',
    'R_work': '_refine.ls_R_factor_R_work',
    'R_free': '_refine.ls_R_factor_R_free',
    'space_group': '_symmetry.space_group_name_H-M',
    'space_group_number': '_symmetry.Int_Tables_number',
    'cell_a_A': '_cell.length_a', 'cell_b_A': '_cell.length_b', 'cell_c_A': '_cell.length_c',
    'cell_alpha': '_cell.angle_alpha', 'cell_beta': '_cell.angle_beta', 'cell_gamma': '_cell.angle_gamma',
    'Z_PDB': '_cell.Z_PDB',
    'matthews_coeff': '_exptl_crystal.density_Matthews',
    'solvent_percent': '_exptl_crystal.density_percent_sol',
    'temperature_K': '_diffrn.ambient_temp',
    'crystal_method': '_exptl_crystal_grow.method',
    'crystal_pH': '_exptl_crystal_grow.pH',
    'crystal_temp_K': '_exptl_crystal_grow.temp',
    'crystal_conditions': '_exptl_crystal_grow.pdbx_details',
    'radiation_source': '_diffrn_source.source',
}
NUMERIC = ['space_group_number', 'resolution_A', 'R_work', 'R_free', 'cell_a_A', 'cell_b_A', 'cell_c_A', 'cell_alpha', 'cell_beta',
           'cell_gamma', 'Z_PDB', 'matthews_coeff', 'solvent_percent', 'temperature_K', 'crystal_pH', 'crystal_temp_K']


def first(d, key):
    v = d.get(key)
    if v is None:
        return None
    v = v[0] if isinstance(v, list) else v
    return None if v in ('?', '.') else ' '.join(str(v).split())


def crystal_system(number):
    """Crystal system from the International Tables space-group number."""
    if pd.isna(number):
        return None
    n = int(number)
    for upper, name in [(2, 'triclinic'), (15, 'monoclinic'), (74, 'orthorhombic'), (142, 'tetragonal'),
                        (167, 'trigonal'), (194, 'hexagonal'), (230, 'cubic')]:
        if n <= upper:
            return name
    return None


def main():
    clean = pd.read_csv(ROOT / '01_dataset/cofactor_complexes_handoff_clean.csv')
    files = {p.stem: p for p in CIF.glob('*.cif')}
    stapled = []
    if INCLUDE_STAPLED:
        lost = pd.read_csv(ROOT / '01_dataset/peptide_contact_check/lost_peptides_contact_check.csv')
        lost = lost[lost.group.str.startswith('A ')]
        stapled = sorted(lost.pdb_id.unique())
        clean = pd.concat([clean, lost.rename(columns={'contacted_chain': 'receptor_chain'}).assign(
            receptor='ESR1', cofactor_type='COACTIVATOR')], ignore_index=True)
        files.update({p: CIF_ALL / f'{p}.cif' for p in stapled})
    stoich = clean.groupby('pdb_id').agg(
        receptors=('receptor', lambda s: ';'.join(sorted(set(s)))),
        receptor_chains=('receptor_chain', 'nunique'),
        cofactor_peptides=('peptide_chain', 'nunique'),
        coactivator_peptides=('cofactor_type', lambda s: (s == 'COACTIVATOR').sum()),
        corepressor_peptides=('cofactor_type', lambda s: (s == 'COREPRESSOR').sum()))
    rows = []
    pdbs = sorted(files)
    for i, pdb in enumerate(pdbs, 1):
        d = MMCIF2Dict(str(files[pdb]))
        row = {'pdb_id': pdb, **{k: first(d, v) for k, v in FIELDS.items()}}
        strands = d.get('_pdbx_poly_seq_scheme.pdb_strand_id', [])
        row['polymer_chains_in_asu'] = len(set(strands if isinstance(strands, list) else [strands]))
        rows.append(row)
        if i % 100 == 0:
            print(f'{i}/{len(pdbs)}', flush=True)
    meta = pd.DataFrame(rows)
    for c in NUMERIC:
        meta[c] = pd.to_numeric(meta[c], errors='coerce')
    meta['deposition_year'] = pd.to_numeric(meta.deposition_date.str[:4], errors='coerce')
    meta['crystal_system'] = meta.space_group_number.map(crystal_system)
    meta['R_gap'] = meta.R_free - meta.R_work
    meta = meta.merge(stoich, left_on='pdb_id', right_index=True, how='left')
    meta['peptides_per_receptor_chain'] = meta.cofactor_peptides / meta.receptor_chains
    meta['in_geometry_set'] = meta.receptor_chains.notna()
    meta['stapled'] = meta.pdb_id.isin(stapled)
    meta.to_csv(HERE / 'structure_metadata.csv', index=False)

    m = meta[meta.in_geometry_set]
    pd.set_option('display.width', 250)
    print('\nstructures:', len(meta), '| in geometry set:', len(m))
    print('method:', m.method.value_counts().to_dict())
    print(m[['resolution_A', 'R_work', 'R_free', 'R_gap', 'solvent_percent', 'temperature_K', 'crystal_pH']]
          .describe().round(3).to_string())
    print('\nmissing values (%):', (m[list(FIELDS)].isna().mean() * 100).round(1).to_dict())
    print('\ncrystal system:', m.crystal_system.value_counts().to_dict())
    print('top space groups:', m.space_group.value_counts().head(8).to_dict())
    print('\nreceptor chains : cofactor peptides per PDB:')
    print(m.groupby(['receptor_chains', 'cofactor_peptides']).size().rename('pdbs').to_string())


if __name__ == '__main__':
    main()
