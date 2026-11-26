"""Pairwise distance descriptors between 5 landmark residues of the AF-2 site (prof's request, Nov 2026).

For every receptor-peptide unit of the 6-descriptor table (clustering_bota/clustered_table_all_units.csv):
  K   charge-clamp Lys on H3: the Lys (outside H12) whose NZ is closest to the backbone O of motif residues 4-5
  E   AF-2 Glu on H12: the Glu in the H12 range (±3) whose CD is closest to the backbone N of motif residue 1
  L1, L4, L5   motif residues 1, 4 and 5 (the motif window is the one in the 6-descriptor table, e.g. LLQLL)
  C   ligand-pocket centroid (02_pockets_centroids/weighted_pocket_centroids_corrected.csv; missing for the
      8 recovered stapled-peptide ERα entries)
K and E are chosen once per receptor (canonical UniProt number): the residue picked most often in that receptor's
coactivator units. The same residue is then used in every unit, including corepressor complexes where H12 moves.
Sanity rule: if a receptor's Lys is on average > 12 A from motif L5 (CA-CA), it is not a clamp; its d_K_* are left
empty (LRH-1).
Descriptors = the 15 distances between the 6 landmarks (CA for residues; d_K_E, d_K_L1, ..., d_L5_C; Angstrom).
Writes pairwise_landmarks.csv (one row per unit) and pairwise_landmark_residues.csv (chosen K and E per receptor).
Run: python pairwise_landmarks.py   (about 5 min, 4 processes; GEOM_WORKERS to change)
"""
import itertools
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.SeqUtils import seq1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRUCTURES = ROOT / 'raw/structures_718_cofactor'
UNITS = HERE / 'clustering_bota/clustered_table_all_units.csv'
LANDMARKS = ['K', 'E', 'L1', 'L4', 'L5', 'C']
LABEL = {'K': 'Lys clamp (H3)', 'E': 'Glu AF-2 (H12)', 'L1': 'motif L1', 'L4': 'motif L4', 'L5': 'motif L5',
         'C': 'pocket centroid'}
numbering = json.load(open(ROOT / '01_dataset/numbering/canonical_numbering_maps_v2.json'))


def motif_window(pchain, rchain, motif):
    """The contiguous window matching the table's motif (X = any residue) with the most receptor contacts."""
    res = [r for r in pchain if (r.id[0] == ' ' or r.id[0].startswith('H_')) and 'CA' in r]
    seq = ''.join(seq1(r.get_resname(), custom_map={}) or 'X' for r in res)
    pattern = '(?=(' + motif.replace('X', '.') + '))'
    ns = NeighborSearch([a for r in rchain if r.id[0] == ' ' for a in r if a.element != 'H'])
    best, best_n = None, -1
    for m in re.finditer(pattern, seq):
        win = res[m.start():m.start() + 5]
        if len(win) < 5 or any(b.id[1] - a.id[1] != 1 for a, b in zip(win, win[1:])):
            continue
        n = len({b.get_parent() for r in win for a in r for b in ns.search(a.coord, 4.0)})
        if n > best_n:
            best, best_n = win, n
    return best


def process(pdb, rows):
    path = next((p for p in (STRUCTURES / f'{pdb}.cif', ROOT / f'raw/structures_all_1994/{pdb}.cif') if p.exists()),
                None)
    if path is None:
        return [{**{k: r[k] for k in ('pdb_id', 'receptor_chain', 'peptide_chain', 'receptor', 'uniprot_id')},
                 'status': 'ERROR structure file not found'} for r in rows]
    st = MMCIFParser(QUIET=True).get_structure(pdb, str(path))[0]
    out = []
    for row in rows:
        rec = {k: row[k] for k in ('pdb_id', 'receptor_chain', 'peptide_chain', 'receptor', 'uniprot_id')}
        try:
            rc, pc = row['receptor_chain'], row['peptide_chain']
            rchain, pchain = st[rc], st[pc]
            mp = numbering.get(f"{pdb}|{row['receptor']}|{rc}", {})
            p2c = {k: int(v) for k, v in mp.items()}
            win = motif_window(pchain, rchain, row['motif'])
            if win is None:
                raise ValueError('motif window not found')
            rec.update({f'{n}_ca': win[i]['CA'].coord.tolist() for n, i in (('L1', 0), ('L4', 3), ('L5', 4))})
            n_l1 = win[0]['N'].coord if 'N' in win[0] else win[0]['CA'].coord
            o_45 = [r[a].coord for r in win[3:5] for a in ('O', 'C') if a in r] or [win[4]['CA'].coord]
            lys, glu = {}, {}
            for r in rchain:
                if r.id[0] != ' ' or 'CA' not in r:
                    continue
                c = p2c.get(f'{r.id[1]}{r.id[2].strip()}')
                if c is None:
                    continue
                if r.get_resname() == 'LYS':
                    tip = r['NZ'].coord if 'NZ' in r else r['CA'].coord
                    lys[c] = (r['CA'].coord.tolist(), float(min(np.linalg.norm(tip - o) for o in o_45)))
                elif r.get_resname() == 'GLU' and row['h12_lo'] <= c <= row['h12_hi']:
                    tip = r['CD'].coord if 'CD' in r else r['CA'].coord
                    glu[c] = (r['CA'].coord.tolist(), float(np.linalg.norm(tip - n_l1)))
            lys = {c: v for c, v in lys.items() if c < row['h12_lo']}
            rec['lys'] = {c: v[0] for c, v in lys.items() if v[1] < 15}
            rec['glu'] = {c: v[0] for c, v in glu.items()}
            rec['K_nearest'] = min(lys, key=lambda c: lys[c][1]) if lys else None
            rec['E_nearest'] = min(glu, key=lambda c: glu[c][1]) if glu else None
            rec['status'] = 'PASS'
        except Exception as ex:
            rec['status'] = f'ERROR {type(ex).__name__}: {ex}'
        out.append(rec)
    return out


def main():
    u = pd.read_csv(UNITS)
    g = pd.read_csv(ROOT / '04_cofactor_geometry/cofactor_geometry_full.csv')
    uni = g.drop_duplicates('receptor').set_index('receptor').uniprot_id
    u['uniprot_id'] = u.receptor.map(uni)
    h12 = pd.read_csv(ROOT / '03_helices/h12_reference_corrected.csv').set_index('uniprot_id')
    u['h12_lo'] = u.uniprot_id.map(h12.h12_start) - 3
    u['h12_hi'] = u.uniprot_id.map(h12.h12_end) + 3
    groups = {p: d.to_dict('records') for p, d in u.groupby('pdb_id')}
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get('GEOM_WORKERS', 4))) as ex:
        futs = [ex.submit(process, p, r) for p, r in groups.items()]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 100 == 0:
                print(f'{i}/{len(futs)} PDB', flush=True)
    r = pd.DataFrame(rows).merge(u[['pdb_id', 'receptor_chain', 'peptide_chain', 'cofactor_type']],
                                 on=['pdb_id', 'receptor_chain', 'peptide_chain'])
    ok = r[r.status == 'PASS']
    # one K and one E per receptor: the residue most often nearest the motif in its coactivator units
    pick = []
    for rec, d in ok.groupby('receptor'):
        co = d[d.cofactor_type == 'COACTIVATOR']
        src = co if len(co) else d
        k = Counter(src.K_nearest.dropna()).most_common(1)
        e = Counter(src.E_nearest.dropna()).most_common(1)
        pick.append({'receptor': rec, 'K_canonical': k[0][0] if k else None, 'K_votes': f'{k[0][1]}/{len(src)}' if k else '',
                     'E_canonical': e[0][0] if e else None, 'E_votes': f'{e[0][1]}/{len(src)}' if e else ''})
    pick = pd.DataFrame(pick)
    pick.to_csv(HERE / 'pairwise_landmark_residues.csv', index=False)
    kmap, emap = pick.set_index('receptor').K_canonical.to_dict(), pick.set_index('receptor').E_canonical.to_dict()
    cen = pd.read_csv(ROOT / '02_pockets_centroids/weighted_pocket_centroids_corrected.csv')
    cmap = {(p, c): [x, y, z] for p, c, x, y, z in
            zip(cen.pdb_id, cen.receptor_chain, cen.centroid_x, cen.centroid_y, cen.centroid_z)}
    out = []
    for _, x in r.iterrows():
        row = {k: x[k] for k in ('pdb_id', 'receptor_chain', 'peptide_chain', 'receptor')}
        if x.status == 'PASS':
            pts = {'K': x.lys.get(kmap.get(x.receptor)) if isinstance(x.lys, dict) else None,
                   'E': x.glu.get(emap.get(x.receptor)) if isinstance(x.glu, dict) else None,
                   'L1': x.L1_ca, 'L4': x.L4_ca, 'L5': x.L5_ca, 'C': cmap.get((x.pdb_id, x.receptor_chain))}
            for a, b in itertools.combinations(LANDMARKS, 2):
                row[f'd_{a}_{b}'] = (float(np.linalg.norm(np.subtract(pts[a], pts[b])))
                                     if pts[a] is not None and pts[b] is not None else np.nan)
        row['status'] = x.status
        out.append(row)
    out = pd.DataFrame(out)
    # sanity rule: a real charge clamp sits next to the motif C-terminus (CA-CA to L5 about 7-8 A in coactivator
    # units). If the chosen Lys is on average farther than 12 A, the receptor has no Lys clamp at that place
    # (LRH-1): its Lys distances are left empty instead of measuring a wrong residue.
    co = out.merge(u[['pdb_id', 'receptor_chain', 'peptide_chain', 'cofactor_type']],
                   on=['pdb_id', 'receptor_chain', 'peptide_chain'])
    med = co[co.cofactor_type == 'COACTIVATOR'].groupby('receptor').d_K_L5.median()
    bad = med[med > 12].index.tolist()
    pick['K_median_CA_to_L5_A'] = pick.receptor.map(med).round(1)
    pick['K_used'] = ~pick.receptor.isin(bad)
    pick.to_csv(HERE / 'pairwise_landmark_residues.csv', index=False)
    out.loc[out.receptor.isin(bad), [c for c in out if c.startswith('d_K_')]] = np.nan
    print('Lys clamp not used for:', bad)
    out.to_csv(HERE / 'pairwise_landmarks.csv', index=False)
    print(out.status.value_counts().to_string())
    print(out.filter(like='d_').describe().round(1).T.to_string())
    print(pick.to_string(index=False))


if __name__ == '__main__':
    main()
