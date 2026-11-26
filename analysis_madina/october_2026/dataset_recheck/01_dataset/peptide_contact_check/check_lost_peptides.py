"""Where do the peptides lost between 924 and 718 PDB bind? (team request, 29 Oct:
"somewhere in between 924 -> 718 we excluded parts that were actually necessary, regarding the
cofactor peptide contact; check those structures")

All 206 PDB lost at that step have a peptide that contacts a long chain (receptor_contact = True)
but no exact LXXLL / CoRNR motif. For each such peptide:
  1 is the contacted long chain really the nuclear receptor? Local alignment of the chain to the
    receptor's human UniProt sequence (identity >= 0.9 over >= 100 aligned residues).
  2 if yes: map the contacted residues to canonical UniProt numbering and compare them with the
    receptor's AF-2 groove = residues contacted by the cofactor motif in >= 50% of that receptor's
    LXXLL complexes in our geometry dataset (04_cofactor_geometry, qc_flag ok).
    groove_coverage = fraction of the groove residues the peptide touches.

Classes:
  'binds AF-2 coactivator groove'   groove_coverage >= 0.5   -> cofactor-site binder without exact motif
  'touches part of the groove'      0.2-0.5                 -> review
  'binds receptor elsewhere'        < 0.2                   -> correct exclusion
  'long chain is not the receptor'  e.g. 14-3-3 with a receptor-derived phosphopeptide
  'no groove reference'             receptor has no LXXLL complexes in our dataset

Output: lost_peptides_contact_check.csv, printed summary.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio import Align
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = ROOT.parent / 'BIOL363-NR-Cofactor-Analysis/data/snapshot'
CIF = ROOT / 'raw/structures_all_1994'
INVENTORY_SCRIPT = ROOT.parent / 'BIOL363-NR-Cofactor-Analysis/scripts/01_inventory.py'
MIN_IDENTITY, MIN_ALIGNED = 0.9, 100
GROOVE_FREQ = 0.5

aligner = Align.PairwiseAligner()
aligner.mode = 'local'
aligner.match_score, aligner.mismatch_score = 2, -1
aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5


def residue_label(r):
    return f'{r.id[1]}{r.id[2].strip()}'


def map_chain(chain, uniprot_seq):
    """Deposited residue label -> canonical UniProt position, with identity of the aligned part."""
    res = [r for r in chain if r.id[0] == ' ']
    seq = ''.join(seq1(r.get_resname(), custom_map={}) or 'X' for r in res)
    if len(seq) < 30:
        return {}, 0.0, 0
    aln = aligner.align(seq, uniprot_seq)[0]
    mapping, same = {}, 0
    for (s0, s1), (u0, u1) in zip(*aln.aligned):
        for k in range(s1 - s0):
            mapping[residue_label(res[s0 + k])] = u0 + k + 1
            same += seq[s0 + k] == uniprot_seq[u0 + k]
    n = len(mapping)
    return mapping, (same / n if n else 0.0), n


def groove_signatures():
    geo = pd.read_csv(ROOT / '04_cofactor_geometry/cofactor_geometry_full.csv')
    g = geo[(geo.qc_flag == 'ok') & (geo.motif_type == 'LXXLL')]
    sig = {}
    for rec, d in g.groupby('receptor'):
        cnt = Counter(int(x) for s in d.contact_residues_canonical.dropna() for x in str(s).split(';') if x)
        sig[rec] = {r for r, c in cnt.items() if c / len(d) >= GROOVE_FREQ}
    return sig


def main():
    ids = dict(re.findall(r'"(\w+)": "([A-Z0-9]{6})"', INVENTORY_SCRIPT.read_text(encoding='utf-8')))
    seqs = json.loads((ROOT / '01_dataset/numbering/uniprot_sequences_all48.json').read_text())
    basic = pd.read_csv(SNAP / 'basic_filtered_structures.csv')
    pep = pd.read_csv(SNAP / 'peptide_cofactor_validation_FIXED.csv')
    kept = pd.read_csv(SNAP / 'confirmed_cofactor_complexes_FIXED.csv')
    lost = sorted(set(basic.pdb_id) - set(kept.pdb_id))
    rows_in = pep[pep.pdb_id.isin(lost)]
    sig = groove_signatures()
    parser = MMCIFParser(QUIET=True)
    out, cache = [], {}
    for r in rows_in.itertuples():
        contacts = [c.split(':') for c in str(r.contact_residues).split(';') if ':' in c]
        by_chain = Counter(ch for ch, _ in contacts)
        main_chain = by_chain.most_common(1)[0][0] if by_chain else None
        acc = ids.get(r.receptor)
        row = {'pdb_id': r.pdb_id, 'receptor': r.receptor, 'peptide_chain': r.peptide_chain,
               'peptide_sequence': r.peptide_sequence, 'rcsb_description': r.rcsb_description,
               'final_cofactor_confidence': r.final_cofactor_confidence, 'n_contacts': r.contact_residue_count,
               'contacted_chain': main_chain, 'chain_identity_to_receptor': None, 'aligned_residues': None,
               'groove_residues_touched': None, 'groove_size': None, 'groove_coverage': None}
        if main_chain is None or acc not in seqs:
            row['class'] = 'no contact data or receptor sequence'
            out.append(row)
            continue
        key = (r.pdb_id, main_chain, acc)
        if key not in cache:
            st = parser.get_structure(r.pdb_id, str(CIF / f'{r.pdb_id}.cif'))
            cache[key] = map_chain(st[0][main_chain], seqs[acc]) if main_chain in st[0] else ({}, 0.0, 0)
        mapping, ident, n = cache[key]
        row.update({'chain_identity_to_receptor': round(ident, 3), 'aligned_residues': n})
        if ident < MIN_IDENTITY or n < MIN_ALIGNED:
            row['class'] = 'long chain is not the receptor'
        elif not sig.get(r.receptor):
            row['class'] = 'no groove reference'
        else:
            canon = {mapping[re.sub(r'^[A-Z]{3}', '', lab)] for ch, lab in contacts
                     if ch == main_chain and re.sub(r'^[A-Z]{3}', '', lab) in mapping}
            g = sig[r.receptor]
            cov = len(canon & g) / len(g)
            row.update({'groove_residues_touched': len(canon & g), 'groove_size': len(g), 'groove_coverage': round(cov, 2)})
            row['class'] = ('binds AF-2 coactivator groove' if cov >= 0.5 else
                            'touches part of the groove' if cov >= 0.2 else 'binds receptor elsewhere')
        out.append(row)
    df = pd.DataFrame(out).sort_values(['class', 'receptor', 'pdb_id'])
    df.to_csv(HERE / 'lost_peptides_contact_check.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 45)
    print('peptides:', len(df), '| PDB:', df.pdb_id.nunique())
    print(df.groupby('class').agg(peptides=('pdb_id', 'size'), pdbs=('pdb_id', 'nunique')).to_string())
    for c in ['binds AF-2 coactivator groove', 'touches part of the groove']:
        d = df[df['class'] == c]
        print(f'\n## {c}: {len(d)} peptides')
        print(d[['pdb_id', 'receptor', 'peptide_sequence', 'rcsb_description', 'groove_coverage', 'n_contacts']]
              .to_string(index=False))
    print('\n## not the receptor, by description:')
    print(df[df['class'] == 'long chain is not the receptor'].rcsb_description.value_counts().head(10).to_string())


if __name__ == '__main__':
    main()
