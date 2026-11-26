"""Exclusion audit: why every structure or peptide left the dataset (prof, 28 Oct: "do not exclude
blindly; understand why each one is out, there may be errors").

Pipeline stages (inputs: GitHub snapshot + Student A/B handoff tables):
  1 parse        mmCIF could not be parsed
  2 zinc         >= 2 Zn atoms -> treated as DNA-binding domain
  3 no_peptide   no chain of 4-49 residues
  4 no_motif     peptide without an exact LXXLL / CoRNR match, or without receptor contact
  5 curation     receptor-chain identity / manual curation (Student A notebook)
  6 geometry_qc  peptide not in the AF-2 groove (04_cofactor_geometry/qc_flags.py)

verdict: 'correct'      the reason is right, the structure has no usable cofactor complex
         'likely_error' a real cofactor complex was lost by a rule
         'review'       cannot be decided automatically; look at the structure

Output: exclusion_audit.csv (one row per excluded receptor-PDB or peptide) and
        exclusion_audit_summary.csv (counts per stage, category, verdict).
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAP = ROOT.parent / 'BIOL363-NR-Cofactor-Analysis/data/snapshot'

DBD_MAX_LEN = 150          # a DBD-only chain is ~70-100 residues; longer = DBD + LBD (multidomain)
FUSION = r'fused|fusion|tether|-SRC\d?\(|-L\d+-SRC|SRC1\(|chimer'
NO_COFACTOR = r'without (a )?co-?activator|without co-?activator|apo\b'
MENTIONS_COFACTOR = r'co-?activator|co-?repressor|peptide|SRC|NCOA|TIF2|GRIP|PGC|SMRT|NCoR|MED1|TRAP220|CBP|LXXLL'


def longest(chains):
    return max([int(x.split(':')[1]) for x in str(chains).split(';') if ':' in x] or [0])


def stage_zinc_parse(f, titles):
    rows = []
    for r in f[f.status != 'SUCCESS'].itertuples():
        rows.append({'stage': '1 parse', 'pdb_id': r.pdb_id, 'receptor': r.receptor,
                     'category': 'mmCIF parse error', 'detail': r.error, 'verdict': 'likely_error'})
    z = f[(f.status == 'SUCCESS') & f.zn_finger_dbd_flag]
    for r in z.itertuples():
        n = longest(r.receptor_chains)
        multi = n > DBD_MAX_LEN
        rows.append({'stage': '2 zinc', 'pdb_id': r.pdb_id, 'receptor': r.receptor,
                     'category': 'multidomain receptor (DBD + LBD)' if multi else 'DNA-binding domain only',
                     'detail': f'{r.n_zinc} Zn; longest chain {n} aa; peptides {r.peptide_chains if pd.notna(r.peptide_chains) else "none"}; '
                               f'LXXLL {r.has_LXXLL}',
                     'verdict': ('likely_error' if r.has_LXXLL or r.has_CoRNR else 'review') if multi else 'correct',
                     'title': titles.get((r.pdb_id, r.receptor), '')})
    return rows


def stage_no_peptide(f, titles):
    rows = []
    z = f[(f.status == 'SUCCESS') & ~f.zn_finger_dbd_flag & ~f.has_peptide]
    for r in z.itertuples():
        t = titles.get((r.pdb_id, r.receptor), '')
        if re.search(FUSION, t, re.I):
            cat, v = 'cofactor fused to the receptor chain', 'likely_error'
        elif re.search(NO_COFACTOR, t, re.I):
            cat, v = 'no cofactor in structure (stated in title)', 'correct'
        elif re.search(MENTIONS_COFACTOR, t, re.I):
            cat, v = 'title mentions a cofactor but no peptide chain', 'review'
        else:
            cat, v = 'no cofactor peptide in structure', 'correct'
        rows.append({'stage': '3 no_peptide', 'pdb_id': r.pdb_id, 'receptor': r.receptor, 'category': cat,
                     'detail': f'chains {r.receptor_chains}', 'verdict': v, 'title': t})
    return rows


def classify_peptide(seq, desc, receptor_contact):
    s, d = str(seq), str(desc)
    if not receptor_contact:
        return 'no receptor contact', 'review'
    if 'X' in s:
        if re.search(r'L..LL|[IL]..[IV]I', s.replace('X', 'L')) or re.search('stapled|coactivator|SRC', d, re.I):
            return 'modified-residue (stapled) coactivator peptide', 'likely_error'
        return 'modified-residue peptide, not a cofactor (e.g. phosphopeptide)', 'correct'
    if re.search('F..LF', s):
        # Prof's decision (28 Oct 2026): the project uses only LXXLL and CoRNR motifs.
        return 'FxxLF motif (AR-type coactivator); out of scope: only LXXLL and CoRNR (prof, 28 Oct)', 'correct'
    if re.search('L..[IVMF]L|L..L[IVMF]|[IVMF]..LL', s):
        # Prof's decision (29 Oct 2026): only exact LXXLL / CoRNR, no one-substitution variants.
        return 'LXXLL variant (one hydrophobic swapped); out of scope: exact LXXLL/CoRNR only (prof)', 'correct'
    if len(s) < 8:
        return 'very short peptide, motif not resolved', 'review'
    return 'no recognisable cofactor motif', 'review'


def stage_no_motif(p, kept):
    rows = []
    key = ['pdb_id', 'peptide_chain']
    ex = p.merge(kept[key].drop_duplicates().assign(_k=1), on=key, how='left')
    ex = ex[ex._k.isna()]
    for r in ex.itertuples():
        cat, v = classify_peptide(r.peptide_sequence, r.rcsb_description, bool(r.receptor_contact))
        rows.append({'stage': '4 no_motif', 'pdb_id': r.pdb_id, 'receptor': r.receptor, 'peptide_chain': r.peptide_chain,
                     'category': cat, 'detail': f'{r.peptide_sequence} | {r.rcsb_description}', 'verdict': v})
    return rows


def stage_curation(handoff, clean):
    key = ['pdb_id', 'peptide_chain']
    gone = handoff.merge(clean[key].drop_duplicates().assign(_k=1), on=key, how='left')
    gone = gone[gone._k.isna()]
    rows = []
    for r in gone.itertuples():
        if r.pdb_id in ('7AOS', '1XLS'):
            cat, v = 'non-human receptor chain', 'correct'
        else:
            cat, v = 'false SHP receptor label; long chain may be GR or ancestral corticoid receptor', 'review'
        rows.append({'stage': '5 curation', 'pdb_id': r.pdb_id, 'receptor': r.receptor, 'peptide_chain': r.peptide_chain,
                     'category': cat, 'detail': f'receptor chain {r.receptor_chain}', 'verdict': v})
    return rows


def stage_geometry():
    g = pd.read_csv(ROOT / '04_cofactor_geometry/cofactor_geometry_full.csv')
    rows = []
    for r in g[g.qc_flag == 'exclude'].itertuples():
        rows.append({'stage': '6 geometry_qc', 'pdb_id': r.pdb_id, 'receptor': r.receptor, 'peptide_chain': r.peptide_chain,
                     'category': 'peptide not in AF-2 groove', 'detail': r.qc_reason, 'verdict': 'review'})
    return rows


def main():
    f = pd.read_csv(SNAP / 'structure_filter_results.csv')
    inv = pd.read_csv(SNAP / 'all_nr_structures.csv')
    titles = {(r.pdb_id, r.receptor): str(r.title) for r in inv.itertuples()}
    p = pd.read_csv(SNAP / 'peptide_cofactor_validation_FIXED.csv')
    kept = pd.read_csv(SNAP / 'confirmed_cofactor_complexes_FIXED.csv')
    handoff = pd.read_csv(HERE / 'cofactor_complexes_handoff.csv')
    clean = pd.read_csv(HERE / 'cofactor_complexes_handoff_clean.csv')

    rows = (stage_zinc_parse(f, titles) + stage_no_peptide(f, titles) + stage_no_motif(p, kept)
            + stage_curation(handoff, clean) + stage_geometry())
    audit = pd.DataFrame(rows)[['stage', 'verdict', 'category', 'pdb_id', 'receptor', 'peptide_chain', 'detail', 'title']]
    audit['manual_verdict'] = ''
    audit['note'] = ''
    audit = audit.sort_values(['stage', 'verdict', 'category', 'receptor', 'pdb_id'])
    audit.to_csv(HERE / 'exclusion_audit.csv', index=False)

    summary = (audit.groupby(['stage', 'category', 'verdict'])
               .agg(rows=('pdb_id', 'size'), pdbs=('pdb_id', 'nunique')).reset_index())
    summary.to_csv(HERE / 'exclusion_audit_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(summary.to_string(index=False))
    print(audit.groupby('verdict').pdb_id.nunique().to_string())


if __name__ == '__main__':
    main()
