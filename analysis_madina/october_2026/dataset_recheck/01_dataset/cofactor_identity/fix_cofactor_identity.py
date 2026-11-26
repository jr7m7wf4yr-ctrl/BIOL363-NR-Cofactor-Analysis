"""Cofactor identity, corrected (needed for the professor's cofactor counts, 28 Oct).

The team's identity (05_annotations.py, column cofactor_identity) matches text patterns in the
peptide description + entry title. Its pattern list misses common spellings ("Steroid receptor
coactivator-1", "Nuclear receptor corepressor 2", PGC-1alpha, GRIP peptide, SHP/NR0B2, RIP140...),
so 299 of 1,257 confirmed peptides are 'unresolved' although most are named.

New identity, in order of reliability:
  1 UNIPROT      the peptide entity's UniProt link in the PDB (the protein the peptide comes from)
  2 DESCRIPTION  extended text patterns on the peptide's own description
  3 TITLE        the same patterns on the entry title, only if exactly one cofactor is named
  4 UNRESOLVED   generic names ("steroid receptor coactivator", "SRC peptide"), synthetic peptides
The team's column is kept unchanged; the new one is written next to it.

Inputs: confirmed_cofactor_complexes_FIXED.csv (snapshot); added_structures/added_confirmed_...csv only with INCLUDE_ADDED
Output: cofactor_identity_v2.csv (one row per confirmed peptide), printed comparison.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MIN_IDENTITY = 0.75   # peptide "comes from" a protein if >= 75% of its residues match a protein window
# The 3 RORγ structures found by sequence search (5X8U, 5X8W, 5X8X) are linked to a gibbon UniProt entry;
# the professor decided (Nov) not to include them, so their 6 peptides are left out by default.
INCLUDE_ADDED = False

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
SNAP = DATASET.parent.parent / 'BIOL363-NR-Cofactor-Analysis/data/snapshot'

# UniProt accession -> cofactor (checked against UniProt, 28 Oct 2026). Mouse/rat orthologs map to
# the same cofactor; the peptide source species is kept in uniprot_species.
UNIPROT = {
    'Q15788': ('SRC-1', 'NCOA1', 'human'), 'P70365': ('SRC-1', 'NCOA1', 'mouse'),
    'Q15596': ('SRC-2 (TIF2/GRIP1)', 'NCOA2', 'human'), 'E7EWM1': ('SRC-2 (TIF2/GRIP1)', 'NCOA2', 'human'),
    'Q61026': ('SRC-2 (TIF2/GRIP1)', 'NCOA2', 'mouse'), 'Q9WUI9': ('SRC-2 (TIF2/GRIP1)', 'NCOA2', 'rat'),
    'Q9Y6Q9': ('SRC-3 (AIB1/RAC3)', 'NCOA3', 'human'),
    'Q9HCD5': ('NCOA5', 'NCOA5', 'human'),
    'Q92793': ('CBP', 'CREBBP', 'human'),
    'Q15648': ('MED1 (TRAP220)', 'MED1', 'human'),
    'Q9UBK2': ('PGC-1α', 'PPARGC1A', 'human'), 'Q86YN6': ('PGC-1β', 'PPARGC1B', 'human'),
    'P48552': ('RIP140', 'NRIP1', 'human'),
    'O75376': ('NCoR', 'NCOR1', 'human'), 'Q9Y618': ('SMRT', 'NCOR2', 'human'),
    'Q15466': ('SHP', 'NR0B2', 'human'), 'P51843': ('DAX1', 'NR0B1', 'human'),
    'Q9UQ80': ('PA2G4 (EBP1)', 'PA2G4', 'human'), 'Q15653': ('NFKBIB', 'NFKBIB', 'human'),
}
# Text patterns (case-insensitive), most specific first. A generic "steroid receptor coactivator"
# without a number stays unresolved on purpose.
PATTERNS = [
    ('SRC-1', 'NCOA1', [r'\bncoa-?1\b', r'\bsrc[- ]?1\b', r'steroid receptor coactivator[- ]?1\b',
                        r'nuclear receptor coactivator[- ]?1\b']),
    ('SRC-2 (TIF2/GRIP1)', 'NCOA2', [r'\bncoa-?2\b', r'\bs[rc]{2}[- ]?2\b', r'\btif-?2\b', r'\bgrip-?1?\b',
                                     r'steroid receptor coactivator[- ]?2\b', r'nuclear receptor coactivator[- ]?2\b',
                                     r'transcriptional intermediary factor 2',
                                     r'glucocorticoid receptor[- ]interacting protein']),
    ('SRC-3 (AIB1/RAC3)', 'NCOA3', [r'\bncoa-?3\b', r'\bsrc[- ]?3\b', r'\baib-?1\b', r'\bactr\b', r'\brac-?3\b',
                                    r'steroid receptor coactivator[- ]?3\b', r'nuclear receptor coactivator[- ]?3\b']),
    ('NCOA5', 'NCOA5', [r'nuclear receptor coactivator[- ]?5\b']),
    ('PGC-1β', 'PPARGC1B', [r'coactivator[- ]?1[- ]?(beta|β|b)\b', r'pgc-?1[- ]?(beta|β|b)\b', r'\bppargc1b\b']),
    ('PGC-1α', 'PPARGC1A', [r'coactivator[- ]?1[- ]?(alpha|α|a)\b', r'pgc-?1[- ]?(alpha|alfa|α|a)?\b',
                            r'\bppargc1a\b', r'ppar ?gamma coactivator']),
    ('MED1 (TRAP220)', 'MED1', [r'\bmed1\b', r'trap ?220', r'drip ?205', r'mediator of rna polymerase ii transcription subunit 1\b']),
    ('CBP', 'CREBBP', [r'\bcrebbp\b', r'\bcbp\b', r'creb[- ]binding protein']),
    ('p300', 'EP300', [r'\bep300\b', r'\bp300\b']),
    ('RIP140', 'NRIP1', [r'\bnrip-?1\b', r'\brip-?140\b', r'nuclear receptor[- ]interacting protein 1\b']),
    ('SMRT', 'NCOR2', [r'\bncor-?2\b', r'\bsmrt\b', r'silencing mediator', r'nuclear receptor co-?repressor 2\b']),
    ('NCoR', 'NCOR1', [r'\bncor-?1\b', r'\bn-?cor\b', r'nuclear receptor co-?repressor 1\b']),
    ('SHP', 'NR0B2', [r'\bshp\b', r'\bnr0b2\b', r'small heterodimer partner', r'nuclear receptor subfamily 0 group b member 2',
                      r'nuclear receptor 0b2']),
    ('DAX1', 'NR0B1', [r'\bdax-?1\b', r'\bnr0b1\b', r'nuclear receptor subfamily 0 group b member 1']),
    ('PA2G4 (EBP1)', 'PA2G4', [r'\bpa2g4\b', r'proliferation-associated protein 2g4']),
]


def match_text(text):
    found = []
    for name, gene, pats in PATTERNS:
        if any(re.search(p, str(text), re.I) for p in pats):
            found.append((name, gene))
    return found


def identify(row):
    acc = str(row.rcsb_entity_uniprot) if pd.notna(row.rcsb_entity_uniprot) else ''
    hits = [UNIPROT[a] for a in acc.split(';') if a in UNIPROT]
    if hits:
        return hits[0][0], hits[0][1], 'UNIPROT', hits[0][2]
    d = match_text(row.rcsb_description)
    if len(d) >= 1:
        return d[0][0], d[0][1], 'DESCRIPTION', ''
    t = {x for x in match_text(row.rcsb_entry_title)}
    if len(t) == 1:
        name, gene = t.pop()
        return name, gene, 'TITLE', ''
    return 'unresolved', '', 'UNRESOLVED', ''


def old_family(identity):
    """Map the team's identity text to the same short names, to compare."""
    s = str(identity)
    if not s or s == 'nan' or 'unresolved' in s:
        return 'unresolved'
    m = match_text(s)
    return m[0][0] if m else s


def uniprot_sequences():
    """Full sequences of the cofactor proteins in UNIPROT, cached next to this script."""
    cache = HERE / 'cofactor_uniprot_sequences.json'
    if cache.exists():
        return json.loads(cache.read_text())
    seqs = {}
    for acc in UNIPROT:
        fasta = requests.get(f'https://rest.uniprot.org/uniprotkb/{acc}.fasta', timeout=60).text
        seqs[acc] = ''.join(fasta.split('\n')[1:])
    cache.write_text(json.dumps(seqs, indent=1))
    return seqs


def best_identity(peptide, protein):
    """Highest fraction of identical residues of the peptide in any gap-free window of the protein.
    X (modified residue) counts as a match. Engineered peptides (point substitutions, added
    terminal residues) therefore still score high."""
    p = np.frombuffer(str(peptide).encode(), dtype='S1')
    q = np.frombuffer(protein.encode(), dtype='S1')
    if len(p) == 0 or len(p) > len(q):
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(q, len(p))
    same = (windows == p) | (p == b'X')
    return float(same.sum(axis=1).max() / len(p))


def check_sequence(row, seqs):
    """Which cofactor proteins contain this peptide (best window identity >= MIN_IDENTITY)?
    Ties within 1e-9 are reported together (e.g. KILHRLLQ is in both SRC-1 and SRC-2)."""
    scores = {}
    for acc, s in seqs.items():
        name = UNIPROT[acc][0]
        scores[name] = max(scores.get(name, 0.0), best_identity(row.peptide_sequence, s))
    top = max(scores.values())
    best = sorted(n for n, v in scores.items() if v >= top - 1e-9) if top >= MIN_IDENTITY else []
    return pd.Series({'sequence_best_identity': round(top, 3),
                      'sequence_says': '; '.join(best) if best else 'no cofactor protein >= 75% identical',
                      'sequence_agrees': row.cofactor_v2 in best,
                      'sequence_ambiguous': len(best) > 1})


def main():
    df = pd.read_csv(SNAP / 'confirmed_cofactor_complexes_FIXED.csv').assign(source='original')
    if INCLUDE_ADDED:
        added = pd.read_csv(DATASET / 'added_structures/added_confirmed_cofactor_complexes_FIXED.csv').assign(
            source='added: missed by UniProt search')
        df = pd.concat([df, added], ignore_index=True)
    df[['cofactor_v2', 'cofactor_gene_v2', 'identity_source_v2', 'uniprot_species']] = df.apply(
        lambda r: pd.Series(identify(r)), axis=1)
    seqs = uniprot_sequences()
    df = pd.concat([df, df.apply(lambda r: check_sequence(r, seqs), axis=1)], axis=1)
    # Description names a different cofactor than the assigned one
    desc_names = df.rcsb_description.map(lambda t: {n for n, _ in match_text(t)})
    df['description_conflict'] = [bool(d) and v not in d for d, v in zip(desc_names, df.cofactor_v2)]
    # Needs a look: sequence contradicts the identity, or description contradicts it and the
    # sequence cannot settle it (ambiguous or no match)
    resolved = df.cofactor_v2 != 'unresolved'
    df['identity_review'] = resolved & (
        (~df.sequence_agrees & (df.sequence_best_identity >= MIN_IDENTITY))
        | (df.description_conflict & (df.sequence_ambiguous | ~df.sequence_agrees)))
    df['cofactor_team_short'] = df.cofactor_identity.map(old_family)
    df['changed_vs_team'] = df.cofactor_team_short != df.cofactor_v2
    cols = ['pdb_id', 'receptor', 'peptide_chain', 'peptide_sequence', 'has_LXXLL', 'has_CoRNR', 'cofactor_type',
            'rcsb_description', 'rcsb_entity_uniprot', 'cofactor_identity', 'cofactor_v2', 'cofactor_gene_v2',
            'identity_source_v2', 'uniprot_species', 'sequence_best_identity', 'sequence_says', 'sequence_agrees',
            'sequence_ambiguous', 'description_conflict', 'identity_review', 'changed_vs_team', 'source']
    df[cols].to_csv(HERE / 'cofactor_identity_v2.csv', index=False)
    res = df[df.cofactor_v2 != 'unresolved']
    print('resolved peptides:', len(res), '| sequence agrees:', res.sequence_agrees.sum(),
          '| no cofactor protein >= 75%:', (res.sequence_best_identity < MIN_IDENTITY).sum(),
          '| ambiguous by sequence:', res.sequence_ambiguous.sum())
    print('\nidentity_review:', df.identity_review.sum())
    print(df[df.identity_review][['pdb_id', 'peptide_sequence', 'rcsb_description', 'rcsb_entity_uniprot', 'cofactor_v2',
                                  'sequence_says', 'sequence_best_identity']].to_string(index=False))
    print('\nresolved but sequence below 75% to every cofactor protein:')
    print(res[res.sequence_best_identity < MIN_IDENTITY][['pdb_id', 'peptide_sequence', 'rcsb_description', 'cofactor_v2',
                                                          'sequence_best_identity']].to_string(index=False))

    pd.set_option('display.width', 250)
    print('peptides:', len(df))
    print('identity source:', df.identity_source_v2.value_counts().to_dict())
    print('resolved: team', (df.cofactor_team_short != 'unresolved').sum(), '-> v2', (df.cofactor_v2 != 'unresolved').sum())
    ch = df[df.changed_vs_team]
    print('\nchanged rows:', len(ch))
    print(ch.groupby(['cofactor_team_short', 'cofactor_v2']).size().sort_values(ascending=False).to_string())
    print('\nstill unresolved:')
    print(df[df.cofactor_v2 == 'unresolved'].rcsb_description.value_counts().to_string())


if __name__ == '__main__':
    main()
