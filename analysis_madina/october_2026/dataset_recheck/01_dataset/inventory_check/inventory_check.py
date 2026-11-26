"""Step 0 of the rebuilt pipeline: is the structure inventory complete?

The original search (BIOL363-NR-Cofactor-Analysis/scripts/01_inventory.py) found PDB entries whose
polymer entity is mapped to the receptor's human UniProt accession. Two checks:
  1. rerun that search today and compare with the September snapshot (new / withdrawn entries);
  2. search RCSB by SEQUENCE (>= 90% identity to the full human UniProt sequence) to find entries
     that contain the receptor but are not mapped to its UniProt accession; then look up the
     organism of every hit not in the snapshot (human = missed, other species = out of scope).

Output: uniprot_search_vs_snapshot.csv, found_by_sequence_not_in_inventory.csv
Needs internet (RCSB Search and Data APIs).
"""
import json
import re
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAPSHOT = ROOT.parent / 'BIOL363-NR-Cofactor-Analysis/data/snapshot/all_nr_structures.csv'
INVENTORY_SCRIPT = ROOT.parent / 'BIOL363-NR-Cofactor-Analysis/scripts/01_inventory.py'
SEQUENCES = ROOT / '01_dataset/numbering/uniprot_sequences_all48.json'
SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
GRAPHQL = 'https://data.rcsb.org/graphql'
MIN_RECEPTOR_LEN = 150   # entity long enough to be a receptor domain, not a peptide or DBD fragment


def search(query, return_type):
    body = {'query': query, 'return_type': return_type, 'request_options': {'return_all_hits': True}}
    r = requests.post(SEARCH, json=body, timeout=180)
    return [] if r.status_code == 204 else [x['identifier'] for x in r.json().get('result_set', [])]


def by_uniprot(acc):
    return set(search({'type': 'terminal', 'service': 'text', 'parameters': {
        'attribute': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession',
        'operator': 'exact_match', 'value': acc}}, 'entry'))


def by_sequence(seq):
    hits = search({'type': 'terminal', 'service': 'sequence', 'parameters': {
        'evalue_cutoff': 1e-10, 'identity_cutoff': 0.9, 'sequence_type': 'protein', 'value': seq}}, 'polymer_entity')
    return {h.split('_')[0] for h in hits}


def entity_details(pdb_ids):
    q = ('query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id struct{title} exptl{method} '
         'polymer_entities{rcsb_polymer_entity{pdbx_description} entity_poly{rcsb_sample_sequence_length} '
         'rcsb_entity_source_organism{scientific_name} rcsb_polymer_entity_container_identifiers'
         '{reference_sequence_identifiers{database_name database_accession}}}}}')
    rows = []
    for i in range(0, len(pdb_ids), 50):
        data = requests.post(GRAPHQL, json={'query': q, 'variables': {'ids': pdb_ids[i:i + 50]}}, timeout=180).json()
        for e in data['data']['entries']:
            method = ((e.get('exptl') or [{}])[0] or {}).get('method', '?')
            for pe in e.get('polymer_entities') or []:
                length = (pe.get('entity_poly') or {}).get('rcsb_sample_sequence_length') or 0
                if length < MIN_RECEPTOR_LEN:
                    continue
                orgs = {str(o.get('scientific_name')) for o in (pe.get('rcsb_entity_source_organism') or [])}
                refs = (pe.get('rcsb_polymer_entity_container_identifiers') or {}).get('reference_sequence_identifiers') or []
                accs = {x['database_accession'] for x in refs if x.get('database_name') == 'UniProt'}
                rows.append({'pdb_id': e['rcsb_id'], 'organism': ';'.join(sorted(orgs)) or 'unknown',
                             'uniprot': ';'.join(sorted(accs)) or 'NONE', 'method': method, 'length': length,
                             'entity': (pe.get('rcsb_polymer_entity') or {}).get('pdbx_description') or '',
                             'title': (e.get('struct') or {}).get('title') or ''})
    return pd.DataFrame(rows)


def main():
    ids = dict(re.findall(r'"(\w+)": "([A-Z0-9]{6})"', INVENTORY_SCRIPT.read_text(encoding='utf-8')))
    seqs = json.loads(SEQUENCES.read_text())
    snap = pd.read_csv(SNAPSHOT)
    in_snapshot = set(snap.pdb_id)

    rows, found = [], {}
    for rec, acc in ids.items():
        live = by_uniprot(acc)
        old = set(snap[snap.uniprot_id == acc].pdb_id)
        seq_hits = by_sequence(seqs[acc]) if acc in seqs else set()
        for p in seq_hits - in_snapshot:
            found.setdefault(p, []).append(rec)
        rows.append({'receptor': rec, 'uniprot': acc, 'snapshot': len(old), 'uniprot_search_now': len(live),
                     'new_since_snapshot': len(live - old), 'withdrawn_since_snapshot': len(old - live),
                     'sequence_hits_90pct': len(seq_hits)})
    pd.DataFrame(rows).to_csv(HERE / 'uniprot_search_vs_snapshot.csv', index=False)

    det = entity_details(sorted(found))
    det['receptor_by_sequence'] = det.pdb_id.map(lambda p: ';'.join(found[p]))
    det['category'] = det.organism.str.contains('Homo sapiens').map(
        {True: 'human receptor with wrong/foreign UniProt mapping: MISSED',
         False: 'non-human receptor (out of scope: human only)'})
    det = det.drop_duplicates(['pdb_id', 'organism', 'uniprot']).sort_values(['category', 'receptor_by_sequence', 'pdb_id'])
    det.to_csv(HERE / 'found_by_sequence_not_in_inventory.csv', index=False)
    print(pd.DataFrame(rows)[['snapshot', 'uniprot_search_now', 'new_since_snapshot', 'withdrawn_since_snapshot']].sum().to_dict())
    print(det.groupby('category').pdb_id.nunique().to_string())


if __name__ == '__main__':
    main()
