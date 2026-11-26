"""Run the 5 human structures missed by the original search through the SAME screening as the
original dataset (BIOL363-NR-Cofactor-Analysis/scripts/03_filter.py).

The 5 entries are human receptors linked in the PDB to a foreign UniProt accession, so the
UniProt-based search never returned them (see 01_dataset/inventory_check/):
  5X8S, 5X8U, 5X8W, 5X8X  human RORγ LBD, linked to G1RH57 (gibbon)
  7ZTV                    human AR LBD (F755L), linked to P15207 (rat)

`chain_sequence` and `analyse_structure` and every threshold below are copied unchanged from
03_filter.py (that script runs on import, so it cannot be imported). Only the download folder
and the receptor labels differ. Flag columns are derived exactly as in 03_filter.py.

Added columns:
  excluded_at_stage  first screening stage the structure fails, or 'passed screening'
  reason             that stage's rule in words, with the structure's values

Output: added_structures_filter_results.csv (same 23 columns as structure_filter_results.csv + 2)
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

HERE = Path(__file__).resolve().parent
STRUCTURE_DIR = HERE.parent.parent / 'raw/structures_added'
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
ADDED = [('5X8S', 'RORC'), ('5X8U', 'RORC'), ('5X8W', 'RORC'), ('5X8X', 'RORC'), ('7ZTV', 'AR')]

# ---- copied unchanged from 03_filter.py -------------------------------------------------
TIMEOUT = 30
MAX_RETRIES = 3
MIN_RECEPTOR_LENGTH = 50
MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 49
LXXLL_PATTERN = re.compile("L[A-Z]{2}LL")
CORNR_PATTERN = re.compile("[IL][A-Z]{2}[IV]I")
EXCLUDED_HETERO = {"HOH", "WAT", "NA", "CL", "K", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI",
                   "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "FMT", "TRS", "MES", "HEP"}


def download_cif(pdb_id):
    pdb_id = str(pdb_id).upper()
    path = STRUCTURE_DIR / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            path.write_bytes(response.content)
            if path.stat().st_size < 1000:
                raise ValueError("Downloaded CIF is suspiciously small")
            return path
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Download failed: {last_error}")


def chain_sequence(chain):
    sequence = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        try:
            aa = seq1(residue.get_resname())
        except Exception:
            aa = "X"
        sequence.append(aa)
    return "".join(sequence)


def analyse_structure(pdb_id, receptor):
    pdb_id = str(pdb_id).upper()
    result = {
        "pdb_id": pdb_id, "receptor": receptor, "status": "FAILED", "receptor_chain_count": 0,
        "peptide_chain_count": 0, "receptor_chains": "", "peptide_chains": "", "peptide_sequences": "",
        "has_LXXLL": False, "has_CoRNR": False, "LXXLL_matches": "", "CoRNR_matches": "", "n_zinc": 0,
        "zn_finger_dbd_flag": False, "has_candidate_ligand": False, "candidate_ligands": "", "error": "",
    }
    try:
        cif_path = download_cif(pdb_id)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(cif_path))
        model = structure[0]
        receptor_chains, peptide_chains, peptide_sequences = [], [], []
        lxxll_matches, cornr_matches = [], []
        ligand_candidates = set()
        zinc_count = 0
        for chain in model:
            seq = chain_sequence(chain)
            length = len(seq)
            if length >= MIN_RECEPTOR_LENGTH:
                receptor_chains.append(f"{chain.id}:{length}")
            elif MIN_PEPTIDE_LENGTH <= length <= MAX_PEPTIDE_LENGTH:
                peptide_chains.append(f"{chain.id}:{length}")
                peptide_sequences.append(f"{chain.id}:{seq}")
                for match in LXXLL_PATTERN.finditer(seq):
                    lxxll_matches.append(f"{chain.id}:{match.group()}:{match.start() + 1}")
                for match in CORNR_PATTERN.finditer(seq):
                    cornr_matches.append(f"{chain.id}:{match.group()}:{match.start() + 1}")
        for residue in model.get_residues():
            if residue.id[0] == " ":
                continue
            resname = residue.get_resname().strip().upper()
            if resname == "ZN":
                zinc_count += 1
            if resname not in EXCLUDED_HETERO:
                heavy_atoms = [atom for atom in residue.get_atoms() if str(atom.element).upper() != "H"]
                if len(heavy_atoms) >= 5:
                    ligand_candidates.add(resname)
        result["receptor_chain_count"] = len(receptor_chains)
        result["peptide_chain_count"] = len(peptide_chains)
        result["receptor_chains"] = ";".join(receptor_chains)
        result["peptide_chains"] = ";".join(peptide_chains)
        result["peptide_sequences"] = ";".join(peptide_sequences)
        result["has_LXXLL"] = len(lxxll_matches) > 0
        result["has_CoRNR"] = len(cornr_matches) > 0
        result["LXXLL_matches"] = ";".join(lxxll_matches)
        result["CoRNR_matches"] = ";".join(cornr_matches)
        result["n_zinc"] = zinc_count
        result["zn_finger_dbd_flag"] = zinc_count >= 2
        result["candidate_ligands"] = ";".join(sorted(ligand_candidates))
        result["has_candidate_ligand"] = len(ligand_candidates) > 0
        result["status"] = "SUCCESS"
    except Exception as e:
        result["error"] = str(e)
    return result
# ---- end of copied code -------------------------------------------------------------------


def stage_and_reason(r):
    """First screening stage the structure fails, in the order of the original filter summary."""
    if r['status'] != 'SUCCESS':
        return '1 processing', f'mmCIF could not be downloaded or parsed: {r["error"]}'
    if not r['has_receptor']:
        return '2 receptor chain', f'no chain of >= {MIN_RECEPTOR_LENGTH} residues'
    if not r['passes_LBD_filter']:
        return '3 LBD / Zn-finger', f'{r["n_zinc"]} zinc atoms (>= 2 is treated as a DNA-binding domain)'
    if not r['has_peptide']:
        return ('4 short peptide',
                f'no separate peptide chain of {MIN_PEPTIDE_LENGTH}-{MAX_PEPTIDE_LENGTH} residues; '
                f'chains present: {r["receptor_chains"]} (receptor only)')
    if not r['has_cofactor_motif']:
        return '5 cofactor motif', f'peptide(s) {r["peptide_sequences"]} contain no LXXLL or CoRNR motif'
    return 'passed screening', 'receptor chain + peptide with LXXLL/CoRNR motif; goes on to contact and annotation checks'


def main():
    df = pd.DataFrame([analyse_structure(p, r) for p, r in ADDED])
    df['has_receptor'] = df['receptor_chain_count'].fillna(0) >= 1
    df['has_peptide'] = df['peptide_chain_count'].fillna(0) >= 1
    df['passes_LBD_filter'] = ~df['zn_finger_dbd_flag'].fillna(False)
    df['passes_basic_filter'] = ((df['status'] == 'SUCCESS') & df['has_receptor'] & df['has_peptide']
                                 & df['passes_LBD_filter'])
    df['has_cofactor_motif'] = df['has_LXXLL'] | df['has_CoRNR']
    df['strong_cofactor_candidate'] = df['passes_basic_filter'] & df['has_cofactor_motif']
    df[['excluded_at_stage', 'reason']] = df.apply(lambda r: pd.Series(stage_and_reason(r)), axis=1)
    df.to_csv(HERE / 'added_structures_filter_results.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 120)
    print(df[['pdb_id', 'receptor', 'receptor_chains', 'peptide_chains', 'n_zinc', 'candidate_ligands',
              'excluded_at_stage', 'reason']].to_string(index=False))


if __name__ == '__main__':
    main()
