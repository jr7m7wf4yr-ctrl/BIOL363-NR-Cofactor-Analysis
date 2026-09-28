from settings import DATA_ROOT, STRUCTURES, display

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm.auto import tqdm
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

DATA_DIR = DATA_ROOT
INPUT_FILE = DATA_DIR / "all_nr_structures.csv"
STRUCTURE_DIR = STRUCTURES
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = 8
TIMEOUT = 30
MAX_RETRIES = 3
MIN_RECEPTOR_LENGTH = 50
MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 49
LXXLL_PATTERN = re.compile("L[A-Z]{2}LL")
CORNR_PATTERN = re.compile("[IL][A-Z]{2}[IV]I")
EXCLUDED_HETERO = {
    "HOH",
    "WAT",
    "NA",
    "CL",
    "K",
    "CA",
    "MG",
    "ZN",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
    "PEG",
    "ACT",
    "FMT",
    "TRS",
    "MES",
    "HEP",
}
df = pd.read_csv(INPUT_FILE)
structures = (
    df[["pdb_id", "receptor"]].drop_duplicates().dropna().reset_index(drop=True)
)
print("Input file:", INPUT_FILE)
print("Receptor-PDB combinations:", len(structures))
print("Unique PDB structures:", structures["pdb_id"].nunique())
print("Unique receptors:", structures["receptor"].nunique())


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
        "pdb_id": pdb_id,
        "receptor": receptor,
        "status": "FAILED",
        "receptor_chain_count": 0,
        "peptide_chain_count": 0,
        "receptor_chains": "",
        "peptide_chains": "",
        "peptide_sequences": "",
        "has_LXXLL": False,
        "has_CoRNR": False,
        "LXXLL_matches": "",
        "CoRNR_matches": "",
        "n_zinc": 0,
        "zn_finger_dbd_flag": False,
        "has_candidate_ligand": False,
        "candidate_ligands": "",
        "error": "",
    }
    try:
        cif_path = download_cif(pdb_id)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(cif_path))
        model = structure[0]
        receptor_chains = []
        peptide_chains = []
        peptide_sequences = []
        lxxll_matches = []
        cornr_matches = []
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
                    lxxll_matches.append(
                        f"{chain.id}:{match.group()}:{match.start() + 1}"
                    )
                for match in CORNR_PATTERN.finditer(seq):
                    cornr_matches.append(
                        f"{chain.id}:{match.group()}:{match.start() + 1}"
                    )
        for residue in model.get_residues():
            if residue.id[0] == " ":
                continue
            resname = residue.get_resname().strip().upper()
            if resname == "ZN":
                zinc_count += 1
            if resname not in EXCLUDED_HETERO:
                heavy_atoms = [
                    atom
                    for atom in residue.get_atoms()
                    if str(atom.element).upper() != "H"
                ]
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
        # Zinc count is a screening heuristic, not a domain annotation.
        result["zn_finger_dbd_flag"] = zinc_count >= 2
        result["candidate_ligands"] = ";".join(sorted(ligand_candidates))
        result["has_candidate_ligand"] = len(ligand_candidates) > 0
        result["status"] = "SUCCESS"
    except Exception as e:
        result["error"] = str(e)
    return result


results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(analyse_structure, row.pdb_id, row.receptor): row.pdb_id
        for row in structures.itertuples(index=False)
    }
    for future in tqdm(
        as_completed(future_map), total=len(future_map), desc="Filtering structures"
    ):
        try:
            results.append(future.result())
        except Exception as e:
            results.append(
                {
                    "pdb_id": future_map[future],
                    "receptor": None,
                    "status": "FAILED",
                    "error": str(e),
                }
            )
filter_df = pd.DataFrame(results)
filter_df["has_receptor"] = filter_df["receptor_chain_count"].fillna(0) >= 1
filter_df["has_peptide"] = filter_df["peptide_chain_count"].fillna(0) >= 1
filter_df["passes_LBD_filter"] = ~filter_df["zn_finger_dbd_flag"].fillna(False)
filter_df["passes_basic_filter"] = (
    (filter_df["status"] == "SUCCESS")
    & filter_df["has_receptor"]
    & filter_df["has_peptide"]
    & filter_df["passes_LBD_filter"]
)
filter_df["has_cofactor_motif"] = filter_df["has_LXXLL"] | filter_df["has_CoRNR"]
filter_df["strong_cofactor_candidate"] = (
    filter_df["passes_basic_filter"] & filter_df["has_cofactor_motif"]
)
filter_df.to_csv(DATA_DIR / "structure_filter_results.csv", index=False)
basic_df = filter_df[filter_df["passes_basic_filter"]].copy()
basic_df.to_csv(DATA_DIR / "basic_filtered_structures.csv", index=False)
cofactor_df = filter_df[filter_df["strong_cofactor_candidate"]].copy()
cofactor_df.to_csv(DATA_DIR / "cofactor_motif_candidates.csv", index=False)
errors_df = filter_df[filter_df["status"] != "SUCCESS"].copy()
errors_df.to_csv(DATA_DIR / "filter_errors.csv", index=False)
summary = pd.DataFrame(
    {
        "stage": [
            "Starting PDB structures",
            "Successfully processed",
            "Long receptor chain present",
            "Passed LBD / Zn-finger filter",
            "Short peptide present",
            "Basic filtered structures",
            "LXXLL / CoRNR motif found",
            "Strong cofactor candidates",
            "Candidate ligand present",
        ],
        "count": [
            filter_df["pdb_id"].nunique(),
            filter_df.loc[filter_df["status"] == "SUCCESS", "pdb_id"].nunique(),
            filter_df.loc[filter_df["has_receptor"], "pdb_id"].nunique(),
            filter_df.loc[filter_df["passes_LBD_filter"], "pdb_id"].nunique(),
            filter_df.loc[filter_df["has_peptide"], "pdb_id"].nunique(),
            filter_df.loc[filter_df["passes_basic_filter"], "pdb_id"].nunique(),
            filter_df.loc[filter_df["has_cofactor_motif"], "pdb_id"].nunique(),
            filter_df.loc[filter_df["strong_cofactor_candidate"], "pdb_id"].nunique(),
            filter_df.loc[filter_df["has_candidate_ligand"], "pdb_id"].nunique(),
        ],
    }
)
summary.to_csv(DATA_DIR / "filter_summary.csv", index=False)
display(summary)
vdr = filter_df[filter_df["receptor"] == "VDR"]
print("\n--- VDR VALIDATION ---")
print("VDR structures before filtering:", vdr["pdb_id"].nunique())
print(
    "VDR flagged as Zn-finger/DBD:",
    vdr.loc[vdr["zn_finger_dbd_flag"], "pdb_id"].nunique(),
)
print("VDR passing LBD filter:", vdr.loc[vdr["passes_LBD_filter"], "pdb_id"].nunique())
print("\nFiles created:")
print("1. structure_filter_results.csv")
print("2. basic_filtered_structures.csv")
print("3. cofactor_motif_candidates.csv")
print("4. filter_summary.csv")
print("5. filter_errors.csv")
