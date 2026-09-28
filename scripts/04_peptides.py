from settings import DATA_ROOT, STRUCTURES, display

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm.auto import tqdm
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

DATA_DIR = DATA_ROOT
STRUCTURE_DIR = STRUCTURES
INPUT_FILE = DATA_DIR / "basic_filtered_structures.csv"
MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 49
MIN_RECEPTOR_LENGTH = 50
CONTACT_CUTOFF = 4.0
MAX_WORKERS = 8
LXXLL_PATTERN = re.compile("L[A-Z]{2}LL")
CORNR_PATTERN = re.compile("[IL][A-Z]{2}[IV]I")
COFACTOR_KEYWORDS = {
    "SRC-1/NCOA1": [
        "src-1",
        "src1",
        "ncoa1",
        "steroid receptor coactivator 1",
        "nuclear receptor coactivator 1",
    ],
    "SRC-2/NCOA2": [
        "src-2",
        "src2",
        "ncoa2",
        "tif2",
        "grip1",
        "steroid receptor coactivator 2",
        "nuclear receptor coactivator 2",
    ],
    "SRC-3/NCOA3": [
        "src-3",
        "src3",
        "ncoa3",
        "aib1",
        "actr",
        "steroid receptor coactivator 3",
        "nuclear receptor coactivator 3",
    ],
    "CBP/p300": ["crebbp", "cbp", "ep300", "p300"],
    "NCoR/NCOR1": ["ncor1", "ncor", "nuclear receptor corepressor 1"],
    "SMRT/NCOR2": ["ncor2", "smrt", "nuclear receptor corepressor 2"],
}
df = pd.read_csv(INPUT_FILE)
structures = (
    df[["pdb_id", "receptor"]].drop_duplicates().dropna().reset_index(drop=True)
)
print("Input:", INPUT_FILE)
print("PDB/receptor combinations:", len(structures))
print("Unique PDB structures:", structures["pdb_id"].nunique())


def get_sequence(chain):
    """
    Amino-acid sequence of a chain.
    """
    letters = []
    for residue in chain.get_residues():
        if not is_aa(residue, standard=False):
            continue
        try:
            letters.append(seq1(residue.get_resname()))
        except:
            letters.append("X")
    return "".join(letters)


def get_chain_description(cif_file):
    """
    Read molecule descriptions from the mmCIF file
    and connect them to chain IDs when possible.
    """
    try:
        cif = MMCIF2Dict(str(cif_file))

        def make_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            return [x]

        entity_ids = make_list(cif.get("_entity.id"))
        descriptions = make_list(cif.get("_entity.pdbx_description"))
        entity_description = {
            str(eid): str(desc) for (eid, desc) in zip(entity_ids, descriptions)
        }
        poly_entity_ids = make_list(cif.get("_entity_poly.entity_id"))
        strand_ids = make_list(cif.get("_entity_poly.pdbx_strand_id"))
        chain_descriptions = {}
        for entity_id, strands in zip(poly_entity_ids, strand_ids):
            description = entity_description.get(str(entity_id), "")
            for chain_id in str(strands).split(","):
                chain_id = chain_id.strip()
                if chain_id:
                    chain_descriptions[chain_id] = description
        return chain_descriptions
    except:
        return {}


def identify_annotation(description):
    """
    Check whether the PDB description contains a known
    cofactor name.
    """
    description_lower = str(description).lower()
    matches = []
    for cofactor, keywords in COFACTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in description_lower:
                matches.append(cofactor)
                break
    if "coactivator" in description_lower and (not matches):
        matches.append("annotated coactivator")
    if "corepressor" in description_lower and (not matches):
        matches.append("annotated corepressor")
    return sorted(set(matches))


def peptide_receptor_contacts(peptide_chain, receptor_chains, cutoff=4.0):
    """
    Count receptor amino-acid residues within cutoff
    distance of any peptide atom.
    """
    # Contacts are pooled across all long-chain candidates in the first model.
    receptor_atoms = []
    for chain in receptor_chains:
        for residue in chain.get_residues():
            if is_aa(residue, standard=False):
                receptor_atoms.extend(list(residue.get_atoms()))
    if not receptor_atoms:
        return (0, [])
    search = NeighborSearch(receptor_atoms)
    contacting_residues = set()
    for residue in peptide_chain.get_residues():
        if not is_aa(residue, standard=False):
            continue
        for atom in residue.get_atoms():
            nearby = search.search(atom.coord, cutoff, level="R")
            for receptor_residue in nearby:
                parent_chain = receptor_residue.get_parent().id
                resnum = receptor_residue.id[1]
                contacting_residues.add(
                    (parent_chain, resnum, receptor_residue.get_resname())
                )
    contacts = sorted(contacting_residues)
    formatted = [f"{chain}:{resname}{resnum}" for (chain, resnum, resname) in contacts]
    return (len(contacts), formatted)


def analyse_pdb(pdb_id, receptor_name):
    pdb_id = str(pdb_id).upper()
    cif_file = STRUCTURE_DIR / f"{pdb_id}.cif"
    output_rows = []
    try:
        if not cif_file.exists():
            raise FileNotFoundError(f"{cif_file} not found")
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(cif_file))
        model = structure[0]
        descriptions = get_chain_description(cif_file)
        receptor_chains = []
        for chain in model:
            seq = get_sequence(chain)
            if len(seq) >= MIN_RECEPTOR_LENGTH:
                receptor_chains.append(chain)
        for chain in model:
            seq = get_sequence(chain)
            length = len(seq)
            if not MIN_PEPTIDE_LENGTH <= length <= MAX_PEPTIDE_LENGTH:
                continue
            lxxll_matches = [
                f"{m.group()}@{m.start() + 1}" for m in LXXLL_PATTERN.finditer(seq)
            ]
            cornr_matches = [
                f"{m.group()}@{m.start() + 1}" for m in CORNR_PATTERN.finditer(seq)
            ]
            has_lxxll = len(lxxll_matches) > 0
            has_cornr = len(cornr_matches) > 0
            motif_evidence = has_lxxll or has_cornr
            description = descriptions.get(chain.id, "")
            annotation_matches = identify_annotation(description)
            annotation_evidence = len(annotation_matches) > 0
            (n_contacts, contacts) = peptide_receptor_contacts(
                chain, receptor_chains, cutoff=CONTACT_CUTOFF
            )
            contact_evidence = n_contacts > 0
            if contact_evidence and (motif_evidence or annotation_evidence):
                confidence = "HIGH_CONFIDENCE"
            elif motif_evidence or annotation_evidence:
                confidence = "REVIEW"
            elif contact_evidence:
                confidence = "CONTACT_ONLY"
            else:
                confidence = "UNLIKELY"
            types = []
            if has_lxxll:
                types.append("COACTIVATOR")
            if has_cornr:
                types.append("COREPRESSOR")
            annotation_text = " ".join(annotation_matches).lower()
            if any(
                (
                    x in annotation_text
                    for x in ["src", "ncoa", "cbp", "p300", "coactivator"]
                )
            ):
                types.append("COACTIVATOR")
            if any((x in annotation_text for x in ["ncor", "smrt", "corepressor"])):
                types.append("COREPRESSOR")
            types = sorted(set(types))
            cofactor_type = ";".join(types) if types else "UNKNOWN"
            output_rows.append(
                {
                    "pdb_id": pdb_id,
                    "receptor": receptor_name,
                    "peptide_chain": chain.id,
                    "peptide_length": length,
                    "peptide_sequence": seq,
                    "pdb_description": description,
                    "has_LXXLL": has_lxxll,
                    "LXXLL_matches": ";".join(lxxll_matches),
                    "has_CoRNR": has_cornr,
                    "CoRNR_matches": ";".join(cornr_matches),
                    "annotation_matches": ";".join(annotation_matches),
                    "has_annotation_evidence": annotation_evidence,
                    "receptor_contact": contact_evidence,
                    "contact_residue_count": n_contacts,
                    "contact_residues": ";".join(contacts),
                    "cofactor_type": cofactor_type,
                    "cofactor_confidence": confidence,
                    "error": "",
                }
            )
        return output_rows
    except Exception as e:
        return [
            {
                "pdb_id": pdb_id,
                "receptor": receptor_name,
                "peptide_chain": None,
                "cofactor_confidence": "ERROR",
                "error": str(e),
            }
        ]


all_rows = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(analyse_pdb, row.pdb_id, row.receptor): row.pdb_id
        for row in structures.itertuples(index=False)
    }
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Validating cofactor peptides"
    ):
        try:
            rows = future.result()
            all_rows.extend(rows)
        except Exception as e:
            all_rows.append(
                {
                    "pdb_id": futures[future],
                    "cofactor_confidence": "ERROR",
                    "error": str(e),
                }
            )
validation_df = pd.DataFrame(all_rows)
validation_df.to_csv(DATA_DIR / "peptide_cofactor_validation.csv", index=False)
confirmed_df = validation_df[
    validation_df["cofactor_confidence"] == "HIGH_CONFIDENCE"
].copy()
confirmed_df.to_csv(DATA_DIR / "confirmed_cofactor_complexes.csv", index=False)
review_df = validation_df[
    validation_df["cofactor_confidence"].isin(["REVIEW", "CONTACT_ONLY"])
].copy()
review_df.to_csv(DATA_DIR / "cofactor_review_needed.csv", index=False)
errors_df = validation_df[validation_df["cofactor_confidence"] == "ERROR"].copy()
errors_df.to_csv(DATA_DIR / "cofactor_validation_errors.csv", index=False)
summary = (
    validation_df["cofactor_confidence"]
    .value_counts()
    .rename_axis("classification")
    .reset_index(name="n_peptide_chains")
)
display(summary)
print(
    "\nUnique PDB structures with HIGH-CONFIDENCE cofactors:",
    confirmed_df["pdb_id"].nunique(),
)
print("High-confidence peptide chains:", len(confirmed_df))
print("\nFiles created:")
print("1. peptide_cofactor_validation.csv")
print("2. confirmed_cofactor_complexes.csv")
print("3. cofactor_review_needed.csv")
print("4. cofactor_validation_errors.csv")
