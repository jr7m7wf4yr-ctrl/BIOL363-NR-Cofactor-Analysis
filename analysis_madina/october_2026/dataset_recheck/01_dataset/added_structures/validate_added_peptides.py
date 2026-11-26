"""Peptide validation for the added structures that passed screening (5X8U, 5X8W, 5X8X), with
the SAME code as the original dataset:
  BIOL363-NR-Cofactor-Analysis/scripts/04_peptides.py    motif + receptor contact (<= 4 A)
  BIOL363-NR-Cofactor-Analysis/scripts/05_annotations.py RCSB entity annotation, cofactor identity,
                                                         final confidence
Functions and constants are copied unchanged (both scripts run on import). The input is the
passing rows of added_structures_filter_results.csv, as 04 takes basic_filtered_structures.csv.

A peptide is kept (as in confirmed_cofactor_complexes_FIXED.csv) when final_cofactor_confidence is
HIGH_CONFIDENCE_IDENTIFIED or HIGH_CONFIDENCE_IDENTITY_UNRESOLVED, i.e. motif AND receptor contact.
Added columns: kept, reason.

Output: added_peptide_cofactor_validation_FIXED.csv (all peptides of the 3 structures),
        added_confirmed_cofactor_complexes_FIXED.csv (kept peptides only)
"""
import re
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

HERE = Path(__file__).resolve().parent
STRUCTURE_DIR = HERE.parent.parent / 'raw/structures_added'

# ---- copied unchanged from 04_peptides.py -------------------------------------------------
MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 49
MIN_RECEPTOR_LENGTH = 50
CONTACT_CUTOFF = 4.0
LXXLL_PATTERN = re.compile("L[A-Z]{2}LL")
CORNR_PATTERN = re.compile("[IL][A-Z]{2}[IV]I")
COFACTOR_KEYWORDS = {
    "SRC-1/NCOA1": ["src-1", "src1", "ncoa1", "steroid receptor coactivator 1", "nuclear receptor coactivator 1"],
    "SRC-2/NCOA2": ["src-2", "src2", "ncoa2", "tif2", "grip1", "steroid receptor coactivator 2",
                    "nuclear receptor coactivator 2"],
    "SRC-3/NCOA3": ["src-3", "src3", "ncoa3", "aib1", "actr", "steroid receptor coactivator 3",
                    "nuclear receptor coactivator 3"],
    "CBP/p300": ["crebbp", "cbp", "ep300", "p300"],
    "NCoR/NCOR1": ["ncor1", "ncor", "nuclear receptor corepressor 1"],
    "SMRT/NCOR2": ["ncor2", "smrt", "nuclear receptor corepressor 2"],
}


def get_sequence(chain):
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
        entity_description = {str(eid): str(desc) for (eid, desc) in zip(entity_ids, descriptions)}
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
                contacting_residues.add((parent_chain, resnum, receptor_residue.get_resname()))
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
            lxxll_matches = [f"{m.group()}@{m.start() + 1}" for m in LXXLL_PATTERN.finditer(seq)]
            cornr_matches = [f"{m.group()}@{m.start() + 1}" for m in CORNR_PATTERN.finditer(seq)]
            has_lxxll = len(lxxll_matches) > 0
            has_cornr = len(cornr_matches) > 0
            motif_evidence = has_lxxll or has_cornr
            description = descriptions.get(chain.id, "")
            annotation_matches = identify_annotation(description)
            annotation_evidence = len(annotation_matches) > 0
            (n_contacts, contacts) = peptide_receptor_contacts(chain, receptor_chains, cutoff=CONTACT_CUTOFF)
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
            if any((x in annotation_text for x in ["src", "ncoa", "cbp", "p300", "coactivator"])):
                types.append("COACTIVATOR")
            if any((x in annotation_text for x in ["ncor", "smrt", "corepressor"])):
                types.append("COREPRESSOR")
            types = sorted(set(types))
            cofactor_type = ";".join(types) if types else "UNKNOWN"
            output_rows.append({
                "pdb_id": pdb_id, "receptor": receptor_name, "peptide_chain": chain.id,
                "peptide_length": length, "peptide_sequence": seq, "pdb_description": description,
                "has_LXXLL": has_lxxll, "LXXLL_matches": ";".join(lxxll_matches),
                "has_CoRNR": has_cornr, "CoRNR_matches": ";".join(cornr_matches),
                "annotation_matches": ";".join(annotation_matches), "has_annotation_evidence": annotation_evidence,
                "receptor_contact": contact_evidence, "contact_residue_count": n_contacts,
                "contact_residues": ";".join(contacts), "cofactor_type": cofactor_type,
                "cofactor_confidence": confidence, "error": "",
            })
        return output_rows
    except Exception as e:
        return [{"pdb_id": pdb_id, "receptor": receptor_name, "peptide_chain": None,
                 "cofactor_confidence": "ERROR", "error": str(e)}]
# ---- end of 04_peptides.py ----------------------------------------------------------------

# ---- copied unchanged from 05_annotations.py ----------------------------------------------
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
QUERY = """
query GetEntries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    polymer_entities {
      rcsb_id
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers {
        asym_ids
        auth_asym_ids
        reference_sequence_identifiers { database_name database_accession }
      }
    }
  }
}
"""
COFACTOR_PATTERNS = {
    "SRC-1 / NCOA1": ["\\bncoa1\\b", "\\bsrc[- ]?1\\b", "steroid receptor coactivator 1", "nuclear receptor coactivator 1"],
    "SRC-2 / NCOA2 / TIF2 / GRIP1": ["\\bncoa2\\b", "\\bsrc[- ]?2\\b", "\\btif2\\b", "\\bgrip1\\b",
                                     "steroid receptor coactivator 2", "nuclear receptor coactivator 2",
                                     "transcriptional intermediary factor 2"],
    "SRC-3 / NCOA3 / AIB1 / ACTR": ["\\bncoa3\\b", "\\bsrc[- ]?3\\b", "\\baib1\\b", "\\bactr\\b",
                                    "steroid receptor coactivator 3", "nuclear receptor coactivator 3"],
    "CBP / CREBBP": ["\\bcrebbp\\b", "\\bcbp\\b", "creb[- ]binding protein"],
    "p300 / EP300": ["\\bep300\\b", "\\bp300\\b"],
    "NCoR / NCOR1": ["\\bncor1\\b", "\\bncor\\b", "nuclear receptor corepressor 1"],
    "SMRT / NCOR2": ["\\bncor2\\b", "\\bsmrt\\b", "silencing mediator"],
}


def annotation_table(pdb_ids):
    result = requests.post(RCSB_GRAPHQL, json={"query": QUERY, "variables": {"ids": pdb_ids}}, timeout=60).json()
    rows = []
    for entry in result["data"]["entries"]:
        title = (entry.get("struct") or {}).get("title", "")
        for entity in entry.get("polymer_entities", []) or []:
            identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            uniprot_ids = [ref.get("database_accession") for ref in identifiers.get("reference_sequence_identifiers") or []
                           if str(ref.get("database_name", "")).upper() in {"UNP", "UNIPROT"} and ref.get("database_accession")]
            rows.append({"pdb_id": entry["rcsb_id"], "entity_id": entity.get("rcsb_id"),
                         "entity_description": (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "",
                         "entry_title": title,
                         "auth_chain_ids": ";".join(map(str, identifiers.get("auth_asym_ids") or [])),
                         "label_chain_ids": ";".join(map(str, identifiers.get("asym_ids") or [])),
                         "entity_uniprot_ids": ";".join(sorted(set(uniprot_ids)))})
    return pd.DataFrame(rows)


def find_chain_annotation(annotation_df, pdb_id, peptide_chain):
    subset = annotation_df[annotation_df["pdb_id"] == pdb_id]
    peptide_chain = str(peptide_chain).strip()
    for method, column in [("AUTH_CHAIN", "auth_chain_ids"), ("LABEL_CHAIN", "label_chain_ids")]:
        for _, entity in subset.iterrows():
            if peptide_chain in str(entity[column]).split(";"):
                return pd.Series({"rcsb_entity_id": entity["entity_id"], "rcsb_description": entity["entity_description"],
                                  "rcsb_entry_title": entity["entry_title"],
                                  "rcsb_entity_uniprot": entity["entity_uniprot_ids"], "chain_match_method": method})
    return pd.Series({"rcsb_entity_id": None, "rcsb_description": None, "rcsb_entry_title": None,
                      "rcsb_entity_uniprot": None, "chain_match_method": "NO_MATCH"})


def classify_annotation(row):
    text = (str(row.get("rcsb_description", "")) + " " + str(row.get("rcsb_entry_title", ""))).lower()
    identities = [name for name, patterns in COFACTOR_PATTERNS.items()
                  if any(re.search(pattern, text, re.I) for pattern in patterns)]
    if not identities:
        if "coactivator" in text:
            identities.append("COACTIVATOR — exact identity unresolved")
        elif "corepressor" in text or "co-repressor" in text:
            identities.append("COREPRESSOR — exact identity unresolved")
    return "; ".join(sorted(set(identities)))


def final_confidence(row):
    motif = bool(row.get("has_LXXLL", False)) or bool(row.get("has_CoRNR", False))
    contact = bool(row.get("receptor_contact", False))
    annotation = bool(row.get("has_annotation_evidence_fixed", False))
    if motif and contact and annotation:
        return "HIGH_CONFIDENCE_IDENTIFIED"
    if motif and contact:
        return "HIGH_CONFIDENCE_IDENTITY_UNRESOLVED"
    if annotation and contact:
        return "ANNOTATED_CONTACT"
    if motif or annotation:
        return "REVIEW"
    if contact:
        return "CONTACT_ONLY"
    return "UNLIKELY"
# ---- end of 05_annotations.py --------------------------------------------------------------

KEPT = {"HIGH_CONFIDENCE_IDENTIFIED", "HIGH_CONFIDENCE_IDENTITY_UNRESOLVED"}
REASONS = {
    "HIGH_CONFIDENCE_IDENTIFIED": "kept: cofactor motif + receptor contact + known cofactor identity",
    "HIGH_CONFIDENCE_IDENTITY_UNRESOLVED": "kept: cofactor motif + receptor contact; cofactor name not in annotation",
    "ANNOTATED_CONTACT": "excluded at peptide validation: annotated cofactor contacting the receptor, but no LXXLL/CoRNR motif",
    "REVIEW": "excluded at peptide validation: motif or annotation but no receptor contact within 4 A",
    "CONTACT_ONLY": "excluded at peptide validation: contacts the receptor but has no motif and no cofactor annotation",
    "UNLIKELY": "excluded at peptide validation: no motif, no annotation, no receptor contact",
}


def main():
    screened = pd.read_csv(HERE / 'added_structures_filter_results.csv')
    passed = screened[screened.passes_basic_filter][['pdb_id', 'receptor']].drop_duplicates()
    rows = [r for p in passed.itertuples(index=False) for r in analyse_pdb(p.pdb_id, p.receptor)]
    df = pd.DataFrame(rows)
    ann = annotation_table(sorted(df.pdb_id.unique()))
    df = pd.concat([df.reset_index(drop=True),
                    df.apply(lambda r: find_chain_annotation(ann, r["pdb_id"], r["peptide_chain"]), axis=1)], axis=1)
    df["cofactor_identity"] = df.apply(classify_annotation, axis=1)
    df["has_annotation_evidence_fixed"] = df["cofactor_identity"].fillna("").str.len() > 0
    df["structural_cofactor_candidate"] = df["receptor_contact"].fillna(False) & (
        df["has_LXXLL"].fillna(False) | df["has_CoRNR"].fillna(False))
    df["cofactor_identity_status"] = "UNRESOLVED"
    df.loc[df["has_annotation_evidence_fixed"], "cofactor_identity_status"] = "IDENTIFIED"
    df["final_cofactor_confidence"] = df.apply(final_confidence, axis=1)
    df["kept"] = df["final_cofactor_confidence"].isin(KEPT)
    df["reason"] = df["final_cofactor_confidence"].map(REASONS)
    df.to_csv(HERE / 'added_peptide_cofactor_validation_FIXED.csv', index=False)
    df[df.kept].drop(columns=['kept', 'reason']).to_csv(HERE / 'added_confirmed_cofactor_complexes_FIXED.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 60)
    print(df[['pdb_id', 'peptide_chain', 'peptide_sequence', 'LXXLL_matches', 'CoRNR_matches', 'contact_residue_count',
              'rcsb_description', 'cofactor_identity', 'final_cofactor_confidence', 'kept']].to_string(index=False))


if __name__ == '__main__':
    main()
