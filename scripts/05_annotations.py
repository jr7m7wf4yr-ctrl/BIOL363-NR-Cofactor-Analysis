from settings import DATA_ROOT, display

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm.auto import tqdm

DATA_DIR = DATA_ROOT
INPUT_FILE = DATA_DIR / "peptide_cofactor_validation.csv"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 40
MAX_WORKERS = 5
MAX_RETRIES = 3
TIMEOUT = 60
df = pd.read_csv(INPUT_FILE)
df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
df["peptide_chain"] = df["peptide_chain"].astype(str)
pdb_ids = sorted(df["pdb_id"].dropna().unique())
print("Loaded:", INPUT_FILE)
print("Rows:", len(df))
print("Unique PDB structures:", len(pdb_ids))
QUERY = """
query GetEntries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id

    struct {
      title
    }

    polymer_entities {

      rcsb_id

      rcsb_polymer_entity {
        pdbx_description
      }

      rcsb_polymer_entity_container_identifiers {
        asym_ids
        auth_asym_ids

        reference_sequence_identifiers {
          database_name
          database_accession
        }
      }
    }
  }
}
"""


def fetch_batch(batch):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                RCSB_GRAPHQL,
                json={"query": QUERY, "variables": {"ids": batch}},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("errors"):
                raise RuntimeError(str(result["errors"]))
            return result["data"]["entries"]
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)
    raise RuntimeError(last_error)


batches = [pdb_ids[i : i + BATCH_SIZE] for i in range(0, len(pdb_ids), BATCH_SIZE)]
print("RCSB batches:", len(batches))
entries = []
api_errors = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {executor.submit(fetch_batch, batch): batch for batch in batches}
    for future in tqdm(
        as_completed(future_map), total=len(future_map), desc="Getting RCSB annotations"
    ):
        batch = future_map[future]
        try:
            result = future.result()
            entries.extend([x for x in result if x is not None])
        except Exception as e:
            for pdb_id in batch:
                api_errors.append({"pdb_id": pdb_id, "error": str(e)})
print("Entries retrieved:", len(entries))
print("API errors:", len(api_errors))
annotation_rows = []
for entry in entries:
    pdb_id = entry["rcsb_id"]
    title = (entry.get("struct") or {}).get("title", "")
    for entity in entry.get("polymer_entities", []) or []:
        entity_info = entity.get("rcsb_polymer_entity") or {}
        description = entity_info.get("pdbx_description") or ""
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        auth_chains = identifiers.get("auth_asym_ids") or []
        label_chains = identifiers.get("asym_ids") or []
        reference_ids = identifiers.get("reference_sequence_identifiers") or []
        uniprot_ids = []
        for ref in reference_ids:
            if str(ref.get("database_name", "")).upper() in {"UNP", "UNIPROT"}:
                accession = ref.get("database_accession")
                if accession:
                    uniprot_ids.append(accession)
        annotation_rows.append(
            {
                "pdb_id": pdb_id,
                "entity_id": entity.get("rcsb_id"),
                "entity_description": description,
                "entry_title": title,
                "auth_chain_ids": ";".join(map(str, auth_chains)),
                "label_chain_ids": ";".join(map(str, label_chains)),
                "entity_uniprot_ids": ";".join(sorted(set(uniprot_ids))),
            }
        )
annotation_df = pd.DataFrame(annotation_rows)
print("Polymer entities retrieved:", len(annotation_df))


def find_chain_annotation(pdb_id, peptide_chain):
    subset = annotation_df[annotation_df["pdb_id"] == pdb_id]
    peptide_chain = str(peptide_chain).strip()
    for _, entity in subset.iterrows():
        chains = str(entity["auth_chain_ids"]).split(";")
        if peptide_chain in chains:
            return pd.Series(
                {
                    "rcsb_entity_id": entity["entity_id"],
                    "rcsb_description": entity["entity_description"],
                    "rcsb_entry_title": entity["entry_title"],
                    "rcsb_entity_uniprot": entity["entity_uniprot_ids"],
                    "chain_match_method": "AUTH_CHAIN",
                }
            )
    for _, entity in subset.iterrows():
        chains = str(entity["label_chain_ids"]).split(";")
        if peptide_chain in chains:
            return pd.Series(
                {
                    "rcsb_entity_id": entity["entity_id"],
                    "rcsb_description": entity["entity_description"],
                    "rcsb_entry_title": entity["entry_title"],
                    "rcsb_entity_uniprot": entity["entity_uniprot_ids"],
                    "chain_match_method": "LABEL_CHAIN",
                }
            )
    return pd.Series(
        {
            "rcsb_entity_id": None,
            "rcsb_description": None,
            "rcsb_entry_title": None,
            "rcsb_entity_uniprot": None,
            "chain_match_method": "NO_MATCH",
        }
    )


new_annotations = df.apply(
    lambda row: find_chain_annotation(row["pdb_id"], row["peptide_chain"]), axis=1
)
fixed_df = pd.concat(
    [df.reset_index(drop=True), new_annotations.reset_index(drop=True)], axis=1
)
COFACTOR_PATTERNS = {
    "SRC-1 / NCOA1": [
        "\\bncoa1\\b",
        "\\bsrc[- ]?1\\b",
        "steroid receptor coactivator 1",
        "nuclear receptor coactivator 1",
    ],
    "SRC-2 / NCOA2 / TIF2 / GRIP1": [
        "\\bncoa2\\b",
        "\\bsrc[- ]?2\\b",
        "\\btif2\\b",
        "\\bgrip1\\b",
        "steroid receptor coactivator 2",
        "nuclear receptor coactivator 2",
        "transcriptional intermediary factor 2",
    ],
    "SRC-3 / NCOA3 / AIB1 / ACTR": [
        "\\bncoa3\\b",
        "\\bsrc[- ]?3\\b",
        "\\baib1\\b",
        "\\bactr\\b",
        "steroid receptor coactivator 3",
        "nuclear receptor coactivator 3",
    ],
    "CBP / CREBBP": ["\\bcrebbp\\b", "\\bcbp\\b", "creb[- ]binding protein"],
    "p300 / EP300": ["\\bep300\\b", "\\bp300\\b"],
    "NCoR / NCOR1": ["\\bncor1\\b", "\\bncor\\b", "nuclear receptor corepressor 1"],
    "SMRT / NCOR2": ["\\bncor2\\b", "\\bsmrt\\b", "silencing mediator"],
}


def classify_annotation(row):
    description = str(row.get("rcsb_description", ""))
    title = str(row.get("rcsb_entry_title", ""))
    # Entry titles can mention other chains; retain ambiguous identities for review.
    text = (description + " " + title).lower()
    identities = []
    for name, patterns in COFACTOR_PATTERNS.items():
        if any((re.search(pattern, text, re.I) for pattern in patterns)):
            identities.append(name)
    if not identities:
        if "coactivator" in text:
            identities.append("COACTIVATOR — exact identity unresolved")
        elif "corepressor" in text or "co-repressor" in text:
            identities.append("COREPRESSOR — exact identity unresolved")
    return "; ".join(sorted(set(identities)))


fixed_df["cofactor_identity"] = fixed_df.apply(classify_annotation, axis=1)
fixed_df["has_annotation_evidence_fixed"] = (
    fixed_df["cofactor_identity"].fillna("").str.len() > 0
)
fixed_df["structural_cofactor_candidate"] = fixed_df["receptor_contact"].fillna(
    False
) & (fixed_df["has_LXXLL"].fillna(False) | fixed_df["has_CoRNR"].fillna(False))
fixed_df["cofactor_identity_status"] = "UNRESOLVED"
fixed_df.loc[fixed_df["has_annotation_evidence_fixed"], "cofactor_identity_status"] = (
    "IDENTIFIED"
)


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


fixed_df["final_cofactor_confidence"] = fixed_df.apply(final_confidence, axis=1)
fixed_df.to_csv(DATA_DIR / "peptide_cofactor_validation_FIXED.csv", index=False)
confirmed_fixed = fixed_df[
    fixed_df["final_cofactor_confidence"].isin(
        ["HIGH_CONFIDENCE_IDENTIFIED", "HIGH_CONFIDENCE_IDENTITY_UNRESOLVED"]
    )
].copy()
confirmed_fixed.to_csv(DATA_DIR / "confirmed_cofactor_complexes_FIXED.csv", index=False)
identified_df = confirmed_fixed[
    confirmed_fixed["cofactor_identity_status"] == "IDENTIFIED"
].copy()
identified_df.to_csv(DATA_DIR / "identified_cofactor_complexes.csv", index=False)
review_fixed = fixed_df[
    fixed_df["final_cofactor_confidence"].isin(
        ["ANNOTATED_CONTACT", "REVIEW", "CONTACT_ONLY"]
    )
].copy()
review_fixed.to_csv(DATA_DIR / "cofactor_review_needed_FIXED.csv", index=False)
pd.DataFrame(api_errors).to_csv(DATA_DIR / "rcsb_annotation_errors.csv", index=False)
print("RESULTS AFTER FIX")
print("Peptide rows:", len(fixed_df))
print(
    "Chains successfully matched to RCSB entity:",
    (fixed_df["chain_match_method"] != "NO_MATCH").sum(),
)
print(
    "Rows with a non-empty RCSB description:",
    fixed_df["rcsb_description"].fillna("").str.len().gt(0).sum(),
)
print(
    "Rows where cofactor identity was identified:",
    fixed_df["has_annotation_evidence_fixed"].sum(),
)
print("Confirmed structural cofactor PDBs:", confirmed_fixed["pdb_id"].nunique())
print("Confirmed + identity-known PDBs:", identified_df["pdb_id"].nunique())
print("\nConfidence categories:")
display(
    fixed_df["final_cofactor_confidence"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="n_peptides")
)
print("\nIdentified cofactors:")
display(
    fixed_df.loc[fixed_df["has_annotation_evidence_fixed"], "cofactor_identity"]
    .value_counts()
    .rename_axis("cofactor")
    .reset_index(name="n_peptides")
)
print("\nExample annotations:")
display(
    fixed_df[
        [
            "pdb_id",
            "receptor",
            "peptide_chain",
            "peptide_sequence",
            "rcsb_description",
            "cofactor_identity",
            "chain_match_method",
            "final_cofactor_confidence",
        ]
    ].head(20)
)
print("\nFiles created:")
print("1. peptide_cofactor_validation_FIXED.csv")
print("2. confirmed_cofactor_complexes_FIXED.csv")
print("3. identified_cofactor_complexes.csv")
print("4. cofactor_review_needed_FIXED.csv")
print("5. rcsb_annotation_errors.csv")
