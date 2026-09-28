from settings import DATA_ROOT, display

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

OUTPUT_DIR = DATA_ROOT
mapping_file = OUTPUT_DIR / "nr_pdb_mapping.csv"
mapping_df = pd.read_csv(mapping_file)
print("Loaded:", mapping_file)
print("Rows:", len(mapping_df))
print("Unique receptors:", mapping_df["receptor"].nunique())
print("Unique UniProt IDs:", mapping_df["uniprot_id"].nunique())
display(mapping_df.head())
receptors_df = (
    mapping_df[["receptor", "uniprot_id"]]
    .drop_duplicates()
    .sort_values("receptor")
    .reset_index(drop=True)
)
display(receptors_df)
print("Receptors to validate:", len(receptors_df))
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb"


def validate_uniprot(row):
    """
    Check whether the UniProt ID stored in our CSV:
    - exists
    - is human
    - is reviewed
    - returns a sensible gene/protein name
    """
    receptor = row["receptor"]
    accession = row["uniprot_id"]
    url = f"{UNIPROT_URL}/{accession}.json"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        returned_accession = data.get("primaryAccession")
        organism = data.get("organism", {})
        organism_name = organism.get("scientificName")
        taxon_id = organism.get("taxonId")
        is_human = taxon_id == 9606
        entry_type = data.get("entryType", "")
        is_reviewed = "reviewed" in entry_type.lower()
        gene_name = None
        genes = data.get("genes", [])
        if genes:
            gene_name = genes[0].get("geneName", {}).get("value")
        protein_name = None
        protein_description = data.get("proteinDescription", {})
        protein_name = (
            protein_description.get("recommendedName", {})
            .get("fullName", {})
            .get("value")
        )
        accession_matches = returned_accession == accession
        passed = accession_matches and is_human and is_reviewed
        return {
            "receptor": receptor,
            "input_uniprot_id": accession,
            "returned_uniprot_id": returned_accession,
            "gene_name": gene_name,
            "protein_name": protein_name,
            "organism": organism_name,
            "taxon_id": taxon_id,
            "reviewed": is_reviewed,
            "accession_matches": accession_matches,
            "validation_passed": passed,
            "error": None,
        }
    except Exception as e:
        return {
            "receptor": receptor,
            "input_uniprot_id": accession,
            "returned_uniprot_id": None,
            "gene_name": None,
            "protein_name": None,
            "organism": None,
            "taxon_id": None,
            "reviewed": False,
            "accession_matches": False,
            "validation_passed": False,
            "error": str(e),
        }


validation_results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(validate_uniprot, row) for (_, row) in receptors_df.iterrows()
    ]
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Checking UniProt IDs"
    ):
        validation_results.append(future.result())
validation_df = pd.DataFrame(validation_results)
validation_df = validation_df.sort_values("receptor").reset_index(drop=True)
display(validation_df)
print("Receptors in our dataset:", len(validation_df))
print("Passed validation:", validation_df["validation_passed"].sum())
print("Failed validation:", (~validation_df["validation_passed"]).sum())
problems_df = validation_df[validation_df["validation_passed"] == False]
if len(problems_df) == 0:
    print("ALL UniProt IDs passed automatic validation.")
else:
    print(f"{len(problems_df)} entries need checking:")
    display(problems_df)
validation_df.to_csv(OUTPUT_DIR / "uniprot_validation.csv", index=False)
print("Saved:", OUTPUT_DIR / "uniprot_validation.csv")
