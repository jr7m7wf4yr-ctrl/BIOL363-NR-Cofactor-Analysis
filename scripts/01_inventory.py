from settings import DATA_ROOT, display

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm.auto import tqdm

OUTPUT_DIR = DATA_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = 8
TIMEOUT = 30
MAX_RETRIES = 3
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
print("Output folder:", OUTPUT_DIR)
NR_UNIPROT_IDS = {
    "AR": "P10275",
    "ESR1": "P03372",
    "ESR2": "Q92731",
    "ESRRA": "P11474",
    "ESRRB": "O95718",
    "ESRRG": "P62508",
    "GR": "P04150",
    "MR": "P08235",
    "PR": "P06401",
    "VDR": "P11473",
    "PPARA": "Q07869",
    "PPARD": "Q03181",
    "PPARG": "P37231",
    "RXRA": "P19793",
    "RXRB": "P28702",
    "RXRG": "P48443",
    "RARA": "P10276",
    "RARB": "P10826",
    "RARG": "P13631",
    "THRA": "P10827",
    "THRB": "P10828",
    "RORA": "P35398",
    "RORB": "Q92753",
    "RORC": "P51449",
    "FXR": "Q96RI1",
    "LXR_ALPHA": "Q13133",
    "LXR_BETA": "P55055",
    "PXR": "O75469",
    "CAR": "Q14994",
    "HNF4A": "P41235",
    "HNF4G": "Q14541",
    "LRH1": "O00482",
    "SF1": "Q13285",
    "NUR77": "P22736",
    "NURR1": "P43354",
    "NOR1": "Q92570",
    "SHP": "Q15466",
    "DAX1": "P51843",
    "TR2": "P13056",
    "TR4": "Q9Y6F6",
    "TLX": "Q9Y466",
    "PNR": "Q9Y5X4",
    "EAR2": "P10588",
    "COUP_TF1": "P10589",
    "COUP_TF2": "P24468",
    "REV_ERB_ALPHA": "P20393",
    "REV_ERB_BETA": "Q14995",
    "GCNF": "Q15406",
}
print("Number of receptors:", len(NR_UNIPROT_IDS))
session = requests.Session()
session.headers.update(
    {"User-Agent": "BIOL363-NR-Cofactor-Project/1.0", "Accept": "application/json"}
)


def request_with_retries(method, url, **kwargs):
    """
    Make an HTTP request with retries.

    Retries temporary errors:
    - timeout
    - connection errors
    - HTTP 429
    - server-side 5xx errors
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.request(method, url, timeout=TIMEOUT, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.RequestException(
                    f"Temporary HTTP error {response.status_code}"
                )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait_time = 2 * attempt
            print(f"\nRequest failed: {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def search_pdb_by_uniprot(uniprot_id):
    """
    Search RCSB for all PDB entries associated with one UniProt accession.
    """
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    response = request_with_retries("POST", RCSB_SEARCH_URL, json=query)
    data = response.json()
    pdb_ids = [result["identifier"] for result in data.get("result_set", [])]
    return sorted(set(pdb_ids))


def collect_all_pdb_ids():
    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_receptor = {
            executor.submit(search_pdb_by_uniprot, uniprot_id): (receptor, uniprot_id)
            for (receptor, uniprot_id) in NR_UNIPROT_IDS.items()
        }
        for future in tqdm(
            as_completed(future_to_receptor),
            total=len(future_to_receptor),
            desc="Searching receptors",
        ):
            (receptor, uniprot_id) = future_to_receptor[future]
            try:
                pdb_ids = future.result()
                for pdb_id in pdb_ids:
                    rows.append(
                        {
                            "receptor": receptor,
                            "uniprot_id": uniprot_id,
                            "pdb_id": pdb_id,
                        }
                    )
            except Exception as e:
                errors.append(
                    {
                        "stage": "RCSB_SEARCH",
                        "receptor": receptor,
                        "uniprot_id": uniprot_id,
                        "pdb_id": None,
                        "error": str(e),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.drop_duplicates()
            .sort_values(["receptor", "pdb_id"])
            .reset_index(drop=True)
        )
    return (df, errors)


(mapping_df, search_errors) = collect_all_pdb_ids()
display(mapping_df.head())
print("Receptor-PDB associations:", len(mapping_df))
print("Unique PDB structures:", mapping_df["pdb_id"].nunique())
print("Search errors:", len(search_errors))
mapping_path = OUTPUT_DIR / "nr_pdb_mapping.csv"
mapping_df.to_csv(mapping_path, index=False)
print("Saved:", mapping_path)


def get_entry_metadata(pdb_id):
    """
    Get entry-level metadata from RCSB Data API.
    """
    url = f"{RCSB_DATA_URL}/{pdb_id}"
    response = request_with_retries("GET", url)
    return response.json()


def parse_entry_metadata(pdb_id, data):
    result = {
        "pdb_id": pdb_id,
        "title": None,
        "experimental_method": None,
        "resolution": None,
        "deposition_date": None,
        "release_date": None,
        "polymer_entity_count": None,
        "nonpolymer_entity_count": None,
    }
    result["title"] = data.get("struct", {}).get("title")
    exptl = data.get("exptl", [])
    if exptl:
        result["experimental_method"] = exptl[0].get("method")
    resolution = data.get("rcsb_entry_info", {}).get("resolution_combined", [])
    if resolution:
        result["resolution"] = resolution[0]
    accession_info = data.get("rcsb_accession_info", {})
    result["deposition_date"] = accession_info.get("deposit_date")
    result["release_date"] = accession_info.get("initial_release_date")
    entry_info = data.get("rcsb_entry_info", {})
    result["polymer_entity_count"] = entry_info.get("polymer_entity_count")
    result["nonpolymer_entity_count"] = entry_info.get("nonpolymer_entity_count")
    return result


def collect_entry_metadata(pdb_ids):
    rows = []
    errors = []
    unique_pdb_ids = sorted(set(pdb_ids))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdb = {
            executor.submit(get_entry_metadata, pdb_id): pdb_id
            for pdb_id in unique_pdb_ids
        }
        for future in tqdm(
            as_completed(future_to_pdb),
            total=len(future_to_pdb),
            desc="Getting RCSB metadata",
        ):
            pdb_id = future_to_pdb[future]
            try:
                raw_data = future.result()
                row = parse_entry_metadata(pdb_id, raw_data)
                rows.append(row)
            except Exception as e:
                errors.append(
                    {
                        "stage": "RCSB_DATA",
                        "receptor": None,
                        "uniprot_id": None,
                        "pdb_id": pdb_id,
                        "error": str(e),
                    }
                )
    metadata_df = pd.DataFrame(rows)
    return (metadata_df, errors)


(metadata_df, metadata_errors) = collect_entry_metadata(mapping_df["pdb_id"])
display(metadata_df.head())
print("Metadata rows:", len(metadata_df))
print("Metadata errors:", len(metadata_errors))
all_nr_df = mapping_df.merge(metadata_df, on="pdb_id", how="left")
display(all_nr_df.head())
print("Rows:", len(all_nr_df))
print("Unique PDB structures:", all_nr_df["pdb_id"].nunique())
summary_df = (
    mapping_df.groupby(["receptor", "uniprot_id"])
    .agg(n_pdb_structures=("pdb_id", "nunique"))
    .reset_index()
    .sort_values("n_pdb_structures", ascending=False)
)
display(summary_df)
all_nr_df.to_csv(OUTPUT_DIR / "all_nr_structures.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "nr_structure_counts.csv", index=False)
all_errors = search_errors + metadata_errors
errors_df = pd.DataFrame(all_errors)
errors_df.to_csv(OUTPUT_DIR / "errors.csv", index=False)
print("Saved:")
print(OUTPUT_DIR / "nr_pdb_mapping.csv")
print(OUTPUT_DIR / "all_nr_structures.csv")
print(OUTPUT_DIR / "nr_structure_counts.csv")
print(OUTPUT_DIR / "errors.csv")
print("Number of receptor labels:", mapping_df["receptor"].nunique())
print("Unique UniProt IDs:", mapping_df["uniprot_id"].nunique())
print("Unique PDB structures:", mapping_df["pdb_id"].nunique())
print("Total receptor-PDB associations:", len(mapping_df))
print("Total errors:", len(all_errors))
vdr_df = mapping_df[mapping_df["receptor"] == "VDR"]
print("VDR structures:", vdr_df["pdb_id"].nunique())
display(vdr_df.head(20))
