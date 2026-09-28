# BIOL 363 — Nuclear Receptor Cofactor Dataset

This is the code and saved data for Aruzhan Yerzhanova's dataset-construction work in **Nuclear Receptor Coactivator/Corepressor Binding Pattern Analysis**. This README is a guide to what is in the repository and what was changed during cleanup. The later geometry and comparison work belongs to the other parts of the team project.

## Start here

The main result is `data/snapshot/confirmed_cofactor_complexes_FIXED.csv`: **718 unique PDB entries, 1,251 peptide–receptor records and 30 receptor labels**. A structure can contribute several records, so records and structures are different counts.

These are saved results from the original run, not results of a new full download. The CSV contents were preserved exactly. The filename says “confirmed”, but selection confirms the pipeline's motif-plus-contact criteria, not experimental proof of every cofactor identity.

## Code

The code was recovered from the “Analyze Project Pipeline” chat and organised into five stages:

| File in `scripts/` | Purpose |
| --- | --- |
| `01_inventory.py` | Query RCSB for structures linked to the 48 receptor accessions in the code, retrieve metadata, and save the receptor–PDB inventory. |
| `02_uniprot.py` | Check accessions represented in the saved mapping against UniProt for accession match, human origin and reviewed status. This checks the 44 represented labels in the supplied snapshot, not every receptor with no PDB hit. |
| `03_filter.py` | Download mmCIF structures, identify long-chain and short-peptide candidates, apply the zinc screening rule, and save the screening audit. |
| `04_peptides.py` | Check peptide motifs, local descriptions and contacts with long-chain candidates. |
| `05_annotations.py` | Retrieve RCSB polymer-entity annotations, match chain IDs, assign candidate cofactor names, and write the final candidate tables. |
| `settings.py` | Set portable output locations and print tables outside Colab. |

`notebooks/BIOL363_NR_pipeline.ipynb` runs these same scripts in order. It does not contain a second copy of the pipeline. `requirements.txt` lists the Python packages needed. `.gitignore` excludes downloaded structures, new run outputs, environments and temporary files.

## Saved data

All supplied CSVs are in `data/snapshot/`.

| File | What it contains |
| --- | --- |
| `all_nr_structures.csv` | Starting inventory with metadata: 2,029 receptor–PDB associations, 1,994 unique PDB entries and 44 receptor labels. |
| `nr_pdb_mapping.csv` | Links between receptor labels, UniProt IDs and PDB IDs. |
| `nr_structure_counts.csv` | Structure counts for each represented receptor label. |
| `uniprot_validation.csv` | Saved accession-validation results. |
| `structure_filter_results.csv` | Full screening audit, including rejected structures and errors. |
| `basic_filtered_structures.csv` | Basic screening output: 943 rows representing 924 unique PDB entries. |
| `filter_summary.csv` | Counts for the different screening flags. These are not all cumulative steps. |
| `peptide_cofactor_validation_FIXED.csv` | Full peptide audit after the annotation fix: 1,501 rows. |
| `confirmed_cofactor_complexes_FIXED.csv` | Main candidate dataset: 1,251 rows representing 718 unique PDB entries. |

The files ending in ` (1).csv` in Downloads were byte-identical duplicates, so only one copy of each is included. Separate review tables can be recovered from the full audit and were not duplicated here. The large structure folder, ZIP archives, presentation drafts and previous-cohort reports are not included. Stage 3 can download structures again.

## Data sources

- [RCSB PDB](https://www.rcsb.org/)
- [RCSB Search API](https://search.rcsb.org/)
- [RCSB Data API](https://data.rcsb.org/)
- [UniProt](https://www.uniprot.org/)
