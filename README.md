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

## What changed during cleanup

- Removed decorative comment banners, repeated spacing, emoji in status messages and an exploratory VDR API call. VDR is still included in the receptor list.
- Removed a displayed example dictionary from the chat that was explanatory text, not executable code.
- Formatted the recovered code consistently and removed unused imports. Short comments remain where a scientific assumption needs explanation.
- Moved package installation out of the pipeline scripts into `requirements.txt` and the notebook setup cell.
- Replaced Colab-only `/content/` paths with repository-relative paths and optional environment variables.
- Replaced notebook-only table display calls with a small console-compatible helper.
- Corrected the `MMCIF2Dict` import to import the callable class. The old import referred to a module; its error was caught and local descriptions were silently left empty. Local descriptions and intermediate annotation-based classifications may therefore differ on a new run.

The receptor list, chain-length cutoffs, zinc rule, motif patterns, contact cutoff and final motif-plus-contact selection rule were retained. Original CSVs were not recalculated or relabelled.

## How to run

Use Python 3.9 or newer. From the repository folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/01_inventory.py
python scripts/02_uniprot.py
python scripts/03_filter.py
python scripts/04_peptides.py
python scripts/05_annotations.py
```

New files are written to `data/generated/`. The snapshot folder is not overwritten. Internet access is needed for RCSB and UniProt requests. A full run downloads many structures and can take time; current database contents may produce different counts from the saved snapshot. Inspect error CSVs after a run.

For Jupyter, open the notebook from this repository. In Colab, first clone the repository and change the working directory to the clone; uploading the notebook alone does not include its scripts.

To reuse an existing structure folder, set `NR_STRUCTURE_DIR` to its path before running stages 3 and 4. To choose a different generated-data folder, set `NR_DATA_DIR`. Avoid pointing `NR_DATA_DIR` at `data/snapshot/` unless intentionally replacing saved outputs.

## How to interpret this dataset

- A chain of at least 50 residues is treated as a receptor candidate, and a chain of 4–49 residues as a peptide candidate. Length alone does not verify receptor identity.
- Two or more zinc residues trigger the likely-DBD flag. This is a practical screen, not a validated LBD annotation.
- Contacts use a 4 Å cutoff in the first model and are pooled across long-chain candidates. Contact strings preserve chain labels, but the code does not resolve a unique, independently verified receptor partner for every peptide.
- Cofactor naming uses entity descriptions and entry titles. Titles can mention other chains. Generic labels and conflicting matches remain in the data.
- In the final snapshot, 170 rows are marked `UNRESOLVED`. Another 125 rows have only a generic coactivator/corepressor label, and 3 have two specific names. The existing `IDENTIFIED` label therefore does not always mean an exact, unambiguous protein identity.
- This recovered code does not implement canonical UniProt residue-number mapping, functional ligand-state classification, or geometry analysis. Any later team work on those steps is separate from the files included here.

## Checks performed during cleanup

- All scripts and notebook code cells passed syntax checks.
- Included CSV copies matched their source files byte for byte; the headline counts were checked against those files.
- Stages 3–5 were run on a small sample using local 1T63 and 4EM9 structures and live RCSB annotations. 1T63 produced one retained cofactor record. 4EM9 was excluded at basic screening because it lacked a short peptide partner.
- The full inventory and all 924 candidate structures were not rerun during cleanup.

## Data sources

- [RCSB PDB](https://www.rcsb.org/)
- [RCSB Search API](https://search.rcsb.org/)
- [RCSB Data API](https://data.rcsb.org/)
- [UniProt](https://www.uniprot.org/)
