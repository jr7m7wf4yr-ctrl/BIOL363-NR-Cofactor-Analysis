# October 2026: dataset re-check

Following the supervisor's feedback, every step at which structures were excluded (1,994 → 924 → 718 → 708) was re-checked, and the reason for each exclusion was recorded. This folder holds the scripts and results of that re-check, plus the cofactor counts and experimental metadata. Author: Madina Maldarbekova.

Dataset used for the counts: **716 PDB structures** (the 708 curated in September plus 8 human ERα structures with stapled SRC-2 peptides). The five structures found by the sequence search that are linked to gibbon or rat UniProt entries (5X8S, 5X8U, 5X8W, 5X8X, 7ZTV) are **not included**, following the supervisor's decision.

## Contents

| Folder | Script | Results | What it does |
|---|---|---|---|
| `01_dataset/` | `exclusion_audit.py` | `exclusion_audit.csv`, `exclusion_audit_summary.csv` | Every excluded structure or peptide with the stage, reason and verdict (correct / likely error / review). Totals: 1,190 correct, 37 likely error, 75 review. |
| `01_dataset/inventory_check/` | `inventory_check.py` | `uniprot_search_vs_snapshot.csv`, `found_by_sequence_not_in_inventory.csv` | Repeats the UniProt-based search and adds a sequence search (≥90% identity). Lists 44 structures not in the inventory: 39 non-human and 5 linked to a non-human UniProt entry. |
| `01_dataset/added_structures/` | `screen_added_structures.py`, `validate_added_peptides.py` | `added_structures_filter_results.csv`, `added_peptide_cofactor_validation_FIXED.csv` | Runs those 5 structures through verbatim copies of the team's filter code (scripts 03–05). Three would pass; none is included (see above). |
| `01_dataset/peptide_contact_check/` | `check_lost_peptides.py` | `lost_peptides_contact_check.csv`, `D1_long_chain_identity.csv` | Re-checks the 206 structures lost between 924 and 718: which chain the peptide touches and whether it sits in the AF-2 groove. Finds the 8 stapled ERα structures lost by mistake. |
| `01_dataset/cofactor_identity/` | `fix_cofactor_identity.py` | `cofactor_identity_v2.csv`, `cofactor_uniprot_sequences.json` (cache) | Cofactor name for each of the 1,251 peptides, from the peptide's UniProt link, checked against its sequence. 1,230 named (98%). |
| `05_metadata_ligands/` | `extract_metadata.py` | `structure_metadata.csv` | Experimental metadata from the mmCIF headers (resolution, R-work/R-free, space group, cell, temperature, crystallisation) and peptide-to-receptor stoichiometry, for the 716 structures (708 curated + 8 stapled ERα). |
| `06_analysis/` | `cofactor_counts.py`, `plot_cofactor_simple.py`, `plot_receptor_simple.py` | `cofactor_counts_overall.csv`, `cofactor_by_receptor.csv`, `cofactors_per_receptor.csv`, `figures/` | Cofactor and receptor counts per PDB entry and Figures 1–2 of the October report. The 8 stapled peptides are named SRC-2 by hand (no UniProt link). |
| `06_analysis/` | `plot_metadata_simple.py` | `figures/fig10_quality.png`, `fig11_crystal_form.png`, `fig12_stoichiometry.png` | Metadata plots: resolution vs R-free (29 weaker structures), share of each receptor's structures in its most common space group, and receptor–peptide copies per crystal. |
| `06_analysis/` | `unusual_structures_list.py` | `unusual_structures_list.csv` | The unusual structures kept for now (supervisor's decision, Nov 2026): 2 crystals with 6 or 8 copies (4P6X, 9ECF), 4 extra SRC-2 peptides outside the groove (1YOK, 1ZDU, 9H66) and the 29 weaker structures (resolution > 3 Å or R-free > 0.30). |

## Running the scripts

The scripts were run from the project folder `BIOL 363`, next to a clone of this repository (`BIOL363-NR-Cofactor-Analysis/data/snapshot` is read for the September tables). The folder names here keep that layout. Some inputs are not uploaded because they are large or belong to other parts of the project:

- mmCIF files (`raw/`): download again from the RCSB PDB (script 03 does this).
- `01_dataset/cofactor_complexes_handoff_clean.csv`: September curated receptor–peptide pairs.
- `04_cofactor_geometry/cofactor_geometry_full.csv`: geometry table (receptor display names and the groove check).
- `06_analysis/pca_clustering.py`: plotting style imported by `cofactor_counts.py`.

Python 3.13, pandas, NumPy, Biopython, matplotlib, requests.
