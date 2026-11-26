# November 2026: clustering, cofactor vs receptor, descriptor importance

Following the supervisor's notes, the binding geometry was clustered with 10 methods, the clusters were compared with receptors, NR families and cofactors, and the descriptors were ranked by importance before and after adding pairwise landmark distances. Author: Madina Maldarbekova. The full write-up (data provenance, methods, results, discussion) is `reports/BIOL 363 Cofactor geometry report.pdf`.

**Main results**

- The only natural cluster split is coactivator vs corepressor; coactivators form one continuous cloud. HDBSCAN is the method to trust.
- The receptor information is there but overlaps: the NR family of unseen PDB entries is predicted 83% of the time on 6 descriptors and 90% on 14 (chance 25%).
- H3–H12 distance and angle rank first and stay first after adding 15 pairwise distances, so H12 position drives the differences.
- With corepressors included the cofactor explains more of the geometry than the receptor (27–42% vs 17–19%); among coactivators the receptor explains more.
- The bound peptide splits RORγ (RIP140 vs SRC), PPARα (PGC-1α vs SRC-1) and RXRα; FXR and ERα do not split by peptide; PPARγ splits only by cofactor type.
- The 29 weaker structures (resolution > 3 Å or R-free > 0.30) do not change the clusters (ARI 0.95–1.00).

## Descriptor tables

| Folder | Table | Units | Descriptors |
|---|---|---|---|
| `06_analysis/clustering_bota/` | `clustered_table_all_units.csv` (team helix-axis table, input) | 1,105 | 6: angles and distances between the H3, H12 and motif axes |
| `06_analysis/clustering/` | `04_cofactor_geometry/cofactor_geometry_full.csv` (not uploaded) | 1,106 | 14: pocket centroid, H12 and H12-core distances and angles, contacts |
| `06_analysis/` | `pairwise_landmarks.csv` (made here) | 1,105 | 15 pairwise distances between Lys clamp (H3), Glu AF-2 (H12), motif L1, L4, L5 and the pocket centroid |

## Contents

| Folder | Script | Results | What it does |
|---|---|---|---|
| `06_analysis/` | `clustering_benchmark.py` (`bota` = 6-descriptor table) | `clustering_bota/`, `clustering/`: `benchmark_metrics.csv/png`, `k_scan.csv`, `descriptor_ranking.csv`, `cluster_drivers.csv`, `cluster_labels_all_methods.csv`, `method_agreement_ARI.csv`, `top3_comparison.png`, `map_pca.png`, `receptors_map_pca.png` | One script for the supervisor's clustering notes: 10 methods (one per family), internal metrics, bootstrap stability, agreement with receptor / family / motif, best method, drivers; maps coloured by receptor, NR family and PPAR / ER subtypes. `classify` step: random-forest classification (`classification_scores.csv`, `classification_confusion.png`). |
| `06_analysis/` | `clustering_method_views.py`, `clustering_family_views.py` | `method_views.png`, `family_colour/` | Each method in its own view (dendrogram, SOM grid, density peaks …), coloured by NR family. `family_colour/` = SOM, Grid, BIRCH, affinity propagation, mean shift on the 6-descriptor table. |
| `06_analysis/compare_6_vs_14_descriptors/` | `compare_6_vs_14_descriptors.py` | `compare_*.png`, `metrics_both.csv` | HDBSCAN, Ward, Spectral, Gaussian mixture with the same settings on both descriptor sets. The number of clusters matters more than the descriptor set. |
| `06_analysis/cofactor_vs_receptor/` | `cofactor_vs_receptor.py` | `variance_partitioning.csv`, `peptide_effect_within_receptor.csv`, 3 figures | Share of the geometry explained by receptor and by cofactor (adjusted R²), examples on the PCA map, peptide effect inside each receptor. |
| `06_analysis/` | `resolution_check.py` | `resolution_check.txt` | Weaker structures: 45 of 1,103 units; clusters unchanged without them; over-represented in HDBSCAN noise. |
| `06_analysis/` | `plot_rorg_motifs.py`, `plot_rorg_cluster2.py` | `figures/fig13`, `fig15`, `rorg_cluster_motifs.csv` | RORγ HDBSCAN groups by motif: the RIP140 group has no LHRLL (23/23 LLQLL); the SRC group is 30/31 LHRLL. |
| `06_analysis/` | `plot_rorg_confound.py` | `figures/fig14`, `rorg_confound_check.csv` | RORγ split vs ligand state and crystal form; what differs between the groups. |
| `06_analysis/` | `plot_motif_by_receptor.py`, `plot_how_they_split.py` | `figures/fig16`, `fig17`, `motif_by_receptor*.csv` | The same within-receptor check for ERα, PPARγ, FXR, RXRα, RORγ, PPARα, and what the groups follow when it is not the peptide. |
| `06_analysis/` | `pairwise_landmarks.py`, `descriptor_importance.py` | `pairwise_landmarks.csv`, `pairwise_landmark_residues.csv`, `descriptor_importance.csv`, `descriptor_importance_changes.txt`, `figures/fig18` | Landmarks per unit (chosen per receptor, checked against the literature), 15 pairwise distances, rankings (ε² between receptors, Cliff's δ coactivator vs corepressor, random-forest importance for NR family), and what changes in the clustering. |
| `reports/` | | report PDF, slides | Report; slides on data quality and the RORγ check; title page. |

## Running the scripts

Run from `BIOL 363/06_analysis/`, in this order (the folder names here keep that layout):

```
python clustering_benchmark.py bota          # 6 descriptors; without 'bota' = 14 descriptors
python clustering_benchmark.py bota classify
python clustering_method_views.py bota
python clustering_family_views.py bota
python compare_6_vs_14_descriptors.py bota
python compare_6_vs_14_descriptors.py ours
python compare_6_vs_14_descriptors.py compose
python cofactor_vs_receptor.py
python resolution_check.py
python plot_rorg_motifs.py
python plot_rorg_cluster2.py
python plot_rorg_confound.py
python plot_motif_by_receptor.py
python plot_how_they_split.py
python pairwise_landmarks.py                 # about 5 min, reads the mmCIF files
python descriptor_importance.py
```

Inputs that are not uploaded (large, or from other parts of the project):

- mmCIF files (`raw/structures_718_cofactor/`, `raw/structures_all_1994/`): download again from the RCSB PDB.
- `04_cofactor_geometry/cofactor_geometry_full.csv`: 14-descriptor geometry table.
- `05_metadata_ligands/structure_metadata.csv` and `ligand_states_auto.csv` (the first is in `october_2026/`).
- `01_dataset/cofactor_identity/cofactor_identity_v2.csv` (in `october_2026/`), `01_dataset/numbering/canonical_numbering_maps_v2.json`, `02_pockets_centroids/weighted_pocket_centroids_corrected.csv`, `03_helices/h12_reference_corrected.csv`.

Per-method maps (`per_method/`) and t-SNE maps are not uploaded; the scripts make them again.

Python 3.13, pandas 3.0, NumPy 2.3, scikit-learn 1.7, SciPy 1.16, matplotlib 3.10, Biopython 1.88. All steps use fixed seeds.
