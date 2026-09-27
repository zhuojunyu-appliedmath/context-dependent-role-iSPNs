This repository contains the analysis code and derived data used to generate figures for "Basal ganglia pathway function depends on striatal state context during decision-making" [link]

## Quick start

Run the notebooks in numerical order. 
Notebooks 01-10 create derived data tables; notebooks 01, 02, 04, 06, 07, 08, and 11 generate figures in the Manuscript and the Supporting Information.

## Figures in the Manuscript
Fig. 1A: schematic of reduced CBGT circuit

Fig. 1B: schematic of mouse two-choice task

Fig. 1C(i): simulated 12D SPN profiles | `01_cbgt_reference_clustering.ipynb` | `figures/main/Fig1C_i_cbgt_12D_profiles`

Fig. 1C(ii): fast/slow SPN correlations | `01_cbgt_reference_clustering.ipynb` | `figures/main/Fig1C_ii_cbgt_fast_slow_correlations`

Fig. 1D: two-stage clustering schematic and validation | `01_cbgt_reference_clustering.ipynb` | `figures/main/Fig1D_two_stage_clustering`

Fig. 2: example empirical SPN profiles | `06_spn_clustering_figures.ipynb` | `figures/main/Fig2_example_empirical_spn_profiles`

Fig. 3: CBGT and IBL CLAWs,  statistics from `09_claw_and_control_ensemble.ipynb` 

Fig. 4: model and empirical prediction tests | `11_prediction_boxplots.ipynb` | `figures/main/Fig4_multiplexed_iSPN_roles`

Fig. 5A: schematic of control ensembles 

Fig. 5B: changes in control ensemble engagement along the CBGT CLAW,  statistics from `09_claw_and_control_ensemble.ipynb` 


## Figures in the Supporting Information 
Fig. S1A: schematic of full CBGT circuit 

Fig. S1B: firing rates of CBGT populations 

Fig. S1C: median decision time distributions of 300 CBGT networks

Fig. S1D: SPN thresholds (Fig. S1A-D reproduced from https://doi.org/10.1371/journal.pcbi.1012966)

Fig. S2: Stage 1 bin-count validation | `02_cbgt_feature1_bin_count.ipynb` | `figures/supporting/FigS2_cbgt_feature1_bin_count_stage1_ari`

Fig. S3: all Steinmetz clustering results | `06_spn_clustering_figures.ipynb` | `figures/supporting/FigS3_steinmetz_profiles_and_correlations`

Fig. S4: all IBL clustering results | `06_spn_clustering_figures.ipynb` | `figures/supporting/FigS4_ibl_profiles_and_correlations`

Fig. S5: Steinmetz shuffle controls | `04_steinmetz_shuffle_controls.ipynb` | `figures/supporting/FigS5_steinmetz_shuffle_controls`

Fig. S6: Steinmetz post-clustering assessment of potential FSI contamination | `07_steinmetz_fsi_audit.ipynb` | `figures/supporting/FigS6_steinmetz_postclustering_spn_fsi_waveform_audit`

Fig. S7: IBL post-clustering assessment of potential FSI contamination | `08_ibl_fsi_audit.ipynb` | `figures/supporting/FigS7_ibl_postclustering_spn_fsi_waveform_audit`

Fig. S8: schematic of control ensemble in the full CBGT circuit, reproduced from https://doi.org/10.64898/2026.02.17.706272

Fig. S9: changes in drift rate and boundary height along the CBGT CLAW, statistics from `09_claw_and_control_ensemble_figures.ipynb` 


## Notebook workflow

### 01. CBGT reference patterns and validation

`01_cbgt_reference_clustering.ipynb` reads the 300 simulated networks, constructs the 12-dimensional pre-decision features and full pre-decision firing rate table, validates the two-stage clustering against known pathway/channel labels, prepares the simulated CLAW state table, and generates Fig. 1C-D.

### 02. Stage 1 bin-count analysis

`02_cbgt_feature1_bin_count.ipynb` reads the derived 12D feature table from Notebook 01 and tests Stage 1 action-channel recovery with total feature dimensions from 2D to 12D. It does not rerun raw-data preprocessing or Stage 2 clustering. It generates SI Fig. S2.

### 03-05. Empirical SPN inference and shuffle controls

`03_steinmetz_spn_clustering.ipynb` and `05_ibl_spn_clustering.ipynb` apply the same two-stage inference pipeline to the Steinmetz and IBL recordings. The labels saved by these notebooks remain fixed in all downstream analyses.

#### Steinmetz recordings

Notebook 03 requires the original recordings from Steinmetz et al. (2019):

https://www.nature.com/articles/s41586-019-1787-x

Official dataset:

https://figshare.com/articles/dataset/Distributed_coding_of_choice_action_and_engagement_across_the_mouse_brain/9974357

Download and extract the following four sessions:

- `Hench_2017-06-18`
- `Lederberg_2017-12-11`
- `Radnitz_2017-01-12`
- `Richards_2017-11-01`

Place the four session folders directly under `data/source/steinmetz/`

`04_steinmetz_shuffle_controls.ipynb` applies three controls to the Steinmetz recordings: within-unit ISI shuffling, evidence-stratified choice-label shuffling, and fast/slow-label shuffling within each choice condition. Each control uses 50 repetitions per session. Original and shuffled datasets are compared by the percentage of predicted temporal criteria satisfied. It generates SI Fig. S5.

#### IBL recordings

Notebook 05 analyzes recordings from the IBL Brain-Wide Map dataset:

https://www.nature.com/articles/s41586-025-09235-0

Official data-release instructions:

https://docs.internationalbrainlab.org/notebooks_external/data_release_brainwidemap.html

ONE data-access documentation:

https://int-brain-lab.github.io/ONE/notebooks/one_quickstart.html

Users do not need to download the IBL recordings manually or add them to this repository. Notebook 03 connects to the public IBL OpenAlyx server through the ONE API and downloads the required data to the local ONE cache.

### 07-08. Post-clustering assessment of potential FSI contamination
`07_steinmetz_fsi_audit.ipynb` and `08_ibl_fsi_audit.ipynb` assess only empirical units already retained and assigned by Notebooks 03 and 05. They measure peak-channel extracellular waveform width and whole-session firing rate and flag a unit as putative FSI only when its waveform width is below 0.15 ms and its firing rate exceeds 10 Hz. 
They generate SI Figs. S6-S7 and save unit-level annotations for Notebook 09.
These notebooks do not rerun clustering or alter any saved subtype label. The CBGT populations are not subjected to this assessment because their SPN identities are known from the model.

### 09. CLAWs and control ensembles

`F_matrix.npy` and `D_matrix.npy` under `data/source/cbgt` were generated in https://doi.org/10.1371/journal.pcbi.1012966

`09_claw_and_control_ensemble.ipynb` binarizes the four inferred SPN populations, compresses consecutive repetitions of the same state, estimates transition and terminal probabilities, and generates state-level behavioral statistics for Fig. 3. 

The notebook also estimates state-level choice probability, decision time, transition probability, and terminal probability. It then projects CBGT state transitions onto the control ensembles and DDM parameters to generate the statistics used in Fig. 5 and SI Fig. S9.

### 10-11. Prediction analyses and boxplots

`10_prediction_statistics.ipynb` performs statistical analyses for three predictions:

1. Left-choice probability in iSPN-only versus dSPN-containing states.
2. Terminal probability before and after same-channel dSPN+iSPN coactivation.
3. Decision time with and without later opponent-channel iSPN recruitment.

`11_prediction_boxplots.ipynb` generates Fig. 4 from the standardized bootstrap, raw decision time, and significance test tables.
