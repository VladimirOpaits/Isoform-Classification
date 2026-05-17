# Isoform Classification — CLAUDE.md

NO COMMENTS IN CODE

## Project goal

Train a Random Forest on SQANTI3 QC features from `UHR_chr22_classification.txt` to classify transcripts as real isoforms (1) or artifacts (0). Use OOB score as the primary validation metric (no held-out split needed — RF bootstrap gives unbiased OOB). Run Bayesian hyperparameter search distributed over MPI workers in the cloud.

Trained model is applied to the unlabeled portion of the same dataset to produce a confidence score per transcript.

## Conda environment

```
conda activate bioml   # /home/vlad/miniconda3/envs/bioml, Python 3.13
```

Direct interpreter path (for scripts/Bash): `/home/vlad/miniconda3/envs/bioml/bin/python`

**Installed**: pandas, numpy, scikit-learn 1.8.0, scipy, joblib  
**Missing** — install via conda-forge:
```
conda install -n bioml -c conda-forge mpi4py -y
```

## Data

`data/UHR_chr22_classification.txt` — tab-separated, 3925 rows × 48 columns. SQANTI3 QC output for chromosome 22, UHR sample.

### Label construction

| Label | Criteria | Count |
|-------|----------|-------|
| **TP = 1** | `structural_category == full-splice_match` AND `subcategory != mono-exon` AND `RTS_stage == False` AND `all_canonical == canonical` | ~413 |
| **TN = 0** | `RTS_stage == True` OR `structural_category` in {genic_intron, genic, intergenic, antisense} | ~913 |
| **Unlabeled** | Everything else — ISM fragments, NIC, NNC, FSM mono-exon, fusion | ~2600 |

Rationale:
- FSM non-mono-exon with no RTS and canonical junctions = high-confidence real isoform (matches annotation exactly).
- RTS_stage=True = confirmed reverse transcriptase switching artifact.
- genic_intron / genic / intergenic / antisense = structurally cannot be real spliced isoforms.

### Columns to drop before modeling

| Column(s) | Reason |
|-----------|--------|
| `n_indels`, `n_indels_junc`, `within_polyA_site`, `dist_to_polyA_site` | 100% NA |
| `ORF_seq`, `seq_A_downstream_TTS` | Raw nucleotide sequences — not RF features |
| `isoform` | Transcript ID (index) |
| `chrom` | Constant (chr22) |
| `structural_category`, `subcategory` | Used for labeling; leaks label into features |

### Feature encoding

| Column | Type | Encoding |
|--------|------|----------|
| `strand` | +/- | binary: + → 1, - → 0 |
| `RTS_stage` | True/False | bool → int |
| `all_canonical` | canonical/non_canonical | binary |
| `bite` | True/False | bool → int |
| `within_CAGE_peak` | True/False | bool → int |
| `polyA_motif_found` | True/False | bool → int |
| `predicted_NMD` | True/False/NA | bool → int, NA → -1 |
| `FSM_class` | A/B/C | ordinal: A=0, B=1, C=2; NA → -1 |
| `coding` | coding/non_coding | binary |
| `polyA_motif` | AATAAA etc / NA | label encode; NA → 0 |
| `min_cov_pos` | junction_N | extract N as integer; NA → 0 |

### NA imputation strategy

| Column(s) | Fill value | Reasoning |
|-----------|-----------|-----------|
| `diff_to_TSS`, `diff_to_TTS` | 999999 | Only defined for FSM/ISM; large sentinel signals "no reference match" |
| `ref_length`, `ref_exons` | 0 | No reference transcript match |
| `diff_to_gene_TSS`, `diff_to_gene_TTS` | median | Small fraction missing |
| `CDS_length`, `CDS_start`, `CDS_end`, `CDS_genomic_start`, `CDS_genomic_end`, `ORF_length` | 0 | Non-coding transcripts |
| `dist_to_CAGE_peak` | -10001 | Beyond observed range (-10000 to 72); sentinel for "no CAGE signal" |
| `sd_cov`, `min_cov`, `min_sample_cov` | 0 | NA = mono-exon (no junction coverage) |
| `all_canonical` | non_canonical | NA = mono-exon, treat as not junction-validated |
| `bite` | False | NA = mono-exon |
| `polyA_dist` | 0 | No polyA signal found |
| `ratio_TSS` | median | Small fraction missing |
| `ratio_exp` | 0 | Missing expression ratio → zero |

## Pipeline structure

```
data_pipeline/
├── preprocess.py        # load → label → encode → impute → save labeled + unlabeled parquet
├── train_rf.py          # load labeled, train RF with given hyperparams, return OOB score
├── hparam_search.py     # MPI master/worker loop with Bayesian optimizer
├── predict.py           # load best model, score unlabeled set, write results CSV
└── config.py            # hyperparameter space definition
```

## Hyperparameter search design (MPI + manual Bayesian BO)

No scikit-optimize. Bayesian optimization is implemented manually: GP surrogate from `sklearn.gaussian_process.GaussianProcessRegressor` + Expected Improvement computed in numpy. RF trains via sklearn with `n_jobs=1` (MPI handles parallelism, not joblib).

**Architecture**: master-worker over MPI.

```
Rank 0 (master)
  ├── maintains list of (X_observed, y_observed) — hyperparam configs + OOB scores
  ├── fits GP surrogate on observed points
  ├── maximises EI over candidate grid to propose next config
  ├── sends config dict to idle worker (via MPI.Send)
  └── collects OOB scores from workers (MPI.Recv), saves best_params.json

Ranks 1..N (workers)
  ├── MPI.Recv config dict from master
  ├── train RandomForestClassifier(oob_score=True, n_jobs=1, **config)
  ├── return oob_score_ to master via MPI.Send
  └── loop until master sends sentinel (None)
```

**Hyperparameter space** (`config.py`):

```python
SPACE = {
    'n_estimators':      ('int',   100,  2000),
    'max_depth':         ('int',   5,    50),
    'min_samples_split': ('int',   2,    30),
    'min_samples_leaf':  ('int',   1,    15),
    'max_features':      ('float', 0.1,  1.0),   # fraction of features
    'max_samples':       ('float', 0.5,  1.0),   # bootstrap fraction
    'class_weight':      ('cat',   ['balanced', None]),
}
```

Categorical params (class_weight) are one-hot encoded for the GP input space.

**EI implementation**:
```python
from scipy.stats import norm
def expected_improvement(X_cand, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_cand, return_std=True)
    z = (mu - y_best - xi) / (sigma + 1e-9)
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
```

**Workflow**:
```
mpirun -n 8 python data_pipeline/hparam_search.py --n-calls 64 --output best_params.json
```
With 7 workers and 64 total evaluations. Each RF on ~1300 labeled rows is fast (<1s), so 64 evals finish in seconds on a single node; scales to cloud by increasing -n.

## Key decisions & constraints

- **No sequence features** — tabular-only RF, consistent with README ablation showing tabular signal dominates.
- **OOB as validation** — avoids train/val split on a small labeled set (~1326 rows); RF bootstrap gives unbiased OOB out of the box.
- **Class imbalance**: TN ≈ 2× TP — use `class_weight='balanced'` as baseline; search over both.
- **No `structural_category` / `subcategory` as features** — these directly encode the label and would leak.
- **Unlabeled set** — ISM fragments, NIC, NNC, FSM mono-exon; classified after training with the best model.
