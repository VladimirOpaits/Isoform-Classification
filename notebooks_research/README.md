# Fast Isoform Quality Classifier for Long-Read RNA-seq

A lightweight ML pipeline for filtering long-read transcriptome data, designed as a fast complementary tool to SQANTI3's quality control workflow.

## Motivation

SQANTI3 is the standard tool for quality control of long-read RNA-seq isoforms, but its filtering step can be a bottleneck for large datasets. This project explores whether a simple ML model trained on SQANTI3's QC features can reproduce isoform quality classification at a fraction of the runtime, enabling rapid filtering of large transcriptomes.

The dataset used is the WTC11 human iPSC line (LRGASP, ENCODE), processed through SQANTI3 QC to obtain ~490k isoform models with their structural categories and quality descriptors.

## Approach

### Label construction

SQANTI3 classifies isoforms into structural categories (FSM, ISM, NIC, NNC, etc.) and subcategories. Based on these, isoforms were initially mapped to four quality tiers: `real_high`, `real_mid`, `uncertain`, `artifact`.

mapping = {
    ('full-splice_match', 'reference_match'):              'real_high',
    ('full-splice_match', 'alternative_3end'):             'real_high',
    ('full-splice_match', 'alternative_5end'):             'real_high',
    ('full-splice_match', 'alternative_3end5end'):         'real_high',
    ('novel_in_catalog',  'combination_of_known_junctions'): 'real_high',
    ('novel_in_catalog',  'combination_of_known_splicesites'): 'real_mid',

    ('full-splice_match', 'mono-exon'):                    'real_mid',
    ('novel_in_catalog',  'intron_retention'):             'uncertain',
    ('novel_in_catalog',  'mono-exon_by_intron_retention'): 'uncertain',
    ('novel_not_in_catalog', 'at_least_one_novel_splicesite'): 'uncertain',
    ('incomplete-splice_match', '3prime_fragment'):        'uncertain',
    ('incomplete-splice_match', '5prime_fragment'):        'uncertain',
    ('incomplete-splice_match', 'internal_fragment'):      'uncertain',

    ('novel_not_in_catalog', 'intron_retention'):          'artifact',
    ('incomplete-splice_match', 'intron_retention'):       'artifact',
    ('incomplete-splice_match', 'mono-exon'):              'artifact',
    ('fusion', 'intron_retention'):                        'artifact',
    ('fusion', 'multi-exon'):                              'artifact',
    ('genic_intron', 'mono-exon'):                         'artifact',
    ('genic_intron', 'multi-exon'):                        'artifact',
    ('genic', 'mono-exon'):                                'artifact',
    ('genic', 'multi-exon'):                               'artifact',
    ('intergenic', 'mono-exon'):                           'artifact',
    ('intergenic', 'multi-exon'):                          'artifact',
    ('antisense', 'mono-exon'):                            'artifact',
    ('antisense', 'multi-exon'):                           'artifact',
}

For training, only the unambiguous extremes were used — `real_high` (n=202,421) and `artifact` (n=66,905) — to avoid noise from subjective intermediate labels. The model is then evaluated on `real_mid` and `uncertain` as held-out classes to test whether it learns a continuous quality signal rather than just memorizing the binary boundary.

**Note on labels:** the SQANTI3 authors explicitly recommend external validation data (CAGE-seq, Quant-seq) for defining ground truth, and demonstrate that even FSM isoforms contain artifacts. The structural-category-based labels used here are an approximation. CAGE/polyA validation data for WTC11 is publicly available (GEO `GSE185917`, ENCODE `ENCSR322MWL`) and would substantially improve label quality if integrated.

### Features

- **Sequence**: raw nucleotide sequence (max 8000 bp, one-hot encoded ATGC; N and padding → zero vector)
- **Tabular** (21 features from SQANTI3): `strand`, `length`, `exons`, `ref_length`, `ref_exons`, `diff_to_gene_TSS/TTS`, `RTS_stage`, `all_canonical`, `perc_A_downstream_TTS`, `protein_length`, `has_orf`, `FSM_class_A/B/C`, `orf_type_*`

Features that would require a complete reference comparison (and thus duplicate SQANTI3's most expensive step) were avoided where possible. The `bite` feature was constant in our dataset and effectively unused.

### Models

Three architectures were compared on the binary task:

| Model | Architecture | Val AUC | Val F1 | Notes |
|-------|--------------|---------|--------|-------|
| Random Forest (4-class) | Sklearn RF on tabular | 0.84 acc | — | Baseline; struggles with `uncertain` |
| Conv-only (4-class) | Dilated 1D-CNN on sequence | 0.64 acc | — | Collapses predictions to majority class |
| Tab-only MLP (binary) | 2-layer MLP on tabular | 0.989 | 0.950 | Fast, simple, near-optimal |
| Combined (binary) | CNN(seq) + MLP(tab) → MLP | 0.988 | 0.963 | Best F1, marginal AUC change |

The combined model uses a 3-block 1D-CNN over one-hot sequences (kernels 7/5/3, channels 128/256/128, with BatchNorm and MaxPool) and a parallel MLP over tabular features. Their representations are concatenated and passed through a final MLP head.

## Key result: out-of-distribution validation

The binary model was trained only on `real_high` vs `artifact`, but evaluated on the full dataset including `real_mid` and `uncertain` — classes never seen during training.

Score distribution by original SQANTI3 class (combined model):

| Class | Mean | Median | 25% | 75% |
|-------|------|--------|-----|-----|
| `artifact` | 0.130 | 0.004 | 0.000 | 0.200 |
| `uncertain` | 0.349 | 0.293 | 0.088 | 0.552 |
| `real_mid` | 0.503 | 0.521 | 0.234 | 0.783 |
| `real_high` | 0.936 | 0.995 | 0.980 | 0.999 |

The model produces a smooth, monotonic gradient across all four classes — including the two it never trained on. This suggests it learned a continuous notion of isoform quality rather than a hard binary boundary. The tab-only model shows the same pattern with slightly more conservative scores.

## Ablation: where does the signal come from?

| Setup | Val AUC | Val F1 |
|-------|---------|--------|
| Tab-only | 0.989 | 0.950 |
| Combined (CNN + Tab) | 0.988 | 0.963 |

Most of the predictive signal lives in the tabular features — particularly `all_canonical` (0.61 correlation with label), `FSM_class_*`, and `length`. The CNN over raw sequence adds a small but measurable improvement in F1 (+1.3 points), suggesting it captures information not fully encoded in the QC features, but the gain is modest relative to the added training cost.

A sequence-only model was not run on the binary task; an earlier 4-class CNN-only experiment failed to learn meaningful boundaries.

## Sanity checks

- No isoform ID overlap between train/val/test splits
- Sequence-level overlap < 0.1% (different isoforms with identical sequences)
- No feature has > 0.65 absolute correlation with the label
- 269,326 isoforms in the binary dataset, 188,262 unique sequences in train

## Practical use

The model outputs a continuous confidence score (0–1) per isoform. Threshold choice depends on the downstream task:

- **0.9** — strict filtering, retains ~75% of `real_high` and almost no other class
- **0.5** — moderate, retains `real_high` and roughly half of `real_mid`
- **0.3** — permissive, removes only clear artifacts

Tab-only inference: ~55k isoforms/second on a single GPU (RTX 4050).

## Limitations

1. **Labels are derived from SQANTI3 structural categories, not orthogonal validation.** The authors of SQANTI3 explicitly recommend CAGE/polyA-seq data for ground truth — this would replace the current heuristic mapping and likely improve calibration on `uncertain`/`real_mid`.
2. **The model depends on SQANTI3 QC output as input.** It accelerates filtering, not the QC step itself.
3. **Trained on a single cell line (WTC11).** Generalization across tissues/organisms is untested.
4. **The 4-class CNN-only experiment failed**, suggesting raw sequence is insufficient on its own without QC features for this label scheme.