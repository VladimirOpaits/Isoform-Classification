import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from settings import DATA_PATH

df = pd.read_csv(DATA_PATH, sep='\t', na_values='NA')

DROP_COLS = [
    'n_indels', 'n_indels_junc',
    'within_polyA_site', 'dist_to_polyA_site',
    'ORF_seq', 'seq_A_downstream_TTS',
    'chrom',
    'associated_gene', 'associated_transcript',
]

df = df.drop(columns=DROP_COLS)
df = df.set_index('isoform')

MIN_SET = 250

tp_primary = df[
    (df['structural_category'] == 'full-splice_match') &
    (df['subcategory'] == 'reference_match') &
    (df['exons'] > 1)
].index

tp_fallback = df[
    (df['structural_category'] == 'full-splice_match') &
    (df['exons'] > 1)
].index

tp_idx = tp_primary if len(tp_primary) >= MIN_SET else tp_fallback

tn_primary = df[
    (df['structural_category'] == 'novel_not_in_catalog') &
    (df['all_canonical'] == 'non_canonical') &
    (df['exons'] > 1)
].index

tn_fallback = df[
    (df['structural_category'] == 'novel_not_in_catalog') &
    (df['exons'] > 1)
].index

tn_idx = tn_primary if len(tn_primary) >= MIN_SET else tn_fallback

df['label'] = -1
df.loc[tp_idx, 'label'] = 1
df.loc[tn_idx, 'label'] = 0

print(f"TP: {len(tp_idx)}  ({'primary' if len(tp_primary) >= MIN_SET else 'fallback'})")
print(f"TN: {len(tn_idx)}  ({'primary' if len(tn_primary) >= MIN_SET else 'fallback'})")
print(f"Unlabeled: {(df['label'] == -1).sum()}")

LABEL_LEAK_COLS = [
    'structural_category',
    'subcategory',
    'all_canonical',
    'diff_to_TSS',
    'diff_to_TTS',
]

df = df.drop(columns=LABEL_LEAK_COLS)

bool_cols = ['RTS_stage', 'within_CAGE_peak', 'polyA_motif_found', 'bite']
for col in bool_cols:
    df[col] = df[col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

df['strand'] = df['strand'].map({'+': 1, '-': 0})
df['coding'] = df['coding'].map({'coding': 1, 'non_coding': 0})

df['FSM_class'] = df['FSM_class'].map({'A': 0, 'B': 1, 'C': 2}).fillna(-1).astype(int)

df['predicted_NMD'] = df['predicted_NMD'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}).fillna(-1).astype(int)

motif_map = {m: i + 1 for i, m in enumerate(sorted(df['polyA_motif'].dropna().unique()))}
df['polyA_motif'] = df['polyA_motif'].map(motif_map).fillna(0).astype(int)

df['min_cov_pos'] = (
    df['min_cov_pos']
    .str.extract(r'(\d+)')[0]
    .astype(float)
    .fillna(0)
    .astype(int)
)

df['ref_length'] = df['ref_length'].fillna(0)
df['ref_exons'] = df['ref_exons'].fillna(0)

df['diff_to_gene_TSS'] = df['diff_to_gene_TSS'].fillna(df['diff_to_gene_TSS'].median())
df['diff_to_gene_TTS'] = df['diff_to_gene_TTS'].fillna(df['diff_to_gene_TTS'].median())

cds_cols = ['ORF_length', 'CDS_length', 'CDS_start', 'CDS_end',
            'CDS_genomic_start', 'CDS_genomic_end']
df[cds_cols] = df[cds_cols].fillna(0)

df['dist_to_CAGE_peak'] = df['dist_to_CAGE_peak'].fillna(-10001)

df['sd_cov'] = df['sd_cov'].fillna(0)
df['min_cov'] = df['min_cov'].fillna(0)
df['min_sample_cov'] = df['min_sample_cov'].fillna(0)
df['bite'] = df['bite'].fillna(0).astype(int)

df['polyA_dist'] = df['polyA_dist'].fillna(0)
df['ratio_TSS'] = df['ratio_TSS'].fillna(df['ratio_TSS'].median())
df['ratio_exp'] = df['ratio_exp'].fillna(0)

labeled = df[df['label'] != -1].copy()
unlabeled = df[df['label'] == -1].copy()

print(f"Labeled:   {len(labeled)}  (TP={labeled['label'].sum()}, TN={(labeled['label'] == 0).sum()})")
print(f"Unlabeled: {len(unlabeled)}")

labeled.to_parquet('data/labeled.parquet')
unlabeled.to_parquet('data/unlabeled.parquet')
