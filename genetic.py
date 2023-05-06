"""
Important genetic functions and constants.
"""

import logging
from util import periodic_logging
import numpy as np
from collections import defaultdict, namedtuple, Counter
from itertools import product

from Bio import SeqRecord
from Bio.Data.CodonTable import standard_dna_table
from Bio.Seq import Seq, reverse_complement, complement
import matplotlib.pyplot as plt

# Build codon forward and back tables. Standard_dna_table.back_table lacks stop codons. We use None to represent stop.
# Standard_dna_table.back_table contains only one codon per protein. Here we build one with all synonymous codons.
CODON_FORWARD_TABLE = standard_dna_table.forward_table.copy()
CODON_FORWARD_TABLE.update({codon: 'stop' for codon in standard_dna_table.stop_codons})

CODON_BACK_TABLE = defaultdict(list)
for codon, aa in CODON_FORWARD_TABLE.items():
    CODON_BACK_TABLE[aa].append(codon)
CODON_BACK_TABLE = dict(CODON_BACK_TABLE)

NUCLEOTIDE_ALPHABET = standard_dna_table.nucleotide_alphabet

RESIDUE_COL = 'residue'


# summary of key feature properties
FeatureBrief = namedtuple('FeatureBrief', ['seq_name', 'type', 'start', 'end', 'subfeatures'])


def kmers_in_rc_order(k):
    """
    Return all k-mers ordered such that reverse complements are adjacent with self-reverse complements at the beginning.
    """
    # TODO: define and return a canonical order of this, consistent and symmetric for any k
    kmers_set = {''.join(combo) for combo in product(standard_dna_table.nucleotide_alphabet, repeat=k)}
    kmers_list = []
    while kmers_set:
        kmer = kmers_set.pop()
        rc = reverse_complement(kmer)
        if kmer == rc:
            kmers_list.insert(0, kmer)
            continue
        kmers_list.append(kmer)
        kmers_list.append(rc)
        kmers_set.remove(rc)
    return kmers_list


def transcribe_residues(df, codon_col, residue_col=RESIDUE_COL, inplace=False):
    """Function to add a residue column to dataframe based on codon column."""
    df_out = df if inplace else df.copy()

    # for reordering
    ind = df_out.columns.get_loc(codon_col)
    cols = list(df_out.columns)
    cols.insert(ind, residue_col)

    df_out[residue_col] = df_out[codon_col].map(CODON_FORWARD_TABLE)

    # reorder to place residue column next to codon column
    df_out = df_out.reindex(columns=cols)

    return df_out


def get_feature_briefs(seq_record: SeqRecord.SeqRecord, feature_type_filter: list[str] = None) \
        -> list[FeatureBrief]:
    """
    In a given SeqRecord, traverse feature subfeature tree, and return a list of all feature ranges,
    with optional filter to specific feature types.
    """

    filtered_features = []
    iter_ct, feature_ct = 0, 0

    logging.info('Traversing feature tree ...')

    feature_stack = seq_record.features.copy()
    while feature_stack:
        # periodic_logging(iter_ct, f'{iter_ct:,} iterations, and {feature_ct:,} features.')
        iter_ct += 1

        feature = feature_stack.pop()
        feature_stack.extend(feature.sub_features)

        if not feature_type_filter or feature.type in feature_type_filter:
            feature_ct += 1
            filtered_feature = FeatureBrief(seq_name=seq_record.name, type=feature.type,
                                            start=feature.location.start.position, end=feature.location.end.position,
                                            subfeatures=len(feature.sub_features))
            filtered_features.append(filtered_feature)

    logging.info(f'Traversed feature tree in {iter_ct:,} iterations. Extracted {feature_ct:,} features.')
    return filtered_features


def base_frequency(seq: Seq, bucket=10000):
    """
    In a given sequence, create frequency of each occurring base in buckets.
    """
    bases = 'GATCNgatcn'
    freqs = {b: np.empty(0) for b in bases}

    for start in range(0, len(seq), bucket):
        periodic_logging(start, f'Position {start}', v=bucket*100)
        logging.info(f'Position {start}')
        ctr = Counter(seq[start:start+bucket])
        for b in bases:
            freqs[b] = np.append(freqs[b], ctr[b]/bucket)

    return freqs


def find_symmetry_length_scale(seq: Seq, k=1, start=2*10**6, span=2000):
    """
    Find the length scale at which k-mers first become as frequent as their reverse complement.
    """
    # TODO: define a heurestic to infer the length scale from the data, and return it
    seq_upper=seq.upper()

    # get
    kmers_set = {''.join(combo) for combo in product(NUCLEOTIDE_ALPHABET, repeat=k)}
    kmer_pairs = []
    while kmers_set:
        kmer = kmers_set.pop()
        kmer_rc = reverse_complement(kmer)
        # kmer_rc = complement(kmer)
        if kmer == kmer_rc:
            continue
        kmer_pairs.append((kmer, kmer_rc))
        kmers_set.remove(kmer_rc)

    x = np.arange(40, span, span//100)
    proportion_diff = defaultdict(list)
    for scale in x:
        for kmer, kmer_rc in kmer_pairs:
            proportion_diff[f'{kmer}-{kmer_rc}'].append((seq_upper[start:start+scale].count(kmer)
                                                - seq_upper[start:start+scale].count(kmer_rc))/scale)

    # TODO: Move plots to visuals.py
    plt.figure()
    for label, values in proportion_diff.items():
        plt.plot(x, values, label=label)
    plt.plot(x, 0*x, linestyle='dashed')
    plt.legend()

    return
