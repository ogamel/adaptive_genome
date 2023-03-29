"""
Important genetic functions and constants.
"""

import logging
from collections import defaultdict, namedtuple
from itertools import product

from Bio import SeqRecord
from Bio.Data.CodonTable import standard_dna_table
from Bio.Seq import reverse_complement

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
