"""
Important genetic functions and constants.
"""

import logging
from util import periodic_logging, rd
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
FeatureBrief = namedtuple('FeatureBrief', ['seq_name', 'type', 'start', 'end', 'strand', 'phase', 'subfeatures'])


def kmers_in_rc_order(k):
    """
    Return all k-mers ordered such that reverse complements are adjacent with self-reverse complements at the beginning.
    """
    # TODO: define and return a canonical order of this, consistent and symmetric for any k
    # Read: https://www.biorxiv.org/content/10.1101/2023.03.09.531845v1.full.pdf
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


def code_frequencies(backtable=CODON_BACK_TABLE):
    """
    Return frequency dictionary.
    Output dict: key = number of codons, value = number of amino acids represented with this number of codons.
    """
    return dict(Counter([len(codon_list) for codon_list in backtable.values()]))


def code_frequency_proportions(backtable=CODON_BACK_TABLE):
    """
    Return proportion dictionary.
    Output dict: key = number of codons, value = proportion of amino acids represented with this number of codons.
    """
    freq_dict = code_frequencies(backtable)
    v_sum = sum(freq_dict.values())
    return {k: rd(v/v_sum) for k, v in freq_dict.items()}


def get_feature_briefs(seq_record: SeqRecord.SeqRecord, feature_type_filter: list[str] = None,
                       merge_overlapping_features: bool = True, merge_opposite_strands: bool = False) \
        -> list[FeatureBrief]:
    """
    In a given SeqRecord, traverse feature subfeature tree, and return a list of all feature ranges,
    with optional filter to specific feature types.
    merge_overlapping_features: If true, merge overlapping features.
    merge_opposite_strands: If true merge opposing strands and set strand = 0,
        if false keep them separate, despite the overlap. An analysis of Chr17 shows such cases are only 0.17% of the
        genome.
    """

    filtered_features_dict = defaultdict(list)
    # feature_indices = []
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
            phase = int(feature.qualifiers['phase'][0]) if 'phase' in feature.qualifiers else None
            filtered_feature = FeatureBrief(seq_name=seq_record.name, type=feature.type,
                                            start=feature.location.start.position, end=feature.location.end.position,
                                            strand=feature.strand, phase=phase,
                                            subfeatures=len(feature.sub_features))
            filtered_features_dict[feature.type].append(filtered_feature)

    filtered_features_dict = {ft_type: sorted(list(set(ft_list))) for ft_type, ft_list in filtered_features_dict.items()}

    # merge overlapping features on the same strand

    merged_features = []
    filtered_ct = 0
    for ft_type, ft_list in filtered_features_dict.items():
        ft_merged_list = []
        merged_start, merged_end, merged_subfeatures = ft_list[0].start, ft_list[0].end, ft_list[0].subfeatures
        merged_strand, merged_phase = ft_list[0].strand, ft_list[0].phase
        for ft in ft_list[1:]:
            if (not merge_overlapping_features) or ft.start > merged_end + 1 or \
                    (not merge_opposite_strands and ft.strand != merged_strand):  # do not merge
                # Three cases not to merge:
                # 1. merge_overlapping_features is False.
                # 2. Features don't overlap.
                # 3. Features overlap from opposite strands and merge_opposite_strands is False

                merged_feature = FeatureBrief(seq_name=seq_record.name, type=ft_type, start=merged_start,
                                              end=merged_end, strand=merged_strand, phase=merged_phase,
                                              subfeatures=merged_subfeatures)
                ft_merged_list.append(merged_feature)  # append previous feature
                merged_start, merged_end, merged_subfeatures = ft.start, ft.end, ft.subfeatures
                merged_strand, merged_phase = ft.strand, ft.phase
            else:  # adjacent or overlapping feature
                # merged_start is that of the first feature
                merged_end = ft.end
                merged_subfeatures += ft.subfeatures
                if ft.strand == merged_strand:  # adjacent or overlapping feature with same strand
                    # merged_strand is that of the first feature
                    # merged_phase is of the first feature for +1 strand, last feature for -1 strand
                    if merged_strand == -1:
                        merged_phase = ft.phase
                else:  # adjacent or overlapping feature with opposite strand
                    # to get here, merge_opposite_strands must be True
                    merged_phase = -1  # phase becomes meaningless
                    merged_strand = 0  # strand meaningless

        # final feature
        merged_feature = FeatureBrief(seq_name=seq_record.name, type=ft_type, start=merged_start, end=merged_end,
                                      strand=merged_strand, phase=merged_phase, subfeatures=merged_subfeatures)
        ft_merged_list.append(merged_feature)

        filtered_ct += len(ft_list)
        merged_features.extend(ft_merged_list)

    logging.info(f'Traversed feature tree in {iter_ct:,} iterations. \n\t\tExtracted {filtered_ct:,} unique features.'
                 f'\n\t\tMerged to {len(merged_features):,} non-overlapping features.')
    return merged_features


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
