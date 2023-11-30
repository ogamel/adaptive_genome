"""
Module for aggregation of nucleotide scores, as a function of annotation features, k-mers, and other properties.
"""

import logging
import random
from typing import Iterator, Callable, Iterable, Optional, Union, List
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio.Data.CodonTable import standard_dna_table

from genetic import get_feature_briefs
from util import periodic_logging, rd, std_to_std_of_mean

ID_COLS = ['k', 'kmer', 'seq_name', 'strand', 'phase', 'ft_start', 'ft_len', 'frame', 'pos']
K_COL, KMER_COL, SEQNAME_COL, STRAND_COL, PHASE_COL, FTSTART_COL, FTLEN_COL, FRAME_COL, POS_COL = ID_COLS
COMPLEMENTED_COLS = [STRAND_COL, FRAME_COL, POS_COL, PHASE_COL, FTSTART_COL, FTLEN_COL]
DILATION_COL = 'dil'
ID_COLS_GAP = [K_COL, KMER_COL, SEQNAME_COL, DILATION_COL, POS_COL]
VALUE_COLS = ['count', 'score_mean', 'score_std']
COUNT_COL, SCORE_MEAN_COL, SCORE_STD_COL = VALUE_COLS

USE_SOFTMASKED = True  # flag to whether to use softmasked nucleotides, in small letters e.g. 'gatc'
NUCLEOTIDES = set(standard_dna_table.nucleotide_alphabet)


def score_stats_by_feature_type(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                                scorer: Callable[[str, int, int], list[float]]) -> pd.DataFrame:
    """
    Do basic analysis of score statistics by feature type, for a passed in score function.
    """
    # scorer: function that takes seqname, start and end, returning score values for specified subsequence.
    # Build set containing, for each feature, its type and score statistics if they exist (over all sequences).
    # this represents the subset of data where we have annotations, a sequence, and a score.
    # Traverse feature tree depth first.

    seq_records = seq_records_gen()

    feature_data_rows = []
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        feature_briefs = get_feature_briefs(seq_record)
        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}.', v=len(feature_briefs)//10)
            scores = scorer(seq_name, ft.start, ft.end)
            s1 = sum(scores)
            s2 = sum([s**2 for s in scores])

            # accumulate counts, sums, and sums of squares, for later statistic calculation
            if not np.isnan(s1):
                feature_data_rows.append(
                    {
                        'seq_name': seq_name,
                        'type': ft.type,
                        's0': ft.end - ft.start,  # score count (i.e. feature length in number of nucleotides)
                        's1': s1,  # score sum
                        's2': s2,  # score sum of squares
                        'subfeatures': ft.subfeatures,
                    })

    ft_df = pd.DataFrame(feature_data_rows)
    if not feature_data_rows:
        logging.info('No input data')
        return ft_df

    logging.info('Aggregating statistics')
    ft_dfg = ft_df.groupby(['seq_name', 'type'],as_index=False).agg(count=('s0','count'), mean_len=('s0','mean'),
            s0=('s0','sum'), s1=('s1','sum'), s2=('s2','sum'), subfeatures=('subfeatures','sum'))

    ft_dfg['mean_len'] = ft_dfg['mean_len'].astype(int)
    ft_dfg['score_mean'] = ft_dfg['s1'] / ft_dfg['s0']
    ft_dfg['score_std'] = ft_dfg['s2'] / ft_dfg['s0'] - ft_dfg['score_mean']**2

    ft_dfg = ft_dfg.sort_values(by='count', ascending=False).reset_index(drop=True)
    ft_dfg = ft_dfg.drop(columns=['s0', 's1', 's2'])

    return ft_dfg


def score_stats_by_kmer(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                        scorer: Callable[[str, int, int], np.array],  feature_type_filter: list[str],
                        k_values: Iterable[int] = (2,)) -> pd.DataFrame:
    """
    Do basic analysis of score statistics by k-mer, only on sequences annotated by a feature type in feature_type_filter.
    score: function that takes seqname, start and end, returning score values for specified subsequence.
    k_values list[int]: k values to analyze.
    """

    seq_records = seq_records_gen()

    kmer_data = defaultdict(lambda: np.zeros(3))  # count, sum, sum of squares
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        feature_briefs = get_feature_briefs(seq_record, feature_type_filter, merge_overlapping_features=True)
        # temp_len_list = []
        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}.', v=len(feature_briefs)//10)

            start, end = ft.start, ft.end

            ft_sequence = seq_record.seq[start: end]  # str
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
            ft_scores = scorer(seq_name, start, end)
            ft_start = start % 3
            ft_len = len(ft_sequence) % 3

            # temp_len_list.append({'len%3': len(ft_sequence) % 3, 'phase': ft.phase, 'strand': ft.strand})
            for k in k_values:

                # loop through the feature sequence and associated scores, in a window of width k, shifting by one
                # nucleotide at a time, updating the current kmer, and values with each iteration
                cur_kmer = ''
                cur_scores = []
                for ind in range(len(ft_sequence)):
                    next_nucleotide = ft_sequence[ind]
                    next_score = ft_scores[ind]

                    # skip k-mers that contain soft-masked lowercase nucleotides or undefined scores.
                    if next_nucleotide != next_nucleotide.upper() or np.isnan(next_score):
                        cur_kmer = ''
                        cur_scores = []
                        continue
                    else:
                        cur_kmer = (cur_kmer + next_nucleotide)[-k:]
                        cur_scores = (cur_scores + [next_score])[-k:]

                        if len(cur_kmer) < k:
                            continue

                    # frame = (ind - (k-1)) % 3
                    # ph_len = (ft.phase+ft_len) % 3
                    phase = ft.phase if ft.phase is not None else 0
                    # print(phase, ft.strand)

                    try:
                        # ind is the index of the last nucleotide in the kmer.
                        if ft.strand is None or ft.strand == 1:
                            frame = (ind - (k-1) - phase) % 3
                        elif ft.strand == -1:
                            frame = (len(ft_sequence) - ind - 1 - phase) % 3
                        else:
                            frame = (ind - (k - 1)) % 3
                    except:
                        print(ft.strand, phase, ft)

                    # track running count, sum, and sum of squares
                    for pos in range(k):
                        kmer_data[(k, cur_kmer, seq_name, ft.strand, phase, ft_start, ft_len, frame, pos)] += [1, cur_scores[pos], cur_scores[pos] ** 2]

    # compute overall count, mean and standard deviation
    kmer_data_agg = []
    for key, sums in kmer_data.items():
        k, cur_kmer, seq_name, strand, phase, ft_start, ft_len, frame, pos = key
        s0, s1, s2 = sums
        kmer_data_agg.append(
            {
                K_COL: k,
                KMER_COL: cur_kmer,
                SEQNAME_COL: seq_name,
                STRAND_COL: strand,
                PHASE_COL: phase,
                FTSTART_COL: ft_start,
                FTLEN_COL: ft_len,
                FRAME_COL: frame,
                POS_COL: pos,
                COUNT_COL: int(s0),
                SCORE_MEAN_COL: s1 / s0,
                SCORE_STD_COL: np.sqrt(s2 / s0 - (s1 / s0) ** 2),
            })

    # create output DataFrame
    kmer_base_df = pd.DataFrame(kmer_data_agg)

    # sort
    sortby_cols = [K_COL, KMER_COL, SEQNAME_COL, STRAND_COL, PHASE_COL, FTSTART_COL, FTLEN_COL, FRAME_COL]
    sort_ascending = [False if col==STRAND_COL else True for col in sortby_cols]
    kmer_base_df = kmer_base_df.sort_values(by=sortby_cols, ascending=sort_ascending).reset_index(drop=True)

    logging.info(f'Computed score stats by k-mer, on {len(kmer_base_df)} k-mer outputs, '
                 f'for {feature_type_filter} feature types.')

    # temp_df = pd.DataFrame(temp_len_list)
    # temp_df = temp_df.groupby(temp_df.columns.tolist(),as_index=False).size()
    return kmer_base_df  # , temp_df


def score_stats_by_dilated_kmer(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                                scorer: Callable[[str, int, int], np.array],
                                feature_type_filter: Optional[list[str]] = None,
                                k_values: Iterable[int] = (2,), dilations=range(1, 25, 4), seed=200,
                                num_chunks=10, chunk_size=10**5) -> pd.DataFrame:
    """
    Do basic analysis of score statistics by artificial gapped k-mer, with gaps between letters.
    only on sequences annotated by a feature type in feature_type_filter. If feature_type_filter not defined, then do
    analysis on num_chunks random chunks of length chunk_size each.
    score: function that takes seqname, start and end, returning score values for specified subsequence.
    k_values Iterable[int]: k values to analyze.
    """

    def update_kmer_data(kmer_data, sequence, scores, strand, phase):
        """Mutates kmer_data dictionary to add the information in the input sequence."""

        for k in k_values:
            for dilation in dilations:
                for ind in range(len(sequence) - k * dilation):
                    cur_kmer = ''.join([sequence[j] for j in range(ind, ind + k * dilation, dilation)])
                    cur_scores = scores[ind:ind + k * dilation:dilation]

                    if any(np.isnan(cur_scores)) or (set(cur_kmer) - NUCLEOTIDES):
                        continue

                    try:
                        # ind is the index of the first nucleotide in the dilated kmer.
                        if strand == 1:
                            frame = (ind - phase) % 3
                        elif ft.strand == -1:
                            frame = (len(sequence) - (ind + (k-1) * dilation) - 1 - phase) % 3
                        else:
                            frame = ind % 3
                    except:
                        print(strand, phase, ft)

                    # track running count, sum, and sum of squares
                    for pos in range(k):
                        kmer_data[(k, cur_kmer, seq_name, dilation, ft.strand, frame, pos)] += \
                            [1, cur_scores[pos], cur_scores[pos] ** 2]

                if k ==1:
                    break
        return

    seq_records = seq_records_gen()

    kmer_data = defaultdict(lambda: np.zeros(3))  # count, sum, sum of squares
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        if feature_type_filter:
            feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
            for i, ft in enumerate(feature_briefs):
                periodic_logging(i, f'Processing feature {i:,}.', v=len(feature_briefs)//10)

                ft_sequence = seq_record.seq[ft.start: ft.end]
                if USE_SOFTMASKED:
                    ft_sequence = ft_sequence.upper()

                ft_scores = scorer(seq_name, ft.start, ft.end)
                update_kmer_data(kmer_data, ft_sequence, ft_scores, ft.strand, ft.phase)
        else:
            random.seed(seed)
            starts = random.sample(range(len(seq_record)-chunk_size), num_chunks)
            starts.sort()

            # ensure chunks starting at starts do not overlap
            for i in range(1, num_chunks):
                starts[i] = max(starts[i], starts[i-1] + chunk_size)

            logging.info(f'Created {num_chunks} random chunks, with the following starts: \n' +
                         '  '.join([f'{s:,}' for s in starts]) + '.')

            for i, start in enumerate(starts):
                periodic_logging(i, f'Processing chunk {i}.', v=1)
                chunk_sequence = seq_record.seq[start: start + chunk_size]
                chunk_scores = scorer(seq_name, start, start + chunk_size)

                update_kmer_data(kmer_data, chunk_sequence, chunk_scores)

    # compute overall count, mean and standard deviation
    kmer_data_agg = []
    for key, sums in kmer_data.items():
        k, cur_kmer, seq_name, dilation, strand, frame, pos = key
        s0, s1, s2 = sums
        kmer_data_agg.append(
            {
                K_COL: k,
                KMER_COL: cur_kmer,
                SEQNAME_COL: seq_name,
                DILATION_COL: dilation,
                STRAND_COL: strand,
                FRAME_COL: frame,
                POS_COL: pos,
                COUNT_COL:  int(s0),
                SCORE_MEAN_COL: s1/s0,
                SCORE_STD_COL: np.sqrt(s2 / s0 - (s1 / s0) ** 2),
            })

    # create output DataFrame
    kmer_base_df = pd.DataFrame(kmer_data_agg)
    out_text = f'for {feature_type_filter} feature types.' if feature_type_filter else \
        f'on {num_chunks} random seqeunce chunks of size {chunk_size:,} each.'

    logging.info(f'Computed score stats by k-mer, on {len(kmer_base_df)} k-mer outputs, {out_text}')
    kmer_base_df = kmer_base_df.sort_values(by=[K_COL, KMER_COL, SEQNAME_COL, STRAND_COL, FRAME_COL, POS_COL])
    kmer_base_df = kmer_base_df.reset_index(drop=True)

    return kmer_base_df


def aggregate_over_position(df_in: pd.DataFrame):
    """
    Aggregate over position. That is, combine nucleotide statistics at each position to get a mean statistic for the
    entire k-mer. Each position, by definition, has the same count, so averaging is straightforward.
    Position does not aggregate additively, i.e. one does not add the counts when aggregating.
    """
    groupby_cols = [col for col in df_in.columns if col in ID_COLS and col != POS_COL]
    df_agg = df_in.groupby(groupby_cols, as_index=False).agg(
        **{COUNT_COL: (COUNT_COL, 'first'), SCORE_MEAN_COL: (SCORE_MEAN_COL, 'mean'),
         SCORE_STD_COL: (SCORE_STD_COL, std_to_std_of_mean)})
    return df_agg


def aggregate_over_additive_field(df_in: pd.DataFrame, additive_col: Union[str, List[str]],
                                  extra_col: Union[str, List[str]]=''):
    """
    Aggregate over an additive column field, i.e. a field where one needs to add the counts when aggregating over it.
    Each value of the field generally has different counts, so one must weigh by the counts.
    extra_col: custom column already added to the dataframe, which should be included in the aggregate.
    """

    def count_weighted_mean(s: pd.Series):
        return np.average(s, weights=df_in.loc[s.index, 'count'])

    def count_weighted_std(s: pd.Series):
        return std_to_std_of_mean(s, weights=df_in.loc[s.index, 'count'])

    if type(additive_col) == str:
        additive_col = [additive_col]

    if type(extra_col) == str:
        extra_col = [extra_col]

    groupby_cols = [col for col in df_in.columns if col in (ID_COLS+extra_col) and col not in additive_col]
    sort_ascending = [False if col==STRAND_COL else True for col in groupby_cols]

    df_agg = df_in.groupby(groupby_cols, as_index=False).agg(
            **{COUNT_COL: (COUNT_COL, 'sum'), SCORE_MEAN_COL: (SCORE_MEAN_COL, count_weighted_mean),
             SCORE_STD_COL: (SCORE_STD_COL, count_weighted_std)}).sort_values(by=groupby_cols, ascending=sort_ascending)
    return df_agg


def aggregate_over_frame(df_in: pd.DataFrame):
    """Aggregate over frame, which is an additive column field."""
    return aggregate_over_additive_field(df_in, FRAME_COL)


def aggregate_over_seq_name(df_in: pd.DataFrame):
    """Aggregate over seq_name, which is an additive column field."""
    return aggregate_over_additive_field(df_in, SEQNAME_COL)


def aggregate_over_strand(df_in: pd.DataFrame):
    """Aggregate over strand, which is an additive column field."""
    return aggregate_over_additive_field(df_in, STRAND_COL)


def aggregate_over_phase(df_in: pd.DataFrame):
    """Aggregate over phase, which is an additive column field."""
    return aggregate_over_additive_field(df_in, PHASE_COL)


def sample_extreme_score_sequences(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                                   scorer: Callable[[str, int, int], np.array],
                                   feature_type_filter: list[str], k: int = 3, padding=6, num_samples=20) -> [dict,dict]:
    """
    Return sample sequences with extreme score, along with its score, calculated mean on a k-mer.
    Padding on either side.
    """

    seq_records = seq_records_gen()

    low = high = None  # thresholds to be calculated
    low_samples, high_samples = {}, {}
    initial_scores = []  # to get distribution, and sample based on it
    for seq_record in seq_records:
        seq_name = seq_record.name

        feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}.', v=len(feature_briefs)//10)

            ft_sequence = seq_record.seq[ft.start: ft.end]  # str
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
            ft_scores = scorer(seq_name, ft.start, ft.end)

            j = 0
            while j < len(ft_scores):
                j+=1
                if len(low_samples) > num_samples and len(high_samples) > num_samples:
                    break
                kmer = ft_sequence[j: j + k]
                kmer_scores = ft_scores[j: j + k]
                if len(kmer_scores) > 0:
                    score_val = np.nanmean(kmer_scores)
                else:
                    continue

                if len(initial_scores) < 10000:
                    initial_scores.append(score_val)
                    continue
                elif low is None or high is None:
                    low, high = np.percentile(initial_scores, [0.1, 99.9])
                    logging.info(f'Defined sampling low {low:0.2f} and high {high:0.2f}.')

                if low <= score_val <= high:
                    continue

                # populate the lists
                if not np.isnan(score_val) and kmer == kmer.upper():  # ignore soft masked kmers, small letters
                    sample_seq_padded = str(seq_record.seq[ft.start + j - padding: ft.start + j + k + padding])
                    sample_scores_padded = rd(scorer(seq_name, ft.start + j - padding, ft.start + j + k + padding))
                    if score_val < low:
                        # seq_name, position, sequence, scores
                        low_samples[(seq_name, ft.start + j - padding, sample_seq_padded)] = sample_scores_padded
                    elif score_val > high:
                        high_samples[(seq_name, ft.start + j - padding, sample_seq_padded)] = sample_scores_padded
                    j+=padding

        if len(low_samples) > num_samples:
            keys = random.sample(low_samples.keys(), num_samples)
            low_samples = {key: low_samples[key] for key in keys}
        if len(high_samples) > num_samples:
            keys = random.sample(high_samples.keys(), num_samples)
            high_samples = {key: high_samples[key] for key in keys}

    return low_samples, high_samples
