"""
Statistical analysis functions for nucleotide scores, as a function of annotation features, k-mers, and other properties.
"""

import logging
import random
from typing import Iterator, Callable, Iterable, Optional, Tuple, List, Union, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.Seq import reverse_complement
from Bio.SeqRecord import SeqRecord
from Bio.Data.CodonTable import standard_dna_table
from numpy import ndarray

from genetic import get_feature_briefs, kmers_in_rc_order
from util import periodic_logging, rd, std_to_std_of_mean

ID_COLS = ['k', 'kmer', 'seq_name', 'frame', 'pos']
K_COL, KMER_COL, SEQNAME_COL, FRAME_COL, POS_COL = ID_COLS
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
            periodic_logging(i, f'Processing feature {i:,}.')
            scores = scorer(seq_name, ft.start, ft.end)
            s1 = sum(scores)
            s2 = sum([s**2 for s in scores])

            # accumulate counts, sums, and sums of squares, for later statistic calculation
            if not np.isnan(s1):
                feature_data_rows.append(
                    {
                        'seq_name': seq_name,
                        'type': ft.type,
                        's0': ft.end - ft.start + 1,  # score count (i.e. feature length in number of nucleotides)
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

        feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}.')

            ft_sequence = seq_record.seq[ft.start: ft.end + 1]  # str
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
            ft_scores = scorer(seq_name, ft.start, ft.end + 1)

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

                    frame = ind % k
                    # track running count, sum, and sum of squares
                    for pos in range(k):
                        kmer_data[(k, cur_kmer, seq_name, frame, pos)] += [1, cur_scores[pos], cur_scores[pos] ** 2]

    # compute overall count, mean and standard deviation
    kmer_data_agg = []
    for key, sums in kmer_data.items():
        k, cur_kmer, seq_name, frame, pos = key
        s0, s1, s2 = sums
        kmer_data_agg.append(
            {
                K_COL: k,
                KMER_COL: cur_kmer,
                SEQNAME_COL: seq_name,
                FRAME_COL: frame,
                POS_COL: pos,
                COUNT_COL: int(s0),
                SCORE_MEAN_COL: s1 / s0,
                SCORE_STD_COL: np.sqrt(s2 / s0 - (s1 / s0) ** 2),
            })

    # create output DataFrame
    kmer_base_df = pd.DataFrame(kmer_data_agg)

    logging.info(f'Computed score stats by k-mer, on {len(kmer_base_df)} k-mer outputs, '
                 f'for {feature_type_filter} feature types.')

    return kmer_base_df


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

    def update_kmer_data(kmer_data, sequence, scores):
        """Mutates kmer_data dictionary to add the information the input sequence."""
        for k in k_values:
            for dilation in dilations:
                for ind in range(len(sequence) - k * dilation):
                    cur_kmer = ''.join([sequence[j] for j in range(ind, ind + k * dilation, dilation)])
                    cur_scores = scores[ind:ind + k * dilation:dilation]

                    if any(np.isnan(cur_scores)) or (set(cur_kmer) - NUCLEOTIDES):
                        continue

                    # track running count, sum, and sum of squares
                    for pos in range(k):
                        kmer_data[(k, cur_kmer, seq_name, dilation, pos)] += [1, cur_scores[pos], cur_scores[pos] ** 2]
        return

    seq_records = seq_records_gen()

    kmer_data = defaultdict(lambda: np.zeros(3))  # count, sum, sum of squares
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        if feature_type_filter:
            feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
            for i, ft in enumerate(feature_briefs):
                periodic_logging(i, f'Processing feature {i:,}.', v=len(feature_briefs)//25)

                ft_sequence = seq_record.seq[ft.start: ft.end + 1]
                if USE_SOFTMASKED:
                    ft_sequence = ft_sequence.upper()
                ft_scores = scorer(seq_name, ft.start, ft.end + 1)

                update_kmer_data(kmer_data, ft_sequence, ft_scores)
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
        k, cur_kmer, seq_name, dilation, pos = key
        s0, s1, s2 = sums
        kmer_data_agg.append(
            {
                K_COL: k,
                KMER_COL: cur_kmer,
                SEQNAME_COL: seq_name,
                DILATION_COL: dilation,
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

    print(kmer_base_df.to_string())
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


def aggregate_over_additive_field(df_in: pd.DataFrame, additive_col: str):
    """
    Aggregate over an additive column field, i.e. a field where one needs to add the counts when aggregating over it.
    Each value of the field generally has different counts, so one must weigh by the counts.
    """

    def count_weighted_mean(s: pd.Series):
        return np.average(s, weights=df_in.loc[s.index, 'count'])

    def count_weighted_std(s: pd.Series):
        return std_to_std_of_mean(s, weights=df_in.loc[s.index, 'count'])

    groupby_cols = [col for col in df_in.columns if col in ID_COLS and col != additive_col]
    df_agg = df_in.groupby(groupby_cols, as_index=False).agg(
            **{COUNT_COL: (COUNT_COL, 'sum'), SCORE_MEAN_COL: (SCORE_MEAN_COL, count_weighted_mean),
             SCORE_STD_COL: (SCORE_STD_COL, count_weighted_std)})
    return df_agg


def aggregate_over_frame(df_in: pd.DataFrame):
    """Aggregate over frame, which is an additive column field."""
    return aggregate_over_additive_field(df_in, FRAME_COL)


def aggregate_over_seq_name(df_in: pd.DataFrame):
    """Aggregate over seq_name, which is an additive column field."""
    return aggregate_over_additive_field(df_in, SEQNAME_COL)

# TODO: based on dilated stats, measure normalized mutual information (averaged over all base combinations) vs dilation
#  ... which will provide length scale of nonlocality of k-mers ... degree of dependence between bases
# https://math.stackexchange.com/questions/3553704/how-to-measure-the-independence-of-random-variables
# def mutual_information(df_in: pd.DataFrame):
#     pass

def corrcoefs_by_score_count(df_in: pd.DataFrame, kmer_col: str = KMER_COL, count_col: str = COUNT_COL,
                             score_col: str = SCORE_MEAN_COL, k_values: list[int] = None):
    """
    Find correlations of score and counts, as well as pairwise correlations of each of these fields between palindrome
    kmer pairs.
    Input df is output from score_stats_by_kmer, possibly subsequently further aggregated.
    """

    # Note: the idea of pairwise correlation has a flaw, in that it is a bit sensitive to the ordering within each
    # pair. We arbitrarily assign the alphabetically earlier member of a palindrome pair member to the first set,
    # and the later to the second. We then take the correlation across the two sets.
    # Nonetheless, the idea is to measure the differences within pairs, and show they are very small relative to
    # the variation of the whole set. Pairwise correlation achieves this.

    if k_values is None:
        k_values = df_in[K_COL].unique()

    sortby_cols = [col for col in df_in.columns if col in ID_COLS]
    df = df_in.sort_values(by=sortby_cols)

    # list to facilitate inverse position sorting in the reverse complement dataframe,
    # e.g. to measure correlation first position in kmer with last in kmer_rc, etc.
    rc_sort_ascending = [False if col == POS_COL else True for col in sortby_cols]

    for k in k_values:
        df_k = df[df[K_COL] == k]

        # start with simplest correlation
        corr_count_score = np.corrcoef(df_k[count_col], df_k[score_col])[0][1]
        print(f'k = {k}, correlation between k-mer count and score {corr_count_score:0.3f}')

        # initiate lists for pairwise values, for a kmer and its reverse complement
        rc_pairwise_counts = [[], []]
        rc_pairwise_scores = [[], []]

        kmers_set = set(kmers_in_rc_order(k))
        while kmers_set:
            kmer = kmers_set.pop()
            kmer_rc = reverse_complement(kmer)

            # skip palindromes, as correlating the same quantity with itself will artificially raise correlation
            # TODO: there should be an exception for scores (but not counts) if POS_COL in sortby_cols, since measuring
            #  score correlation between different positions of a palindrome is valid, but perhaps only half of each
            #  palindrome to avoid duplication?
            #  Similarly, if k is odd, consider excluding the middle position since then we are measuring correlation
            #  with itself.
            if kmer == kmer_rc:
                continue
            kmers_set.remove(kmer_rc)

            if kmer > kmer_rc:  # alphabetically order kmer and its reverse complement
                kmer, kmer_rc = kmer_rc, kmer

            df_kmer = df_k[df_k[KMER_COL] == kmer]
            df_kmer_rc = df_k[df_k[KMER_COL] == kmer_rc]

            # invert position sorting in reverse complement, if position column exists,
            df_kmer_rc = df_kmer_rc.sort_values(by=sortby_cols, ascending=rc_sort_ascending)

            rc_pairwise_counts[0].extend(df_kmer[count_col])
            rc_pairwise_counts[1].extend(df_kmer_rc[count_col])
            rc_pairwise_scores[0].extend(df_kmer[score_col])
            rc_pairwise_scores[1].extend(df_kmer_rc[score_col])

        corr_pairwise_count = np.corrcoef(*rc_pairwise_counts)[0][1]
        print(f'k = {k}, pairwise reverse complement correlation between k-mer counts {corr_pairwise_count:0.3f}')

        corr_pairwise_score = np.corrcoef(*rc_pairwise_scores)[0][1]
        print(f'k = {k}, pairwise reverse complement correlation between k-mer scores {corr_pairwise_score:0.3f}')

    return  # corr_count_score, corr_pairwise_count, corr_pairwise_score


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
            periodic_logging(i, f'Processing feature {i:,}.')

            ft_sequence = seq_record.seq[ft.start: ft.end + 1]  # str
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
            ft_scores = scorer(seq_name, ft.start, ft.end + 1)

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
