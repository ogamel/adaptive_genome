"""
Statistical analysis functions for nucleotide scores, as a function of annotation features, k-mers, and other properties.
"""

import logging
import random
from typing import Iterator, Callable

import numpy as np
import pandas as pd
from Bio.Seq import reverse_complement
from Bio.SeqRecord import SeqRecord

from genetic import get_feature_briefs, kmers_in_rc_order
from util import periodic_logging, rd, std_to_std_of_mean

ID_COLS = ['k', 'kmer', 'seq_name', 'frame', 'pos']
K_COL, KMER_COL, SEQNAME_COL, FRAME_COL, POS_COL = ID_COLS
VALUE_COLS = ['count', 'score_mean', 'score_std']
COUNT_COL, SCORE_MEAN_COL, SCORE_STD_COL = VALUE_COLS

USE_SOFTMASKED = True  # flag to whether to use softmasked nucleotides, in small letters e.g. 'gatc'


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
                        scorer: Callable[[str, int, int], list[float]],  feature_type_filter: list[str],
                        k_values: list[int] = None) -> pd.DataFrame:
    """
    Do basic analysis of score statistics by k-mer, only on sequences annotated by a feature type in feature_type_filter.
    score: function that takes seqname, start and end, returning score values for specified subsequence.
    k_values list[int]: k values to analyze. Default is all k in the dataframe.
    """

    seq_records = seq_records_gen()

    if k_values is None:
        k_values = [2]  # default value

    kmer_data_rows = []
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}.')

            ft_sequence = str(seq_record.seq[ft.start: ft.end + 1])
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
                    for pos in range(k):
                        kmer_data_rows.append(
                            {
                                K_COL: k,
                                KMER_COL: cur_kmer,
                                SEQNAME_COL: seq_name,
                                FRAME_COL: frame,
                                POS_COL: pos,
                                'score': cur_scores[pos],
                            })

    kmer_df = pd.DataFrame(kmer_data_rows)

    # compute score stats (mean, variance, count) per kmer
    kmer_base_df = kmer_df.groupby(ID_COLS, as_index=False).agg(
        **{COUNT_COL: ('score', 'count'), SCORE_MEAN_COL: ('score', 'mean'), SCORE_STD_COL: ('score', 'std')})

    logging.info(f'Computed score stats by k-mer, on {len(kmer_base_df)} k-mers, for {feature_type_filter} feature types.')

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


def sample_sequence_by_score(seq_records_gen: Callable[[], Iterator[SeqRecord]], feature_type_filter: list[str],
                            score: Callable[[str, int, int], list[float]], k: int = 3) -> list[list[str, list[float]]]:
    """Return sample sequences with high variation, along with its score, calculated mean on a k-mer."""
    # TODO: reimplement sample sequence function

    seq_records = seq_records_gen()

    samples = []
    iter_ct, feature_ct = 0, 0
    for seq_record in seq_records:
        seq_name = seq_record.name

        feature_stack = seq_record.features.copy()
        while feature_stack:
            periodic_logging(iter_ct, f'{iter_ct} iterations, {feature_ct} features with accepted types, {len(samples)} samples.')
            iter_ct += 1

            feature = feature_stack.pop()
            feature_stack.extend(feature.sub_features)

            if feature.type in feature_type_filter:
                feature_ct += 1
                # loop through kmers in feature, find average score over all k positions for each kmer.
                s, e = feature.location.start.position, feature.location.end.position

                for i in range(s, e):
                    kmer = str(seq_record.seq[i: i + k])
                    score_val = np.nanmean(score(seq_name, i, i + k))
                    if not np.isnan(score_val) and kmer == kmer.upper():  # ignore soft masked kmers, small letters

                        # probabilistically sample the sequence and its score when very variable
                        if random.random() < 0.0001 and score_val < 0.2 :
                            samples.append([str(seq_record.seq[i-9: i+12]),list(rd(score(seq_name, i-9, i+12)))])

    return samples