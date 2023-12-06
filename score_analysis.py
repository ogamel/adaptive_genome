"""
Module to analyse the scores output by score_collation.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import gmean
from Bio.Seq import reverse_complement

from genetic import kmers_in_rc_order
from typing import Iterator, Callable, Iterable, Optional
from score_collation import KMER_COL, COUNT_COL, SCORE_MEAN_COL, SCORE_STD_COL, DILATION_COL, \
    K_COL, ID_COLS, POS_COL, FRAME_COL, STRAND_COL, COMPLEMENTED_COLS, aggregate_over_additive_field, \
    aggregate_over_position


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

    # list to facilitate inverse position and strand sorting in the reverse complement dataframe,
    # e.g. to measure correlation first position in kmer with last in kmer_rc, etc.
    rc_sort_ascending = [False if col in [POS_COL, STRAND_COL] else True for col in sortby_cols]

    print(f'Correlations with columns {sortby_cols}')

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
        corr_pairwise_score = np.corrcoef(*rc_pairwise_scores)[0][1]

        print(f'\tpairwise reverse complement correlation between '
              f'\n\t\tk-mer counts {corr_pairwise_count:0.3f},'
              f'\n\t\tk-mer scores {corr_pairwise_score:0.3f}.')

    return  # corr_count_score, corr_pairwise_count, corr_pairwise_score


def diff_stats_by_score_count(df_in: pd.DataFrame, kmer_col: str = KMER_COL, count_col: str = COUNT_COL,
                              score_col: str = SCORE_MEAN_COL, k_values: list[int] = None):
    """
    Find normalized mean square difference of score and counts between kmers and their reverse complement for the
    discovered complement relationship, as well as for one where each feature column (e.g. strand, frame) is inverted.
    This tests the validity of the discovered complement relationship, as well as the necessity of each feature column
    for this to hold.
    Input df is output from score_stats_by_kmer, possibly subsequently further aggregated.
    """

    COUNT_DIFF = 'ct_d'
    SCORE_DIFF = 'sc_d'

    if k_values is None:
        k_values = df_in[K_COL].unique()

    sortby_cols = [col for col in df_in.columns if col in ID_COLS]
    complemented_cols = [col for col in df_in.columns if col in COMPLEMENTED_COLS]

    df = df_in.sort_values(by=sortby_cols)
    df[COUNT_DIFF] = pd.Series(dtype='Int64')
    df[SCORE_DIFF] = pd.Series(dtype='float')
    for inverted_col in complemented_cols:
        df[COUNT_DIFF + '_' + inverted_col] = pd.Series(dtype='Int64')
        df[SCORE_DIFF + '_' + inverted_col] = pd.Series(dtype='float')

    # list to facilitate inverse position sorting in the reverse complement dataframe,
    # e.g. to measure correlation first position in kmer with last in kmer_rc, etc.
    rc_sort_ascending = [False if col in [POS_COL, STRAND_COL] else True for col in sortby_cols]

    for k in k_values:
        df_k = df[df[K_COL] == k]

        kmers_set = set(kmers_in_rc_order(k))
        while kmers_set:
            kmer = kmers_set.pop()
            kmer_rc = reverse_complement(kmer)

            # skip palindromes
            if kmer == kmer_rc:
                continue
            kmers_set.remove(kmer_rc)

            if kmer > kmer_rc:  # alphabetically order kmer and its reverse complement
                kmer, kmer_rc = kmer_rc, kmer

            df_kmer = df_k[df_k[KMER_COL] == kmer]
            df_kmer_rc = df_k[df_k[KMER_COL] == kmer_rc]

            # invert position sorting in reverse complement, if position column exists,
            df_kmer_rc = df_kmer_rc.sort_values(by=sortby_cols, ascending=rc_sort_ascending)

            df_kmer[COUNT_DIFF] = df_kmer[count_col]-df_kmer_rc[count_col].values
            df_kmer[SCORE_DIFF] = df_kmer[score_col]-df_kmer_rc[score_col].values
            df_k.update(df_kmer[[COUNT_DIFF, SCORE_DIFF]])
            df.update(df_k[[COUNT_DIFF, SCORE_DIFF]])

            # Need a ratio of difference between diff between one and its complement, to that to its "near complement"
            # where a given feature is replaced by a non-complement value. This is meant to show the relevance of this
            # feature.
            for inverted_col in complemented_cols:
                df_kmer_rc_cp = df_kmer_rc.copy()
                if inverted_col == POS_COL:
                    # shuffle position in RC, from complement position to +1 (arbitrarily)
                    df_kmer_rc_cp[inverted_col] = (df_kmer_rc_cp[inverted_col] + 1) % k
                elif inverted_col == FRAME_COL:
                    # shuffle frame in RC, from complement frame to +1 (arbitrarily)
                    df_kmer_rc_cp[inverted_col] = (df_kmer_rc_cp[inverted_col] + 1) % 3
                elif inverted_col == STRAND_COL:
                    df_kmer_rc_cp[inverted_col] = - df_kmer_rc_cp[inverted_col]
                else:
                    # make RC column same as original
                    df_kmer_rc_cp[inverted_col] = df_kmer[inverted_col].values

                # resort after shuffling
                df_kmer_rc_cp = df_kmer_rc_cp.sort_values(by=sortby_cols, ascending=rc_sort_ascending)

                df_kmer[COUNT_DIFF + '_' + inverted_col] = df_kmer[count_col] - df_kmer_rc_cp[count_col].values
                df_kmer[SCORE_DIFF + '_' + inverted_col] = df_kmer[score_col] - df_kmer_rc_cp[score_col].values
                df_k.update(df_kmer[[COUNT_DIFF + '_' + inverted_col, SCORE_DIFF + '_' + inverted_col]])
                df.update(df_k[[COUNT_DIFF + '_' + inverted_col, SCORE_DIFF + '_' + inverted_col]])

        count_std_k = np.std(df_k[count_col])
        score_std_k = np.std(df_k[score_col])

        # equivalent to mean square difference / 2, where difference is measure in units of std. dev.
        print(f'\nk = {k}. Root mean square difference (normalized by twice the overall std. dev.) between pairs '
              f'complementary in columns:'
              f'\n\t\t{", ".join(complemented_cols)}'
              f'\n\t\t\tCounts: {np.sqrt((df_k[COUNT_DIFF]**2).mean())/(2 * count_std_k):.4f}. '
              f'Score: {np.sqrt((df_k[SCORE_DIFF]**2).mean())/(2 * score_std_k):.4f}.')


        for inverted_col in complemented_cols:
            print(f'\t\t above except for {inverted_col}'
                  f'\n\t\t\tCounts: {np.sqrt((df_k[COUNT_DIFF + "_" + inverted_col]**2).mean())/(2 * count_std_k):.4f}. '
                  f'Score: {np.sqrt((df_k[SCORE_DIFF + "_" + inverted_col]**2).mean())/(2 * score_std_k):.4f}.')

    return df


def mutual_information_by_dilation(df_in: pd.DataFrame, do_triple:bool = False) -> pd.DataFrame:
    """
    Computes mutual information vs dilation. Uses normalized counts of a k-mer among other k-mers with same k as its
    probability.
    Input df_in is generated by score_stats_by_dilated_kmer, must have k=1 and k=2.
    do_triple is a flag specifying whether to compute
    Output df_summary is dataframe of raw and normalized mutual information organized by k and dil (dilation).

    References on mutual information (two-way)
    https://math.stackexchange.com/questions/3553704/how-to-measure-the-independence-of-random-variables
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    https://pdfs.semanticscholar.org/7914/8020ef42aff36f0649bccc94c9711f9b884f.pdf

    References on mutual information (three-way)
    # https: // math.stackexchange.com / questions / 943107 / what - is -the - mutual - information - of - three - variables
    # https://en.wikipedia.org/wiki/Interaction_information
    """
    # TODO: check for each kmer before the sum, in case it is high for particular combinations... ...
    # TODO: right now doing just counts extend to score somehow

    # columns we will group the probabilities by - usually strand and frame
    complemented_cols = [col for col in COMPLEMENTED_COLS if col in df_in.columns and col != POS_COL]
    PAIR_PROB_COL = 'pair_prob'
    SINGLE_PROB_COL = 'sgl_prob'
    SINGLE_SCORE_COL = 'sgl_score'

    """Mono-nucleotide probabilities"""
    df1 = df_in[(df_in.k == 1)].copy()
    df1[SINGLE_PROB_COL] = df1[COUNT_COL] / df1.groupby(complemented_cols + [DILATION_COL])[COUNT_COL].transform(sum)

    # aggregated by kmer
    df1_agg = aggregate_over_additive_field(df1, complemented_cols)
    # df1_agg = df1[[KMER_COL, COUNT_COL, SCORE_MEAN_COL,]].groupby([KMER_COL]).aggregate(sum)
    df1_agg[SINGLE_PROB_COL] = df1_agg[COUNT_COL] / df1_agg[COUNT_COL].sum()
    df1_agg = df1_agg.reset_index()

    # entropy contribution per mononucleotide
    df1['prob_h'] = - df1[SINGLE_PROB_COL]*np.log(df1[SINGLE_PROB_COL])

    # normalization factors
    # # mutual information normalization factor for each strand and frame value, to get I_norm from I.
    # # seems unnecessary to separate them by strand and value as the factors are all very close.
    # df_norm_factor = df1[complemented_cols + ['prob_h']].groupby(complemented_cols).agg(sum)
    I_norm_factor = (- df1_agg[SINGLE_PROB_COL] * np.log(df1_agg[SINGLE_PROB_COL])).sum()
    score_rms_norm_factor = df1[SCORE_MEAN_COL].std()

    # function to return column value (e.g. probability) of a mono-nucleotide, as a function of strand and frame
    def _single_col(kmer, col, s=None, f=None):
        if s is None or f is None:
            return df1_agg[(df1_agg[KMER_COL] == kmer)][col].values[0]
        else:
            return df1[(df1[KMER_COL] == kmer) & (df1[STRAND_COL] == s) & (df1[FRAME_COL] == f)][col].values[0]

    def single_prob(kmer, s=None, f=None):
        return _single_col(kmer, SINGLE_PROB_COL, s, f)

    def single_score(kmer, s=None, f=None):
        return _single_col(kmer, SCORE_MEAN_COL, s, f)

    """Dilated dimer probabilities."""
    df_kmer = df_in[(df_in.k == 2)].reset_index(drop=True).copy()
    # pair probability saved in the pos == 0 row
    df_kmer[PAIR_PROB_COL] = df_kmer[COUNT_COL] / df_kmer[df_kmer.pos == 0].groupby(complemented_cols +
                                                                              [DILATION_COL])[COUNT_COL].transform(sum)

    # add dilation to the frame if (pos==1 and strand==1) or (pos==0 and strand==-1). i.e. 2*pos - strand == 1
    df_kmer[SINGLE_PROB_COL] = df_kmer.apply(lambda x: single_prob(x[KMER_COL][x[POS_COL]], x[STRAND_COL],
                                (x[FRAME_COL] + ((2 * x[POS_COL] - x[STRAND_COL]) == 1) * x[DILATION_COL]) % 3), axis=1)
    df_kmer[SINGLE_SCORE_COL] = df_kmer.apply(lambda x: single_score(x[KMER_COL][x[POS_COL]], x[STRAND_COL],
                                (x[FRAME_COL] + ((2 * x[POS_COL] - x[STRAND_COL]) == 1) * x[DILATION_COL]) % 3), axis=1)

    # mutual information
    df_kmer['I'] = (df_kmer[PAIR_PROB_COL] * np.log(df_kmer[PAIR_PROB_COL] /
                    (df_kmer[df_kmer.pos == 0][SINGLE_PROB_COL] * df_kmer[df_kmer.pos == 1][SINGLE_PROB_COL].values)))
    # simplify with mean, equivalent to not splitting by complemented_cols in the first place
    df_kmer['I_norm'] = df_kmer['I'] / I_norm_factor
    # Note: can define this with separate normalization factor for each strand and frame, but not much difference

    # mean square score diff
    df_kmer['score_msd'] = (df_kmer[SCORE_MEAN_COL] - df_kmer[SINGLE_SCORE_COL]) ** 2

    """Dilated dimer probabilities, aggregated over strand and frame, by kmer."""
    df_kmer_agg = aggregate_over_additive_field(df_kmer, [STRAND_COL, FRAME_COL], extra_col='dil')
    df_kmer_agg[PAIR_PROB_COL] = df_kmer_agg[COUNT_COL] / df_kmer_agg[df_kmer_agg.pos == 0].groupby(
                                                                              [DILATION_COL])[COUNT_COL].transform(sum)
    df_kmer_agg[SINGLE_PROB_COL] = df_kmer_agg.apply(lambda x: single_prob(x[KMER_COL][x[POS_COL]]), axis=1)
    df_kmer_agg[SINGLE_SCORE_COL] = df_kmer_agg.apply(lambda x: single_score(x[KMER_COL][x[POS_COL]]), axis=1)

    # mutual information
    df_kmer_agg['I'] = (df_kmer_agg[PAIR_PROB_COL] * np.log(df_kmer_agg[PAIR_PROB_COL] /
                            (df_kmer_agg[df_kmer_agg.pos == 0][SINGLE_PROB_COL] *
                             df_kmer_agg[df_kmer_agg.pos == 1][SINGLE_PROB_COL].values)))
    df_kmer_agg['I_norm'] = df_kmer_agg['I'] / I_norm_factor

    # mean square score diff
    df_kmer_agg['score_msd'] = (df_kmer_agg[SCORE_MEAN_COL] - df_kmer_agg[SINGLE_SCORE_COL]) ** 2

    """Summary dataframes."""
    df_summary = df_kmer[['k', DILATION_COL] + complemented_cols + ['I', 'I_norm', 'score_msd']].groupby(['k',
        DILATION_COL] + complemented_cols).aggregate({'I': 'sum', 'I_norm': 'sum', 'score_msd': 'mean'}).reset_index()
    # normalize by std of monomer scores
    df_summary['score_rms'] = np.sqrt(df_summary['score_msd']) / score_rms_norm_factor
    df_summary = df_summary.drop(columns='score_msd')

    df_summary_agg = df_kmer_agg[['k', DILATION_COL, 'I', 'I_norm', 'score_msd']].groupby(['k', DILATION_COL])\
        .aggregate({'I': 'sum', 'I_norm': 'sum', 'score_msd': 'mean'}).reset_index()
    # normalize by std of monomer scores
    df_summary_agg['score_rms'] = np.sqrt(df_summary_agg['score_msd']) / score_rms_norm_factor
    df_summary_agg = df_summary_agg.drop(columns='score_msd')

    # add aggregated data to the same dataframe, with empty strand and frame columns.
    df_summary = df_summary.merge(df_summary_agg, how='outer')

    df_summary[STRAND_COL] = df_summary[STRAND_COL].astype("Int64")
    df_summary[FRAME_COL] = df_summary[FRAME_COL].astype("Int64")
    return df_summary


# TODO: compute mutual information of k-mer vs its subwords ... e.g. P(ACC) vs P(A)P(C)P(C) vs P(AC)P(C) vs P(A)P(CC)
# seems this needs deep thought and defining new quantities to truly understand it
# compare only with the level below it.

# TODO: compute mutual information on codon level, of a codon being followed by others (64 x 64) ... or by Amino Acid
#  level

# TODO: check amino acid network ... i.e. reverse complement leads to another AA? but that is on opposite strand.
#  May be look at sum of frequency of all codons that make same AA. actually, does RC imply anything about relative
#  frequency of AA? I don't think anything simple