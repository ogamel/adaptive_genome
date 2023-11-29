"""
Module to analyse the scores output by score_collation.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import gmean
from Bio.Seq import reverse_complement

from genetic import kmers_in_rc_order
from typing import Iterator, Callable, Iterable, Optional
from score_collation import KMER_COL, COUNT_COL, SCORE_MEAN_COL, SCORE_STD_COL, \
    K_COL, ID_COLS, POS_COL, FRAME_COL, STRAND_COL, FTLEN_COL, COMPLEMENTED_COLS, aggregate_over_additive_field, \
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
    rc_sort_ascending = [False if col in [POS_COL, STRAND_COL, FTLEN_COL] else True for col in sortby_cols]
    # rc_sort_ascending = [False if col in [POS_COL, STRAND_COL, FTLEN_COL, FRAME_COL] else True for col in sortby_cols]

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
    Find pvalues of score and counts of reverse complement kmers being equal, stratified by the ID_FIELDS present in
    the dataframe and their hardcoded relationships that we test.
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
    # rc_sort_ascending = [False if col in [POS_COL, STRAND_COL, FRAME_COL] else True for col in sortby_cols]

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
                elif inverted_col in [FRAME_COL, FTLEN_COL]:
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

    return df  #[np.isfinite(df[COUNT_DIFF])]


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

    """Mono-nucleotide probabilities"""
    df1 = df_in[(df_in.k == 1)].copy()
    df1['prob'] = df1[COUNT_COL] / df1.groupby(complemented_cols + ['dil'])[COUNT_COL].transform(sum)

    # aggregated by kmer
    df1_agg = df1[[KMER_COL, COUNT_COL]].groupby([KMER_COL]).aggregate(sum)
    df1_agg['prob'] = df1_agg[COUNT_COL] / df1_agg[COUNT_COL].sum()
    df1_agg = df1_agg.reset_index()

    # entropy contribution per mononucleotide
    df1['prob_h'] = - df1['prob']*np.log(df1['prob'])

    # mutual information normalization factor for each strand and frame value, to get I_norm from I.
    # seems unnecessary to separate them by strand and value as the factors are all very close.
    df_norm_factor = df1[complemented_cols + ['prob_h']].groupby(complemented_cols).agg(sum)

    # function to return mono-nucleotide probability, as a function of strand and frame
    def single_prob(kmer, s=None, f=None):
        if s is None or f is None:
            return df1_agg[(df1_agg[KMER_COL] == kmer)]['prob'].values[0]
        else:
            return df1[(df1[KMER_COL] == kmer) & (df1[STRAND_COL] == s) & (df1[FRAME_COL] == f)]['prob'].values[0]

    """Dilated dimer probabilities."""
    df_kmer = df_in[(df_in.k == 2) & (df_in.pos == 0)].copy()
    df_kmer['prob'] = df_kmer[COUNT_COL] / df_kmer.groupby(complemented_cols + ['dil'])[COUNT_COL].transform(sum)

    # probabilities of each base pair in the 2-mer
    df_kmer['prob_pos0'] = df_kmer.apply(
        lambda x: single_prob(x[KMER_COL][0], x[STRAND_COL], x[FRAME_COL]), axis=1)
    df_kmer['prob_pos1'] = df_kmer.apply(  # second base has a different frame, depending on dilation
        lambda x: single_prob(x[KMER_COL][1], x[STRAND_COL], (x[FRAME_COL] + x[STRAND_COL] * x['dil']) % 3), axis=1)

    # mutual information
    df_kmer['I'] = (df_kmer.prob * np.log(df_kmer.prob / (df_kmer.prob_pos0 * df_kmer.prob_pos1)))
    # simplify with mean, equivalent to not splitting by complemented_cols in the first place
    df_kmer['I_norm'] = df_kmer['I'] / df_norm_factor['prob_h'].mean()
    # Note: can define this with separate normalization factor for each strand and frame, but not much difference

    """Dilated dimer probabilities, aggregated by kmer."""
    df_kmer_agg = df_kmer[[K_COL, KMER_COL, 'dil', COUNT_COL]].groupby([K_COL, KMER_COL, 'dil']).aggregate(sum)
    df_kmer_agg['prob'] = df_kmer_agg[COUNT_COL] / df_kmer_agg.groupby(['dil'])[COUNT_COL].transform(sum)
    df_kmer_agg = df_kmer_agg.reset_index()

    # probabilities of each base pair in the 2-mer
    df_kmer_agg['prob_pos0'] = df_kmer_agg[KMER_COL].str[0].map(single_prob)
    df_kmer_agg['prob_pos1'] = df_kmer_agg[KMER_COL].str[1].map(single_prob)

    # mutual information
    df_kmer_agg['I'] = (df_kmer_agg.prob * np.log(df_kmer_agg.prob / (df_kmer_agg.prob_pos0 * df_kmer_agg.prob_pos1)))
    df_kmer_agg['I_norm'] = df_kmer_agg['I'] / (- df1_agg.prob * np.log(df1_agg.prob)).sum()

    """Summary dataframes."""
    df_summary = df_kmer[['k', 'dil'] + complemented_cols + ['I', 'I_norm']].groupby(['k', 'dil'] + complemented_cols)\
        .aggregate(sum).reset_index()
    df_summary_agg = df_kmer_agg[['k', 'dil', 'I', 'I_norm']].groupby(['k', 'dil']).aggregate(sum).reset_index()

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