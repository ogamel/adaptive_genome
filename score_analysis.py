"""
Module to analyse the scores output by score_collation.py.
"""

import numpy as np
import pandas as pd
from Bio.Seq import reverse_complement

from genetic import kmers_in_rc_order
from score_collation import KMER_COL, COUNT_COL, SCORE_MEAN_COL, K_COL, ID_COLS, POS_COL

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


# TODO: based on dilated stats, measure normalized mutual information (averaged over all base combinations) vs dilation
#  ... which will provide length scale of nonlocality of k-mers ... degree of dependence between bases
# TODO: check before the sum, in case it is high for particular combinations... ...
# check for different sequence types
# https://math.stackexchange.com/questions/3553704/how-to-measure-the-independence-of-random-variables
# https://en.wikipedia.org/wiki/Pointwise_mutual_information
# https://pdfs.semanticscholar.org/7914/8020ef42aff36f0649bccc94c9711f9b884f.pdf
def mutual_information(df_in: pd.DataFrame):
    # P(a)P(b) and P(a,b) for a given dilation
    # assume input has single probabilities, i.e. k=1 and k=2

    # for count

    # get mono-nucleotide probabilities
    df1 = df_in[(df_in.k==1)].copy()
    df1['prob'] = df1['count'] / df1['count'].sum()
    single_prob = dict(zip(df1.kmer, df1.prob))

    # get dilated base pair probabilities
    df_out = pd.DataFrame(columns=['dil', 'I', 'I_norm'])
    for dil in df_in.dil.unique():
        df2 = df_in[(df_in.dil==dil) & (df_in.k==2) & (df_in.pos==0)].copy()
        df2['prob'] = df2['count'] / df2['count'].sum()
        df2['prob_pos0'] = df2.kmer.str[0].map(single_prob)
        df2['prob_pos1'] = df2.kmer.str[1].map(single_prob)

        # compute normalized mutual information
        I = (df2.prob*np.log(df2.prob/(df2.prob_pos0 * df2.prob_pos1))).sum()
        I_norm = I / (- df1.prob * np.log(df1.prob)).sum()

        df_out.loc[len(df_out)] = [dil, I, I_norm]

    df_out.dil = df_out.dil.astype(int)

    return df_out






# TODO: compute mutual information of k-mer vs its subwords ... e.g. P(ACC) vs P(A)P(C)P(C) vs P(AC)P(C) vs P(A)P(CC)
# seems this needs deep thought and defining new quantities to truly understand it
