"""
Visual aids and plots, usualyl applied to outputs of analysis functions.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, colormaps

from genetic import kmers_in_rc_order
from score_analysis import K_COL, KMER_COL, COUNT_COL, SCORE_MEAN_COL, POS_COL

warnings.simplefilter(action='ignore', category=FutureWarning)

from data.paths import FIGS_PATH

# FIG_SIZE = (8, 6)
FIG_SIZE = (12, 9)

blues_cmap = colormaps['Blues']
greens_cmap = colormaps['Greens']


def stat_by_kmer_plot(df: pd.DataFrame, kmer_col: str = KMER_COL, count_col: str = COUNT_COL,
                      score_col: str = SCORE_MEAN_COL, splitby_col: str = None, k_values: list[int] = None,
                      title_prefix='', fig_size=FIG_SIZE, figs_path=FIGS_PATH):
    """
    Plot statistical distributions of score and counts by k-mer.
    Input df is output from score_stats_by_kmer, possibly subsequently further aggregated.
    x-axis kmers placing reverse complements adjacent, starting with palindromes.
    split_col: to further subdivide result within each k-mer, e.g. seq_name, pos, or frame
    """

    if k_values is None:
        k_values = df[K_COL].unique()

    for k in k_values:
        # sort such that reverse complement k-mers are adjacent
        kmers = kmers_in_rc_order(k)
        df_k = df[df[kmer_col].isin(kmers)]
        df_k = df_k.sort_values(by=kmer_col, key=lambda column: column.map(lambda e: kmers.index(e)))

        plt.figure(figsize=fig_size)
        title = f'{title_prefix} Stats by K-mer, k={k}'
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        ax0.set_ylabel(score_col)

        ax1 = plt.subplot(gs[1])
        ax1.set_ylabel(count_col)

        if splitby_col is None:
            ax0.bar(df_k[kmer_col], df_k[score_col], color='g')
            ax1.bar(df_k[kmer_col], df_k[count_col], color='b')
        else:
            title += f', split by {splitby_col}'

            sb_vals = sorted(df_k[splitby_col].unique())
            x = np.arange(len(df_k[kmer_col].unique()))
            n = len(sb_vals)
            w = 0.8/n
            for i, sb_val in enumerate(sb_vals):
                label = f'{splitby_col}={sb_val}'
                df_k_sb = df_k[df_k[splitby_col] == sb_val]

                ax0.bar(x + i * w, df_k_sb[score_col], width=w, label=label, color=greens_cmap((i+1)/n))

                # count doesn't split by position, so make bars the same color to look like one
                count_color = 'b' if splitby_col == POS_COL else blues_cmap((i+1)/n)
                ax1.bar(x + i*w, df_k_sb[count_col], width=w, color=count_color)

            ax1.set_xticks(x+w*(n-1)/2, df_k[kmer_col].unique())
            # ax0.legend(loc='upper left', ncols=n)

        ax0.title.set_text(title)
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)

        if k >= 3:
            plt.xticks(fontsize=7, rotation=90)

        plt.savefig(f'{figs_path}/kmer_score_k={k}')
    return
