"""
Experimental research script to find patterns correlations between genomic variability / conservation and sequence
or annotation information.
Conservation score is measured by Genomic Evolutionary Rate Profiling (GERP).
All data obtained from Ensembl release 109.
"""
import logging
from importlib import reload
import visuals
visuals=reload(visuals)
import score_analysis
import score_collation
score_analysis = reload(score_analysis)
score_collation = reload(score_collation)
import pandas as pd

from data.load import read_sequence, read_annotation_generator, read_gerp_scorer
from data.paths import chr17_paths  # paths to source data files
from data.process import get_train_test_x_y
from score_collation import score_stats_by_kmer, score_stats_by_dilated_kmer, sample_extreme_score_sequences, aggregate_over_additive_field, score_stats_by_feature_type, aggregate_over_position
from score_nn_modeling import LocalWindowModel, ModelTrainer
from score_analysis import mutual_information_by_dilation, corrcoefs_by_score_count, diff_stats_by_score_count
from visuals import plot_mutual_information
from genetic import get_feature_briefs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.precision', 3)
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
    # start the analysis with human chromosome 17
    paths = chr17_paths

    # get the raw sequence dictionary, from a FASTA file
    seq_dict = read_sequence(paths.sequence)

    # # (optional) examine annotations, from the annotation GFF file
    # examine_annotation(paths.annotation)

    # get the annotated sequence generator function, from the annotation GFF file
    seq_records_gen = read_annotation_generator(paths.annotation, seq_dict=seq_dict)

    # get GERP retrieval function, from the BigWig file
    gerp_scorer = read_gerp_scorer(paths.gerp)

    df= score_stats_by_feature_type(seq_records_gen, gerp_scorer)

    """Analysis"""
    # # analyze by kmer for CDS (coding sequence) features
    # kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['CDS'], k_values=[1, 2, 3, 4, 5])
    # # kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['lnc_RNA'], k_values=[1,2,3])
    # # kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['five_prime_UTR'], k_values=[1,2,3])
    # dff = aggregate_over_additive_field(kmer_base_df, ['ft_len', 'phase','ft_start'])
    #
    # _ = corrcoefs_by_score_count(dff)
    # _ = diff_stats_by_score_count(dff)

    # analyze by dilated kmer for CDS (coding sequence) features
    kmer_base_df_cds = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, ['CDS'], k_values=(1, 2,), dilations=range(1,20))
    # df_out_cds, df2 = mutual_information_by_dilation(kmer_base_df_cds)
    df_summary = mutual_information_by_dilation(kmer_base_df_cds)
    plot_mutual_information(df_summary, title_prefix='cds_grp')

    # # analyze by dilated kmer for genes features
    # kmer_base_df_gene = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, ['gene'], k_values=(1, 2,), dilations=range(1,20))
    # df_out_gene = mutual_information_by_dilation(kmer_base_df_gene)
    # plot_mutual_information(df_out_gene, title_prefix='gene')
    #
    # # analyze by dilated kmer for samples from whole chromosome
    # kmer_base_df_whole = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, k_values=(1, 2,), dilations=range(1,20))
    # df_out_whole = mutual_information_by_dilation(kmer_base_df_whole)
    # plot_mutual_information(df_out_whole, title_prefix='whole_genome')

    # get extreme samples
    # low_samples, high_samples = sample_extreme_score_sequences(seq_records_gen, gerp_scorer, ['CDS'])

    # """Modeling (neural network in JAX)"""
    # x_train, y_train, x_test, y_test = get_train_test_x_y(seq_records_gen, gerp_scorer, ['CDS'])
    #
    # model = LocalWindowModel()
    #
    # model_trainer = ModelTrainer(model, epochs=1000)
    # model_trainer.train(x_train, y_train)
    #
    # loss_test = model_trainer.model.loss(x_test, y_test)
    # logging.info(f'Test loss {loss_test:.3f}.')
