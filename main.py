"""
Experimental research script to find patterns correlations between genomic variability / conservation and sequence
or annotation information.
Conservation score is measured by Genomic Evolutionary Rate Profiling (GERP).
All data obtained from Ensembl release 109.
"""
import logging
import score_analysis
import score_collation

import pandas as pd

from data.load import read_sequence, read_annotation_generator, read_gerp_scorer
from data.paths import chr17_paths, wgs_paths  # paths to source data files
from data.process import get_train_test_x_y
from score_collation import score_stats_by_kmer, score_stats_by_dilated_kmer, sample_extreme_score_sequences, \
    aggregate_over_additive_field, score_stats_by_feature_type, aggregate_over_position, STRAND_COL
from score_nn_modeling import LocalWindowModel, ModelTrainer
from score_analysis import mutual_information_by_dilation, corrcoefs_by_score_count, diff_stats_by_score_count
from visuals import plot_mutual_information_by_dilation, plot_mutual_information_by_dilation_by_kmer
from genetic import get_feature_briefs, CODON_FORWARD_TABLE, MAIN_SEQ_NAMES
from protein import ProtFam

# from importlib import reload
# score_analysis = reload(score_analysis)
# score_collation = reload(score_collation)
# import visuals
# visuals = reload(visuals)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.precision', 3)
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
    # start the analysis with human chromosome 17
    # paths = chr17_paths
    paths = wgs_paths

    # get the raw sequence dictionary, from a FASTA file
    seq_dict = read_sequence(paths.sequence)

    # # (optional) examine annotations, from the annotation GFF file
    # examine_annotation(paths.annotation)

    # get the annotated sequence generator function, from the annotation GFF file
    seq_records_gen = read_annotation_generator(paths.annotation, seq_dict=seq_dict)

    # feature brief experiment
    # feature_briefs = get_feature_briefs(next(seq_records_gen()), ['CDS'], merge_overlapping_features=True,
    #                                       get_prot_fam=False)
    # feature_briefs = get_feature_briefs(next(seq_records_gen()), ['CDS'], merge_overlapping_features=True,
    #                                     get_prot_fam=True)

    # get GERP retrieval function, from the BigWig file
    gerp_scorer = read_gerp_scorer(paths.gerp)

    # analyze stats by feature type
    df = score_stats_by_feature_type(seq_records_gen, gerp_scorer)

    """Analysis"""
    # analyze by kmer for CDS (coding sequence) features
    kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['CDS'], k_values=[1, 2, 3])
    # kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['lnc_RNA'], k_values=[1,2,3])
    # kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['five_prime_UTR'], k_values=[1,2,3])

    _ = corrcoefs_by_score_count(kmer_base_df)
    _ = diff_stats_by_score_count(kmer_base_df)

    from importlib import reload
    score_analysis = reload(score_analysis)
    from score_analysis import mutual_information_by_dilation, corrcoefs_by_score_count, diff_stats_by_score_count
    stats = {seq_name: {} for seq_name in MAIN_SEQ_NAMES}
    for seq_name in MAIN_SEQ_NAMES:
        stats[seq_name]['many_cols'] = {}

        df_seq = kmer_base_df[kmer_base_df.seq_name == seq_name]
        print(f'\n\nScore correlations for {seq_name}')
        stats[seq_name]['many_cols']['corr'] = corrcoefs_by_score_count(df_seq)
        stats[seq_name]['many_cols']['diff'] = diff_stats_by_score_count(df_seq)

    for seq_name in MAIN_SEQ_NAMES:
        stats[seq_name]['few_cols'] = {}

        print(f'\n\nCount correlations for {seq_name}')
        df_seq = kmer_base_df[kmer_base_df.seq_name == seq_name]
        df_seq0 = aggregate_over_additive_field(df_seq, STRAND_COL)
        stats[seq_name]['few_cols']['corr'] = corrcoefs_by_score_count(df_seq0)
        stats[seq_name]['few_cols']['diff'] = diff_stats_by_score_count(df_seq0)

    df_filter = kmer_base_df[~kmer_base_df.seq_name.isin(['19','Y'])]
    _ = diff_stats_by_score_count(df_filter)


    _ = diff_stats_by_score_count(kmer_base_df)

    # without strand - we find count correlates here
    df0 = aggregate_over_additive_field(kmer_base_df, STRAND_COL)
    _ = corrcoefs_by_score_count(df0)
    _ = diff_stats_by_score_count(df0)

    """Codons"""
    df3 = kmer_base_df[kmer_base_df.k==3]
    df3['aa'] = df3.kmer.map(CODON_FORWARD_TABLE)
    # TODO: print for strand and frame

    # """Dilated"""
    # # analyze by dilated kmer for CDS (coding sequence) features
    # kmer_base_df_cds = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, ['CDS'], k_values=(1, 2,), dilations=range(1,20))
    # df_summary, df_kmer = mutual_information_by_dilation(kmer_base_df_cds)
    # plot_mutual_information_by_dilation(df_summary, title_prefix='cds_grp')
    # plot_mutual_information_by_dilation_by_kmer(df_kmer, title_prefix='cds_grp')

    # # analyze by dilated kmer for genes features
    # kmer_base_df_gene = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, ['gene'], k_values=(1, 2,), dilations=range(1,20))
    # df_out_gene = mutual_information_by_dilation(kmer_base_df_gene)
    # plot_mutual_information_by_dilation(df_out_gene, title_prefix='gene')
    #
    # # analyze by dilated kmer for samples from whole chromosome
    # kmer_base_df_whole = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, k_values=(1, 2,), dilations=range(1,20))
    # df_out_whole = mutual_information_by_dilation(kmer_base_df_whole)
    # plot_mutual_information_by_dilation(df_out_whole, title_prefix='whole_genome')

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
