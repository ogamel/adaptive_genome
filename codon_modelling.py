"""
Model codon specific properties in the genome and genetic code.
"""

import pandas as pd
import logging
import jax.numpy as jnp

from data.load import read_sequence, read_annotation_generator, read_gerp_scorer
from data.paths import chr17_paths  # paths to source data files
from data.process import process_codon_seq_score, N_BASES, train_test_x_y_from_seq_score
from score_collation import score_stats_by_kmer, score_stats_by_dilated_kmer, sample_extreme_score_sequences, \
    aggregate_over_additive_field, score_stats_by_feature_type, aggregate_over_position, STRAND_COL
from score_nn_modeling import LocalWindowModel, ModelTrainer
from score_analysis import mutual_information_by_dilation, corrcoefs_by_score_count, diff_stats_by_score_count
from visuals import plot_mutual_information_by_dilation, plot_mutual_information_by_dilation_by_kmer
from genetic import get_feature_briefs, CODON_FORWARD_TABLE

from xgboost.sklearn import XGBClassifier, XGBRegressor

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

    seq_train, score_train, seq_test, score_test = process_codon_seq_score(seq_records_gen, gerp_scorer, ['CDS'],
                                                                           codon_window=9)

    print(f'Dimensions: seq_train {seq_train.shape}, score_train {score_train.shape}, '
          f'seq_test {seq_test.shape}, score_test {score_test.shape}')

    ml_models = {}
    modes = ['bases_to_base', 'bases_to_score', 'scores_to_base', 'scores_to_score', 'bases+scores_to_base',
             'bases+scores_to_score', 'bases_trivial', 'scores_trivial']
    for mode in modes:
        x_train, y_train, x_test, y_test = train_test_x_y_from_seq_score(seq_train, score_train, seq_test, score_test,
                                                                         mode=mode)

        if mode.endswith('base'):
            # model = XGBClassifier(tree_method="hist", enable_categorical=False)
            model = XGBClassifier(
                objective='multi:softmax',  # for multiclass classification
                num_class=N_BASES,  # number of classes
                eval_metric='mlogloss'  # mlogloss is a common metric for multiclass classification
            )
        else:  # mode.endswith('score')
            model = XGBRegressor()

        # model = model_class()
        model.fit(x_train, y_train)  #.ravel()

        train_R2 = model.score(x_train, y_train)
        test_R2 = model.score(x_test, y_test)
        logging.info(f'Mode: {mode}.  Train R^2: {train_R2:.3f}.  Test R^2: {test_R2:.3f}.')

        ml_models[mode] = model
