"""
Model codon specific properties in the genome and genetic code.
"""

import pandas as pd
import logging
import jax.numpy as jnp
from collections import defaultdict
from genetic import kmers_list, NUCLEOTIDE_ALPHABET
from util import rd
from importlib import reload
import genetic
reload(genetic)
import sklearn

from data.load import read_sequence, read_annotation_generator, read_gerp_scorer
from data.paths import chr17_paths  # paths to source data files
from data.process import process_codon_seq_score, N_BASES, train_test_x_y_from_seq_score, sequence_to_one_hot_array
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

codon_window = 5

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
                                                                           codon_window=codon_window)

    print(f'Dimensions: seq_train {seq_train.shape}, score_train {score_train.shape}, '
          f'seq_test {seq_test.shape}, score_test {score_test.shape}')

    ml_models = {}
    out_scores = {}  # defaultdict(list)

    modes = ['bases_to_base', 'bases_to_score', 'scores_to_base', 'scores_to_score', 'bases+scores_to_base',
             'bases+scores_to_score']  # 'bases_trivial', 'scores_trivial']
    for mode in modes:
        ml_models[mode] = []
        out_scores[mode] = {'train': [], 'test': []}
        for c in range(codon_window):
            x_train, y_train, x_test, y_test = train_test_x_y_from_seq_score(seq_train[:, 3 * c * N_BASES:],
                                    score_train[:, 3*c:], seq_test[:, 3 * c * N_BASES:], score_test[:, 3*c:], mode=mode)
            # print(f'Dimensions c={c}: x_train {x_train.shape}, y_train {y_train.shape}, '
            #       f'x_test {x_test.shape}, y_test {y_test.shape}')

            if mode.endswith('base'):
                # model = XGBClassifier(tree_method="hist", enable_categorical=False)
                model = XGBClassifier(
                    objective='multi:softmax',  # for multiclass classification
                    num_class=N_BASES,  # number of classes
                    eval_metric='mlogloss'  # mlogloss is a common metric for multiclass classification
                )
                score_name = 'mean acc'
            else:  # mode.endswith('score')
                model = XGBRegressor()
                score_name = 'R^2'

            # model = model_class()
            model.fit(x_train, y_train)

            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            out_scores[mode]['train'].append(train_score)
            out_scores[mode]['test'].append(test_score)

            logging.info(f'Mode: {mode}. Window: {codon_window-c}. Train {score_name}: {train_score:.3f}.  '
                         f'Test {score_name}: {test_score:.3f}.')

            ml_models.append(model)

    for mode in modes:
        train_score_str = "\t".join((str(rd(x)) for x in out_scores[mode]["train"]))
        test_score_str = "\t".join((str(rd(x)) for x in out_scores[mode]["test"]))
        print(f'{mode}\t{train_score_str}\t{test_score_str}')

    # check the base to base case with first two bases of a codon, see how they predict the third, and compare this
    # with the list of codon frequencies
    model = ml_models['bases_to_base'][4]
    kmers = kmers_list(2, order='alphabetical')
    x_trial = jnp.stack([sequence_to_one_hot_array(kmer).flatten() for kmer in kmers])
    y_trial = model.predict(x_trial)
    last_bases = [NUCLEOTIDE_ALPHABET[i] for i in y_trial]
    out_codons = [double + base for double, base in zip(kmers, last_bases)]
    # as expected, it just chooses the most common of the four codons with the first two bases fixed

