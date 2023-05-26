"""
Experimental research script to find patterns correlations between genomic variability / conservation and sequence
or annotation information.
Conservation score is measured by Genomic Evolutionary Rate Profiling (GERP).
All data obtained from Ensembl release 109
"""
import logging

from data.load import read_sequence, read_annotation_generator, read_gerp_scorer
from data.paths import chr17_paths  # paths to source data files
from data.process import get_train_test_x_y
from score_analysis import score_stats_by_kmer, score_stats_by_dilated_kmer, sample_extreme_score_sequences
from score_modeling import LocalWindowModel, ModelTrainer

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

    """Analysis"""
    # # analyze by kmer for CDS (coding sequence) features
    # kmer_base_df = score_stats_by_kmer(seq_records_gen, gerp_scorer, ['CDS'], k_values=[1, 2, 3])

    # analyze by dilated kmer for CDS (coding sequence) features
    # kmer_base_df = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, ['CDS'], k_values=(2,), dilations=range(1,4))

    # analyze by dilated kmer for genes features
    # kmer_base_df = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, ['gene'], k_values=(2,), dilations=range(1,4))

    # analyze by gapped kmer for samples from whole chromosome
    # kmer_base_df_whole = score_stats_by_dilated_kmer(seq_records_gen, gerp_scorer, k_values=(2,), dilations=range(1,21))

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
