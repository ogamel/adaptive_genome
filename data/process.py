"""Module to process loaded data, to prepare it for nn and analysis where necessary."""

import logging
from functools import partial
from typing import Iterator, Callable

import jax
import jax.numpy as jnp
import numpy as np
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split

from genetic import NUCLEOTIDE_ALPHABET, get_feature_briefs
from util import periodic_logging

BASE_TO_INT = {base: i for i, base in enumerate(NUCLEOTIDE_ALPHABET)}  # encode base to int
N_BASES = len(BASE_TO_INT)

# window in base pairs, note Inversion Symmetry for k-mers roughly holds up to k=10, we cover it from both sides
# DEFAULT_BP_WINDOW = 19
DEFAULT_BP_WINDOW = 8
# window in one-hot encoded parameters
DEFAULT_WINDOW = DEFAULT_BP_WINDOW * N_BASES
DEFAULT_CODON_WINDOW = 6

HIDDEN_LAYERS = (2, DEFAULT_WINDOW//2)  # number and width of hidden layers
SEED = 200

USE_SOFTMASKED = True  # flag to whether to use softmasked nucleotides, in small letters e.g. 'gatc'
MAX_TRAIN_ROWS = 300000


def sequence_to_one_hot_array(sequence: str):
    """
    Convert the input sequence string to a jax numpy array based on concatenated one hot encoding of each nucleotide.
    Returned array is two-dimensional, where each row represents a sequence position, and each column a one hot encoded
    category (i.e. a member of the nucleotide alphabet).
    """
    return jax.nn.one_hot(jnp.array(list(map(BASE_TO_INT.get, sequence)), dtype=int), N_BASES)


def onehot_to_num(input):
    """Converts array where each row is one hot encoded vector to vector of numbers."""
    return jnp.argmax(input, axis=1)


# @partial(jax.jit, static_argnums=(1,3))
def sliding_window(in_array: jnp.array, window: int = DEFAULT_BP_WINDOW, step=1):
    """
    Combine rows of 2d array in a sliding window, returns a new 2d array where each row is a concatenation of (window)
    consecutive rows in in_array.
    """
    n_rows, n_cols = in_array.shape
    window = min(window, n_rows)
    starts = jnp.arange(n_rows - window, step=step)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(in_array, (start, 0), (window, n_cols)).flatten())(starts)


def get_train_test_x_y(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                       scorer: Callable[[str, int, int], list[float]], feature_type_filter: list[str],
                       max_train_rows=MAX_TRAIN_ROWS):
    """Build data to be fed into neural network to predict score based on sequence. NaNs are ignored."""

    seq_records = seq_records_gen()

    x_train, y_train = x_test, y_test = jnp.empty((0,DEFAULT_WINDOW)), jnp.empty((0,1))

    # collect train and test data
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
        train_indices, test_indices = train_test_split(range(len(feature_briefs)), train_size=0.8,
                                                                     random_state=SEED)

        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}. x_train shape: {x_train.shape}', v=len(feature_briefs)//100)

            if len(x_train) > max_train_rows:
                break
            d = DEFAULT_BP_WINDOW//2

            # expand sequence range by half the local window on either side
            ft_sequence = str(seq_record.seq[ft.start - d: ft.end + d])
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
                # TODO: update code below so that if USE_SOFTMASKED is False, small letters are ignored without dropping
                #  the entire feature. Unclear how to do this efficiently in a window function though.

            ft_scores = jnp.array(scorer(seq_name, ft.start, ft.end))
            nan_indices = jnp.isnan(ft_scores)

            ft_x = sliding_window(sequence_to_one_hot_array(ft_sequence))
            ft_x = ft_x[~nan_indices, :]
            ft_y = ft_scores[~nan_indices].reshape((-1, 1))

            assert len(ft_x) == len(ft_y)
            if len(ft_y) == 0:
                continue

            if i in train_indices:
                # append to train
                x_train = jnp.append(x_train, ft_x, axis=0)
                y_train = jnp.append(y_train, ft_y, axis=0)
            else:
                # append to test
                x_test = jnp.append(x_test, ft_x, axis=0)
                y_test = jnp.append(y_test, ft_y, axis=0)

    return x_train, y_train, x_test, y_test


def process_codon_seq_score(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                            scorer: Callable[[str, int, int], list[float]], feature_type_filter: list[str],
                            codon_window=DEFAULT_CODON_WINDOW, strand=1, max_train_rows=MAX_TRAIN_ROWS):
    """
    Build data to be fed into regressor or neural network to predict score based on sequence. NaNs are ignored.
    If y_frame is set, only get y with that frame value. If None, then get everything.

    mode:
    bases to base
    bases to score
    bases + aa to base
    scores to score
    bases + scores to score
    bases + scores to base
    bases + scores + aa to base
    """

    seq_records = seq_records_gen()

    seq_train_list, score_train_list, seq_test_list, score_test_list = [], [], [], []

    # collect train and test data
    for seq_record in seq_records:
        seq_name = seq_record.name
        logging.info(f'Sequence {seq_name} ...')

        feature_briefs = get_feature_briefs(seq_record, feature_type_filter)
        train_indices, test_indices = train_test_split(range(len(feature_briefs)), train_size=0.8,
                                                                     random_state=SEED)

        for i, ft in enumerate(feature_briefs):
            periodic_logging(i, f'Processing feature {i:,}.', v=len(feature_briefs)//100)

            if len(seq_train_list) > max_train_rows:
                break
            if strand is not None and ft.strand != strand:
                continue
            if not ft.start + ft.phase < ft.end:
                continue

            ft_sequence = str(seq_record.seq[ft.start + ft.phase: ft.end])  # be in coding frame
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
                # TODO: update code below so that if USE_SOFTMASKED is False, small letters are ignored without dropping
                #  the entire feature. Unclear how to do this efficiently in a window function though.

            ft_scores = jnp.array(scorer(seq_name, ft.start + ft.phase, ft.end)).reshape(-1, 1)

            seq_array = sliding_window(sequence_to_one_hot_array(ft_sequence), window=3*codon_window,
                                       step=3)  # step is one codon
            score_array = sliding_window(ft_scores, window=3*codon_window, step=3)

            # remove rows with nan score
            nan_row_indices = jnp.isnan(score_array).any(axis=1)
            seq_array = seq_array[~nan_row_indices]
            score_array = score_array[~nan_row_indices]

            if len(seq_array) == 0:
                continue

            if i in train_indices:
                # append to train
                seq_train_list.append(seq_array)
                score_train_list.append(score_array)
            else:
                # append to test
                seq_test_list.append(seq_array)
                score_test_list.append(score_array)

    seq_train, score_train = jnp.concatenate(seq_train_list), jnp.concatenate(score_train_list)
    seq_test, score_test = jnp.concatenate(seq_test_list), jnp.concatenate(score_test_list)

    return seq_train, score_train, seq_test, score_test


def train_test_x_y_from_seq_score(seq_train, score_train, seq_test, score_test, mode='bases_to_base', target_y_frame=2):
    """Return the right train and testt slices of input arrays, which are created by process_codon_seq_score."""

    x_train = y_train = x_test = y_test = None
    # choose x and y depending on training mode
    if mode == 'bases_to_base':
        x_train, y_train = seq_train[:, :-N_BASES], onehot_to_num(seq_train[:, -N_BASES:])
        x_test, y_test = seq_test[:, :-N_BASES], onehot_to_num(seq_test[:, -N_BASES:])
    elif mode == 'bases_to_score':
        x_train, y_train = seq_train, score_train[:, -1]
        x_test, y_test = seq_test, score_test[:, -1]
    elif mode == 'scores_to_base':
        x_train, y_train = score_train, onehot_to_num(seq_train[:, -N_BASES:])
        x_test, y_test = score_test, onehot_to_num(seq_test[:, -N_BASES:])
    elif mode == 'scores_to_score':
        x_train, y_train = score_train[:, :-1], score_train[:, -1]
        x_test, y_test = score_test[:, :-1], score_test[:, -1]
    elif mode == 'bases+scores_to_base':
        x_train, y_train = jnp.append(seq_train[:, :-N_BASES], score_train, axis=1), \
                           onehot_to_num(seq_train[:, -N_BASES:])
        x_test, y_test = jnp.append(seq_test[:, :-N_BASES], score_test, axis=1), \
                         onehot_to_num(seq_test[:, -N_BASES:])
    elif mode == 'bases+scores_to_score':
        x_train, y_train = jnp.append(seq_train, score_train[:, :-1], axis=1), score_train[:, -1]
        x_test, y_test = jnp.append(seq_test, score_test[:, :-1], axis=1), score_test[:, -1]
    elif mode == 'random_to_base':
        x_train, y_train = np.random.rand(*seq_train.shape), onehot_to_num(seq_train[:, -N_BASES:])
        x_test, y_test = np.random.rand(*seq_test.shape), onehot_to_num(seq_test[:, -N_BASES:])
    elif mode == 'random_to_score':
        x_train, y_train = np.random.rand(*score_train.shape), score_train[:, -1]
        x_test, y_test = np.random.rand(*score_test.shape), score_test[:, -1]
    # elif mode == 'trivial_to_base':
    #     x_train, y_train = seq_train, onehot_to_num(seq_train[:, -N_BASES:])
    #     x_test, y_test = seq_test, onehot_to_num(seq_test[:, -N_BASES:])
    # elif mode == 'trivial_to_score':
    #     x_train, y_train = score_train, score_train[:, -1]
    #     x_test, y_test = score_test, score_test[:, -1]

    return x_train, y_train, x_test, y_test
