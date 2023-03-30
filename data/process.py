"""Module to process loaded data, to prepare it for modeling and analysis where necessary."""

import logging
from functools import partial
from typing import Iterator, Callable

import jax
import jax.numpy as jnp
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split

from genetic import NUCLEOTIDE_ALPHABET, get_feature_briefs
from util import periodic_logging

BASE_TO_INT = {base: i for i, base in enumerate(NUCLEOTIDE_ALPHABET)}  # encode base to int
N_BASES = len(BASE_TO_INT)

# window in base pairs, note Inversion Symmetry for k-mers roughly holds up to k=10, we cover it from both sides
DEFAULT_BP_WINDOW = 19
# window in one-hot encoded parameters
DEFAULT_WINDOW = DEFAULT_BP_WINDOW * N_BASES

HIDDEN_LAYERS = (2, DEFAULT_WINDOW//2)  # number and width of hidden layers
SEED = 200

USE_SOFTMASKED = True  # flag to whether to use softmasked nucleotides, in small letters e.g. 'gatc'


def sequence_to_one_hot_array(sequence: str):
    """
    Convert the input sequence string to a jax numpy array based on concatenated one hot encoding of each nucleotide.
    Returned array is two-dimensional, where each row represents a sequence position, and each column a one hot encoded
    category (i.e. a member of the nucleotide alphabet).
    """
    return jax.nn.one_hot(jnp.array(list(map(BASE_TO_INT.get, sequence))), N_BASES)


@partial(jax.jit, static_argnums=(1,))
def sliding_window(in_array: jnp.array, window: int = DEFAULT_BP_WINDOW):
    """
    Combine rows of 2d array in a sliding window, returns a new 2d array where each row is a concatenation of (window)
    consecutive rows in in_array.
    """
    n_rows, n_cols = in_array.shape
    window = min(window, n_rows)
    starts = jnp.arange(n_rows - window + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(in_array, (start, 0), (window, n_cols)).flatten())(starts)


def get_train_test_x_y(seq_records_gen: Callable[[], Iterator[SeqRecord]],
                       scorer: Callable[[str, int, int], list[float]], feature_type_filter: list[str]):
    """Build neural network to predict score based on sequence. """

    MAX_ROWS = 200000
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
            periodic_logging(i, f'Processing feature {i:,}. x_train shape: {x_train.shape}', v=100)

            if len(x_train) > MAX_ROWS:
                break
            d = DEFAULT_BP_WINDOW//2

            ft_len = ft.end - ft.start + 1
            # expand sequence range by half the local window on either side
            ft_sequence = str(seq_record.seq[ft.start - d: ft.end + 1 + d])
            if USE_SOFTMASKED:
                ft_sequence = ft_sequence.upper()
                # TODO: update code below so that if USE_SOFTMASKED is False, small letters are ignored without dropping
                #  the entire feature. Unclear how to do this efficiently in a window function though.
            ft_scores = scorer(seq_name, ft.start, ft.end + 1)

            # print(ft_sequence)
            ft_x = sliding_window(sequence_to_one_hot_array(ft_sequence))

            ft_scores_array = jnp.array(ft_scores).reshape((ft_len, 1))
            # TODO: consider a better way to deal with nan than imputation of the mean
            ft_y = jnp.nan_to_num(ft_scores_array, nan=jnp.nanmean(ft_scores_array))

            assert len(ft_x) == len(ft_y)

            if i in train_indices:
                # append to train
                x_train = jnp.append(x_train, ft_x, axis=0)
                y_train = jnp.append(y_train, ft_y, axis=0)
            else:
                # append to test
                x_test = jnp.append(x_test, ft_x, axis=0)
                y_test = jnp.append(y_test, ft_y, axis=0)

    return x_train, y_train, x_test, y_test
