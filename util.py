"""
Utility functions module.
"""

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
# formatter = logging.Formatter('\033[1A\x1b[2K%(message)s')


ROUND_DECIMALS = 3


def periodic_logging(n, msg=None, v=25000):
    """Log number message if n is divisible by v. Default msg value is n."""
    # TODO: overwrite previous info line, and lower default v.
    if not msg:
        msg = n
    if n % v == 0:
        logging.info(msg)
    return


def rd(n, d=ROUND_DECIMALS):
    """Round n to the nearest d decimal points."""
    return np.round(n, d)


def std_to_std_of_mean(std_array, weights=None):
    """
    Given an array of the standard deviations of some quantities, what is the standard deviation of the quantities'
    (possibly weighted) average.
    """
    if weights is None:
        weights = np.ones(len(std_array))
    weights = weights / np.sum(weights)
    return np.sqrt(np.sum(np.square(np.multiply(std_array, weights))))


def std_to_std_of_sum(std_array, mean_array, counts_array):
    """
    Given an array of the standard deviations of some quantities, what is the standard deviation of the quantities' sum.
    """
    # std = np.sqrt(s2 / s0 - (s1 / s0) ** 2)
    s2 = ((std_array ** 2 + mean_array ** 2) * counts_array).sum()
    s1 = (mean_array * counts_array).sum()
    s0 = counts_array.sum()
    return np.sqrt(s2 / s0 - (s1 / s0) ** 2)
