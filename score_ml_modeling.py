"""
Predictive machine learning of GERP scores using classic machine learning techniques.
The goal is to predict GERP as a function of annotation features and the sequence.
"""

from util import periodic_logging, rd
from typing import Iterator, Callable
import logging
import numpy as np
import pandas as pd
import typing
import random
from pprint import pprint
from collections import Counter, defaultdict
from Bio.SeqRecord import SeqRecord
from genetic import NUCLEOTIDE_ALPHABET, get_feature_briefs
from functools import partial
from data.load import read_gerp_scorer
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

from data.process import DEFAULT_WINDOW, DEFAULT_BP_WINDOW, N_BASES



LEARNING_RATE = 0.1
SEED = 200  # TODO: put default seed in one place, now it is duplicated. actually put all common constants in one place
MAX_EPOCHS = 100

