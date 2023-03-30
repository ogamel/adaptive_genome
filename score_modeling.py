"""
Predictive machine learning modeling of GERP scores.
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

import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
# import jaxopt
# import jax.example_libraries.optimizers as jax_opt
import equinox as eqx

from jax.random import PRNGKey
from data.process import DEFAULT_WINDOW

HIDDEN_LAYERS = (2, DEFAULT_WINDOW//2)  # number and width of hidden layers
LEARNING_RATE = 0.1
SEED = 200  # TODO: put default seed in one place, now it is duplicated. actually put all common constants in one place
MAX_EPOCHS = 100


class Model(eqx.Module):
    """Abstract bass class for models."""
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def loss(self, x, y):
        pass

    @abstractmethod
    def update(self, x, y):
        pass


class ModelTrainer:
    """Class that wraps a model with its training parameters, optimizer."""
    model: Model
    learning_rate: float
    optim: optax.GradientTransformation
    opt_state: PyTree

    def __init__(self, model: Model, learning_rate=LEARNING_RATE, epochs=MAX_EPOCHS, optimizer=optax.adam):
        """Initialize a model trainer, from a model and its training hyper-parameters."""

        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optim = optimizer(learning_rate)
        self.opt_state = self.optim.init(eqx.filter(model, eqx.is_array))

    def __call__(self, x):
        """Model inference call."""
        return self.model(x)

    def train(self, x, y):
        """Model training call, returns trained version of current model on input and output x and y."""

        @eqx.filter_jit
        def step(model, opt_state, x, y):
            loss_value, grads = eqx.filter_value_and_grad(LocalWindowModel.loss)(model, x, y)
            updates, opt_state = self.optim.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        for i in range(self.epochs):
            self.model, self.opt_state, loss_value = step(self.model, self.opt_state, x, y)
            periodic_logging(i, f'Epoch {i:,}. Training loss: {loss_value:.4f}.', v=10)

        logging.info('Training complete.')

        return self.model


class LocalWindowModel(Model):
    """Basic fully connected neural network, applied to a sliding window around the nucleotide of interest."""
    layers: list
    extra_bias: jax.Array

    def __init__(self, key=None, window=DEFAULT_WINDOW, hidden_layers=HIDDEN_LAYERS):
        if key is None:
            key = PRNGKey(SEED)

        nh, wh = hidden_layers
        keys = jax.random.split(key, nh + 2)
        self.layers = [eqx.nn.Linear(window, wh, key=keys[0])]  # input layer
        self.layers += [eqx.nn.Linear(wh, wh, key=keys[i+1]) for i in range(nh)]  # hidden layers
        self.layers += [eqx.nn.Linear(wh, 1, key=keys[-1])]  # output layer

        # trainable extra bias
        self.extra_bias = jnp.ones(1)

    def __call__(self, x):
        """Model inference call."""
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias

    def loss(self, x, y):
        """Calculate model loss function value for given input and output."""
        # TODO: consider a measure of error different from MSE, one more in tune with how GERP score is calculated
        pred_y = jax.vmap(self)(x)
        return jnp.nanmean((y - pred_y) ** 2)

    @jax.jit
    def update(self, x, y):
        """
        Basic model training call, returns single training step, unoptimized, updated version of current model
        on input and output x and y. Preferable to use ModelTrainer.train(x, y).
        """

        # loss function is called with self (i.e. the model along with its parameters) as an explicit parameter,
        # since jax.grad differentiates w.r.t. the first argument
        grads = jax.grad(LocalWindowModel.loss)(self, x, y)
        return jax.tree_util.tree_map(lambda m, g: m - LEARNING_RATE * g, self, grads)
