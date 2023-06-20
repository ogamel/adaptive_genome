"""
Predictive machine learning of GERP scores using neural networks.
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
from data.process import DEFAULT_WINDOW, DEFAULT_BP_WINDOW, N_BASES
from nn.components import TransformerLayer

HIDDEN_LAYERS = (2, DEFAULT_WINDOW//2)  # number and width of hidden layers

LEARNING_RATE = 0.1
SEED = 200  # TODO: put default seed in one place, now it is duplicated. actually put all common constants in one place
MAX_EPOCHS = 100


class Model(eqx.Module):
    """Abstract bass class for models."""
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Model initialization call."""
        pass

    @abstractmethod
    def __call__(self, x):
        """Model inference call."""
        pass

    @abstractmethod
    def loss(self, x, y):
        """Computing the loss for inference on feature x, with ground truth y."""
        pass

    def score(self, x, y):
        """R^2 score, based on loss() function above, which is assumed to be mean square error."""
        return 1-self.loss(x, y)/jnp.var(y)

    @abstractmethod
    def update(self, x, y):
        """Update the model parameter one step by training on features x with ground truth y."""
        pass


class ModelTrainer:
    """Class that wraps a model with its training parameters, optimizer."""
    # TODO: make it so model is modified in place as the trainer updates it.
    model: Model
    learning_rate: float
    epochs: int
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
            loss_value, grads = eqx.filter_value_and_grad(model.__class__.loss)(model, x, y)
            updates, opt_state = self.optim.update(grads, opt_state, model)
            # TODO: This line needs to change model in place
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        var_y = jnp.var(y)

        for i in range(self.epochs):
            self.model, self.opt_state, loss_value = step(self.model, self.opt_state, x, y)
            periodic_logging(i, f'Epoch {i:,}. Training loss: {loss_value:.4f}. R^2: {1-loss_value/var_y:.4f}.', v=100)

        logging.info('Training complete.')

        return


class LocalWindowModel(Model):
    """Naive fully connected neural network, applied to a sliding window around the nucleotide of interest."""

    layers: list[eqx.nn.Linear]
    extra_bias: jax.Array

    def __init__(self, key=None, window=DEFAULT_WINDOW, hidden_layers=HIDDEN_LAYERS):
        if key is None:
            key = PRNGKey(SEED)

        # wh is always even (since 4 members of alphabet), though the number of bases in the window can be odd
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


class SymmetricLocalWindowModel(Model):
    """
    Fully connected neural network, symmetric in inputs, applied to a sliding window around the nucleotide of interest.
    Symmetry combined with one hot encoding of 'GATC' amounts to hardocding the euivalence of reverse complement scores.
    This is since G--> [1 0 0 0] and its complement C--> [0 0 0 1] has the reverse vector. Same with A and T.
    Hidden layer size is also halved, due to the lower number of independent variables.
    TODO: revisit halving the hidden layer size.
    """

    layers: list[eqx.nn.Linear]
    extra_bias: jax.Array

    def __init__(self, key=None, window=DEFAULT_WINDOW, hidden_layers=HIDDEN_LAYERS):
        if key is None:
            key = PRNGKey(SEED)

        nh, wh = hidden_layers

        keys = jax.random.split(key, nh + 2)
        self.layers = [eqx.nn.Linear(window//2, wh, key=keys[0])]  # input layer
        self.layers += [eqx.nn.Linear(wh, wh, key=keys[i+1]) for i in range(nh)]  # hidden layers
        self.layers += [eqx.nn.Linear(wh, 1, key=keys[-1])]  # output layer

        # trainable extra bias
        self.extra_bias = jnp.ones(1)

    def __call__(self, x):
        """Model inference call."""
        half_l = len(x)//2
        layer = self.layers[0]
        x = jax.nn.relu(layer(x[:half_l]) + layer(x[:half_l-1:-1]))
        for layer in self.layers[1:-1]:
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
        grads = jax.grad(SymmetricLocalWindowModel.loss)(self, x, y)
        return jax.tree_util.tree_map(lambda m, g: m - LEARNING_RATE * g, self, grads)


class LocalTransformerEncoderModel(Model):
    """
    Self Attention applied to same sliding window as LocalWindowModel.
    Steps: Positional encoding, MultiHead Attention with residual, feedforward with residual
    """

    NUM_TRANSFORMER_LAYERS = 3
    NUM_ATTENTION_HEADS = 3
    DROPOUT_RATE = 0.1

    position_embedder: eqx.nn.Embedding
    layers: list[TransformerLayer]
    extra_bias: jax.Array

    def __init__(self, key=None, num_transformer_layers=NUM_TRANSFORMER_LAYERS, hidden_size=N_BASES,
                 num_heads=NUM_ATTENTION_HEADS, dropout_rate=DROPOUT_RATE, attention_dropout_rate=DROPOUT_RATE):
        if key is None:
            key = PRNGKey(SEED)

        keys = jax.random.split(key, num_transformer_layers + 2)
        self.position_embedder = eqx.nn.Embedding(num_embeddings=DEFAULT_BP_WINDOW,
                                                  embedding_size=N_BASES, key=keys[0])

        self.layers = [TransformerLayer(hidden_size, hidden_size, num_heads, dropout_rate,
                                            attention_dropout_rate, keys[i+1]) for i in range(num_transformer_layers)]
        self.layers += [eqx.nn.Linear(DEFAULT_BP_WINDOW*N_BASES, 1, key=keys[-1])]  # output layer

        self.extra_bias = jnp.ones(1)  # trainable extra bias

    def __call__(self, x):
        """Model inference call."""
        num_embeddings, embedding_size = self.position_embedder.num_embeddings, self.position_embedder.embedding_size
        x = jnp.reshape(x, [num_embeddings, embedding_size])

        position_ids = jnp.arange(num_embeddings)
        positions = self.position_embedder(position_ids)
        x += positions

        for layer in self.layers[:-1]:
            x = layer(x)
        x = jnp.reshape(x, [-1])  # flatten before final fully connected layer
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
        grads = jax.grad(LocalTransformerEncoderModel.loss)(self, x, y)
        return jax.tree_util.tree_map(lambda m, g: m - LEARNING_RATE * g, self, grads)
