"""Commonly used neural network components."""

import jax
import equinox as eqx


class FeedForwardBlock(eqx.Module):
    """
    A single transformer feed forward block.
    Based on https://docs.kidger.site/equinox/examples/bert/
    """

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, hidden_size, intermediate_size, dropout_rate, key):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(in_features=hidden_size, out_features=intermediate_size, key=mlp_key)
        self.output = eqx.nn.Linear(in_features=intermediate_size, out_features=hidden_size, key=output_key)

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs, enable_dropout = True, key = None):
        # Feed-forward.
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual and layer norm.
        output += inputs
        output = self.layernorm(output)

        return output


class AttentionBlock(eqx.Module):
    """
    A single transformer attention block.
    Based on https://docs.kidger.site/equinox/examples/bert/
    """

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.static_field()

    def __init__(self, hidden_size, num_heads, dropout_rate, attention_dropout_rate, key):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, inputs, enable_dropout = False, key: "jax.random.PRNGKey" = None):
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        result = jax.vmap(self.layernorm)(result)
        return result


class TransformerLayer(eqx.Module):
    """
    A single transformer layer.
    Based on https://docs.kidger.site/equinox/examples/bert/
    """

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(self, hidden_size, intermediate_size, num_heads, dropout_rate, attention_dropout_rate, key):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(self, inputs, *, enable_dropout = False, key = None):
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(inputs, enable_dropout=enable_dropout, key=attn_key)
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output
