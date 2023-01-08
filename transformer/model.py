import jax
import haiku as hk
import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class MultiHeadAttention(hk.Module):
    """
    Multi-head attention layer
    """
    
    num_heads: int
    num_features: int
    dropout_rate: float
        
    def __call__(self, 
                 q: jnp.ndarray, 
                 v: jnp.ndarray, 
                 k: jnp.ndarray, 
                 mask: jnp.ndarray, 
                 rng) -> jnp.ndarray:

        # q, v, k: (batch_size, seq_len, num_features)
        q = hk.Linear(self.num_features)(q)  
        v = hk.Linear(self.num_features)(v)
        k = hk.Linear(self.num_features)(k)
        q = self.split_heads(q)
        v = self.split_heads(v)
        k = self.split_heads(k)

        # attention eq (1)
        q = q / jnp.sqrt(self.num_features)
        logits = jnp.einsum('ijkl,ijml->ijkm', q, k)
        logits = logits - 1e9 * (1 - mask)
        weights = jax.nn.softmax(logits)
        weights = hk.dropout(rng, self.dropout_rate, weights)
        output = weights @ v
        output = self.combine_heads(output)
        output = hk.Linear(self.num_features)(output)

        return output
    
    def split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch_size, seq_len, num_features]
        # output: [batch_size, num_heads, seq_len, num_features // num_heads]
        x = jnp.reshape(x, (x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads))
        return jnp.transpose(x, (0, 2, 1, 3))
    
    def combine_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch_size, num_heads, seq_len, num_features // num_heads]
        # output: [batch_size, seq_len, num_features]
        x = jnp.transpose(x, (0, 2, 1, 3))
        return jnp.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))


@dataclass
class PositionWiseFeedForward(hk.Module):
    """
    Position-wise feed-forward layer, eq (2)
    """

    num_features: int
    dropout_rate: float

    def __call__(self, x: jnp.ndarray, rng) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        x = hk.Linear(self.num_features)(x)
        x = jax.nn.relu(x)
        x = hk.dropout(rng, self.dropout_rate, x)
        x = hk.Linear(self.num_features)(x)
        x = hk.dropout(rng, self.dropout_rate, x)
        return x


@dataclass
class Embedding(hk.Module):
    """
    Embedding layer
    """
    
    vocab_size: int
    num_features: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch_size, seq_len)
        x = hk.Embed(self.vocab_size, self.num_features)(x)
        return x


@dataclass
class EncoderLayer(hk.Module):
    """
    Encoder layer
    """

    num_heads: int
    num_features: int
    dropout_rate: float

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, rng) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        attention = MultiHeadAttention(self.num_heads, self.num_features, self.dropout_rate)
        feed_forward = PositionWiseFeedForward(self.num_features, self.dropout_rate)
        x = attention(x, x, x, mask, rng)
        x = feed_forward(x, rng)
        return x


@dataclass
class Encoder(hk.Module):
    """
    Encoder
    """

    num_layers: int
    num_heads: int
    num_features: int
    dropout_rate: float

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, rng) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        for _ in range(self.num_layers):
            x = EncoderLayer(self.num_heads, self.num_features, self.dropout_rate)(x, mask, rng)
        return x
    

@dataclass
class DecoderLayer(hk.Module):
    """
    Decoder layer
    """

    num_heads: int
    num_features: int
    dropout_rate: float

    def __call__(self, x: jnp.ndarray, encoder_output: jnp.ndarray, mask: jnp.ndarray, rng) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        attention = MultiHeadAttention(self.num_heads, self.num_features, self.dropout_rate)
        feed_forward = PositionWiseFeedForward(self.num_features, self.dropout_rate)
        x = attention(x, x, x, mask, rng)
        x = attention(x, encoder_output, encoder_output, mask, rng)
        x = feed_forward(x, rng)
        return x


@dataclass
class Decoder(hk.Module):
    """
    Decoder
    """

    num_layers: int
    num_heads: int
    num_features: int
    dropout_rate: float

    def __call__(self, x: jnp.ndarray, encoder_output: jnp.ndarray, mask: jnp.ndarray, rng) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        for _ in range(self.num_layers):
            x = DecoderLayer(self.num_heads, self.num_features, self.dropout_rate)(x, encoder_output, mask, rng)
        return x
    

@dataclass
class Transformer(hk.Module):
    """
    Transformer
    """

    num_layers: int
    num_heads: int
    vocal_size: int
    num_features: int
    dropout_rate: float

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray, rng) -> jnp.ndarray:
        embedding = Embedding(self.vocal_size, self.num_features)
        encoder = Encoder(self.num_layers, self.num_heads, self.num_features, self.dropout_rate)
        decoder = Decoder(self.num_layers, self.num_heads, self.num_features, self.dropout_rate)
        x = embedding(x)
        y = embedding(y)
        x = encoder(x, mask, rng)
        y = decoder(y, x, mask, rng)
        return y


if __name__ == '__main__':
    # Test
    
    def f(x: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray, rng):
        return Transformer(2, 2, 100, 64, 0.1)(x, y, mask, rng)
    
    x = jnp.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = jnp.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    mask = jnp.array([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]], [[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]])  # (2, 1, 3, 3)
    
    model = hk.transform(f)
    rng = jax.random.PRNGKey(42)
    
    init_rng, rng = jax.random.split(rng)
    params = model.init(init_rng, x, y, mask, rng)
    out = model.apply(params, rng, x, y, mask, rng)
    print(out.shape)
