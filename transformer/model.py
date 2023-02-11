import jax
import jax.numpy as jnp

from flax import linen as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer
    """
    
    num_heads: int
    num_features: int
    dropout_rate: float
        
    @nn.compact
    def __call__(self, 
                 q: jnp.ndarray, 
                 v: jnp.ndarray, 
                 k: jnp.ndarray, 
                 mask: jnp.ndarray,
                 eval_mode: bool) -> jnp.ndarray:

        # q, v, k: (batch_size, seq_len, num_features)
        q = nn.Dense(self.num_features)(q)  
        v = nn.Dense(self.num_features)(v)
        k = nn.Dense(self.num_features)(k)
        q = self.split_heads(q)
        v = self.split_heads(v)
        k = self.split_heads(k)

        # attention eq (1)
        q = q / jnp.sqrt(self.num_features)
        logits = jnp.einsum('ijkl,ijml->ijkm', q, k)
        logits = logits - 1e9 * (1 - mask)
        weights = nn.softmax(logits)
        weights = nn.Dropout(self.dropout_rate, deterministic=eval_mode)(weights)
        output = weights @ v
        output = self.combine_heads(output)
        output = nn.Dense(self.num_features)(output)

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


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward layer, eq (2)
    """

    num_features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        x = nn.Dense(self.num_features)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=eval_mode)(x)
        x = nn.Dense(self.num_features)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=eval_mode)(x)
        return x


class Embedding(nn.Module):
    """
    Embedding layer
    """
    
    vocab_size: int
    num_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch_size, seq_len)
        x = nn.Embed(self.vocab_size, self.num_features)(x)
        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer
    """

    num_heads: int
    num_features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        attention = MultiHeadAttention(self.num_heads, self.num_features, self.dropout_rate)
        feed_forward = PositionWiseFeedForward(self.num_features, self.dropout_rate)
        x = attention(x, x, x, mask, eval_mode)
        x = feed_forward(x, eval_mode)
        return x


class Encoder(nn.Module):
    """
    Encoder
    """

    num_layers: int
    num_heads: int
    num_features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        for _ in range(self.num_layers):
            x = EncoderLayer(self.num_heads, self.num_features, self.dropout_rate)(x, mask, eval_mode)
        return x
    

class DecoderLayer(nn.Module):
    """
    Decoder layer
    """

    num_heads: int
    num_features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, encoder_output: jnp.ndarray, mask: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        attention = MultiHeadAttention(self.num_heads, self.num_features, self.dropout_rate)
        feed_forward = PositionWiseFeedForward(self.num_features, self.dropout_rate)
        x = attention(x, x, x, mask, eval_mode)
        x = attention(x, encoder_output, encoder_output, mask, eval_mode)
        x = feed_forward(x, eval_mode)
        return x


class Decoder(nn.Module):
    """
    Decoder
    """

    num_layers: int
    num_heads: int
    num_features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, encoder_output: jnp.ndarray, mask: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
        # x: (batch_size, seq_len, num_features)
        for _ in range(self.num_layers):
            x = DecoderLayer(self.num_heads, self.num_features, self.dropout_rate)(x, encoder_output, mask, eval_mode)
        return x
    

class Transformer(nn.Module):
    """
    Transformer
    """

    num_layers: int
    num_heads: int
    vocal_size: int
    num_features: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
        embedding = Embedding(self.vocal_size, self.num_features)
        encoder = Encoder(self.num_layers, self.num_heads, self.num_features, self.dropout_rate)
        decoder = Decoder(self.num_layers, self.num_heads, self.num_features, self.dropout_rate)
        x = embedding(x)
        y = embedding(y)
        x = encoder(x, mask, eval_mode)
        y = decoder(y, x, mask, eval_mode)
        return y


if __name__ == '__main__':
    # Test
    x = jnp.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = jnp.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    mask = jnp.array([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]], [[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]])  # (2, 1, 3, 3)
    
    model = Transformer(2, 2, 100, 64, 0.1)
    rng = jax.random.PRNGKey(42)
    
    init_rng, rng = jax.random.split(rng)
    params = model.init(init_rng, x, y, mask, eval_mode=True)
    out = model.apply(params, x, y, mask, eval_mode=False, rngs={'dropout': rng})
    print(out.shape)
