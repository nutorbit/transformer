import jax
import haiku as hk
import jax.numpy as jnp


class MultiHeadAttention(hk.Module):

    def __init__(self, num_heads: int, num_features: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        
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
        logits = q @ k.T 
        logits = logits - 1e9 * (1 - mask)
        weights = jax.nn.softmax(logits)
        weights = hk.dropout(rng, self.dropout_rate)(weights)
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
