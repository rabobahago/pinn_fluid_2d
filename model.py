import jax
import jax.numpy as jnp
from flax import linen as nn


class PINN(nn.Module):
    layers: list

    @nn.compact
    def __call__(self, x):
        for width in self.layers[:-1]:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.layers[-1])(x)
        return x