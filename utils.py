import jax
import jax.numpy as jnp


def sample_domain(n_points):
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (n_points, 1), minval=0.0, maxval=1.0)
    y = jax.random.uniform(key, (n_points, 1), minval=0.0, maxval=1.0)
    return jnp.hstack([x, y])


def sample_boundary(n_points):
    key = jax.random.PRNGKey(1)

    x = jax.random.uniform(key, (n_points, 1), minval=0.0, maxval=1.0)
    y = jnp.zeros_like(x)

    x_bc = jnp.hstack([x, y])

    u_bc = jnp.zeros((n_points, 1))
    v_bc = jnp.zeros((n_points, 1))

    return x_bc, u_bc, v_bc