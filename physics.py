import jax
import jax.numpy as jnp

# Extract u, v, p
def uvp(params, apply_fn, x):
    output = apply_fn(params, x)
    u = output[:, 0:1]
    v = output[:, 1:2]
    p = output[:, 2:3]
    return u, v, p


# Residuals (Navier–Stokes)

def residuals(params, apply_fn, x, rho, nu):

    def u_fn(x_):
        return uvp(params, apply_fn, x_.reshape(1, -1))[0][0, 0]

    def v_fn(x_):
        return uvp(params, apply_fn, x_.reshape(1, -1))[1][0, 0]

    def p_fn(x_):
        return uvp(params, apply_fn, x_.reshape(1, -1))[2][0, 0]

    # First derivatives
    grad_u = jax.grad(u_fn)(x)
    grad_v = jax.grad(v_fn)(x)
    grad_p = jax.grad(p_fn)(x)

    # Second derivatives
    hess_u = jax.hessian(u_fn)(x)
    hess_v = jax.hessian(v_fn)(x)

    lap_u = hess_u[0, 0] + hess_u[1, 1]
    lap_v = hess_v[0, 0] + hess_v[1, 1]

    # Values
    u = u_fn(x)
    v = v_fn(x)

    # Continuity
    continuity = grad_u[0] + grad_v[1]

    # Momentum
    momentum_x = u * grad_u[0] + v * grad_u[1] + (1 / rho) * grad_p[0] - nu * lap_u
    momentum_y = u * grad_v[0] + v * grad_v[1] + (1 / rho) * grad_p[1] - nu * lap_v

    return continuity, momentum_x, momentum_y

# Vectorize over batch
residuals_batch = jax.vmap(
    residuals,
    in_axes=(None, None, 0, None, None)
)