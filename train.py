import jax
import jax.numpy as jnp
import optax
from physics import residuals
from model import PINN
from utils import sample_domain, sample_boundary
from physics import residuals_batch

# Model (GLOBAL — IMPORTANT)

model = PINN(layers=[2, 64, 64, 64, 3])

# Initialize parameters
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 2)))

def loss_fn(params, x, rho, nu):
    cont, mom_x, mom_y = residuals_batch(
    params,
    model.apply,   # 🔥 THIS IS THE FIX
    x,
    rho,
    nu
)

    return (
        jnp.mean(cont**2)
        + jnp.mean(mom_x**2)
        + jnp.mean(mom_y**2)
    )


def boundary_loss(params, x_bc, u_bc, v_bc):
    output = model.apply(params, x_bc)

    u_pred = output[:, 0:1]
    v_pred = output[:, 1:2]

    return jnp.mean((u_pred - u_bc) ** 2) + jnp.mean((v_pred - v_bc) ** 2)


def total_loss(params, x_f, x_bc, u_bc, v_bc, rho, nu):
    return loss_fn(params, x_f, rho, nu) + boundary_loss(params, x_bc, u_bc, v_bc)



# Optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)



# Training Step (NO model here)
@jax.jit
def train_step(params, opt_state, x_f, x_bc, u_bc, v_bc, rho, nu):

    def loss_wrapper(params):
        return total_loss(params, x_f, x_bc, u_bc, v_bc, rho, nu)

    grads = jax.grad(loss_wrapper)(params)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state



# Data Sampling
x_f = sample_domain(1000)
x_bc, u_bc, v_bc = sample_boundary(200)

# Training Loop
for epoch in range(10000):

    params, opt_state = train_step(
        params,
        opt_state,
        x_f,
        x_bc,
        u_bc,
        v_bc,
        1.0,   # rho
        0.01   # nu
    )

    if epoch % 1000 == 0:
        loss_val = total_loss(
            params,
            x_f,
            x_bc,
            u_bc,
            v_bc,
            1.0,
            0.01
        )
        print(f"Epoch {epoch}, Loss: {loss_val}")
        
import pickle

# assume `params` is your trained model parameters
with open("params.pkl", "wb") as f:
    pickle.dump(params, f)