import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Grid Creation
def create_grid(n_points=100):
    x = jnp.linspace(0, 1, n_points)
    y = jnp.linspace(0, 1, n_points)

    X, Y = jnp.meshgrid(x, y)

    XY = jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])

    return np.array(X), np.array(Y), XY


# Prediction
def predict_field(params, model, XY, n_points):
    output = model.apply(params, XY)

    output = np.array(output)

    u = output[:, 0].reshape(n_points, n_points)
    v = output[:, 1].reshape(n_points, n_points)
    p = output[:, 2].reshape(n_points, n_points)

    return u, v, p



# Plot Velocity
def plot_velocity(X, Y, u, v):
    plt.figure()

    X = np.array(X)
    Y = np.array(Y)
    u = np.array(u)
    v = np.array(v)

    plt.quiver(X, Y, u, v)
    plt.title("Velocity Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Plot Streamlines
def plot_streamlines(X, Y, u, v):
    plt.figure()

    X = np.array(X)
    Y = np.array(Y)
    u = np.array(u)
    v = np.array(v)

    plt.streamplot(X, Y, u, v)
    plt.title("Streamlines")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Plot Pressure
def plot_pressure(X, Y, p):
    plt.figure()

    X = np.array(X)
    Y = np.array(Y)
    p = np.array(p)

    plt.contourf(X, Y, p, levels=50)
    plt.colorbar()
    plt.title("Pressure Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Main Visualization Function
def visualize(params, model, n_points=100):
    X, Y, XY = create_grid(n_points)

    u, v, p = predict_field(params, model, XY, n_points)

    plot_velocity(X, Y, u, v)
    plot_streamlines(X, Y, u, v)
    plot_pressure(X, Y, p)