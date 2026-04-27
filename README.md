# 🌊 PINN Fluid 2D — Physics-Informed Neural Network for 2D Navier–Stokes Flow

A minimal but complete implementation of a **Physics-Informed Neural Network (PINN)** that learns to solve the **steady-state 2D incompressible Navier–Stokes equations** without any labelled simulation data. Built with [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/google-deepmind/optax).

---

## 📐 Problem Statement

This project solves the **2D incompressible Navier–Stokes equations** in a unit square domain `[0, 1] × [0, 1]`:

**Continuity (mass conservation):**

```
∂u/∂x + ∂v/∂y = 0
```

**Momentum (x-direction):**

```
u ∂u/∂x + v ∂u/∂y + (1/ρ) ∂p/∂x − ν ∇²u = 0
```

**Momentum (y-direction):**

```
u ∂v/∂x + v ∂v/∂y + (1/ρ) ∂p/∂y − ν ∇²v = 0
```

Where:
- `u`, `v` — velocity components in x and y directions
- `p` — pressure field
- `ρ = 1.0` — fluid density
- `ν = 0.01` — kinematic viscosity

**Boundary Condition:** No-slip condition (`u = 0`, `v = 0`) on the bottom wall (`y = 0`).

---

## 🏗️ Project Structure

```
pinn_fluid_2d/
│
├── model.py          # PINN neural network architecture (Flax/Linen)
├── physics.py        # Navier–Stokes residuals via JAX automatic differentiation
├── utils.py          # Domain and boundary point samplers
├── train.py          # Training loop, loss functions, optimizer, model saving
├── call_visualize.py # Script to load trained params and run visualization
├── visualize.ipynb   # Jupyter notebook with all visualization routines
└── params.pkl        # Saved trained model parameters (generated after training)
```

---

## ⚙️ How It Works

### 1. Neural Network (`model.py`)
A fully-connected feedforward network (`PINN`) takes a 2D coordinate `(x, y)` as input and outputs three values: `[u, v, p]` — the velocity components and pressure at that point.

```
Architecture: [2] → [64] → [64] → [64] → [3]
Activation:   tanh (hidden layers), linear (output layer)
```

### 2. Physics Residuals (`physics.py`)
Instead of training on labelled data, the network is penalised for **violating the governing equations**. JAX's automatic differentiation (`jax.grad`, `jax.hessian`) is used to compute all spatial derivatives analytically from the network output. The residuals of the continuity and momentum equations form the physics loss.

`jax.vmap` vectorises the residual computation efficiently across a batch of collocation points.

### 3. Data Sampling (`utils.py`)
- **Domain points** (`sample_domain`): 1,000 random collocation points drawn uniformly from `[0,1]²` — used for the physics loss.
- **Boundary points** (`sample_boundary`): 200 points on the bottom edge (`y = 0`) with zero-velocity labels — used for the boundary condition loss.

### 4. Training (`train.py`)
The total loss is a sum of:

| Loss Component | Description |
|---|---|
| `loss_fn` | Mean-squared PDE residuals (continuity + x-momentum + y-momentum) |
| `boundary_loss` | MSE between predicted and enforced boundary velocities |

**Optimizer:** Adam (`lr = 1e-3`), **10,000 epochs**, JIT-compiled training step.

Loss is printed every 1,000 epochs. After training, model parameters are saved to `params.pkl`.

### 5. Visualization (`visualize.ipynb`, `call_visualize.py`)
The trained model is evaluated on a uniform `100 × 100` grid over the domain. Three plots are produced:

| Plot | Description |
|---|---|
| **Velocity Field** | Quiver plot of `(u, v)` vectors |
| **Streamlines** | Streamline plot showing flow patterns |
| **Pressure Field** | Filled contour plot of the pressure `p` |

---

## 🚀 Getting Started

### Prerequisites

Install dependencies (Python 3.10+ recommended):

```bash
pip install jax jaxlib flax optax matplotlib numpy
```

> **Note:** For GPU support, install the appropriate `jaxlib` CUDA wheel from the [JAX installation guide](https://github.com/google/jax#installation).

### Training the Model

```bash
python train.py
```

This will:
- Train the PINN for 10,000 epochs
- Print loss every 1,000 steps
- Save the trained parameters to `params.pkl`

**Example output:**
```
Epoch 0,    Loss: 0.4821
Epoch 1000, Loss: 0.0312
Epoch 2000, Loss: 0.0089
...
Epoch 9000, Loss: 0.0021
```

### Visualizing Results

**Option A — Python script:**
```bash
python call_visualize.py
```

**Option B — Jupyter Notebook:**
```bash
jupyter notebook visualize.ipynb
```

Run all cells to generate the velocity field, streamline, and pressure plots interactively.

---

## 📊 Results

After training, the PINN learns to approximate the steady-state flow field satisfying the Navier–Stokes equations. The outputs include:

- **Velocity field** showing the direction and magnitude of fluid flow across the domain
- **Streamlines** tracing fluid particle paths
- **Pressure contours** revealing the pressure distribution driven by the flow

---

## 🧠 Key Concepts

| Concept | Role in this project |
|---|---|
| **PINN** | Neural network trained with physics-based loss instead of labelled data |
| **Collocation points** | Random interior points where PDE residuals are evaluated |
| **Automatic differentiation** | JAX computes exact spatial derivatives of the network |
| **JIT compilation** | `@jax.jit` accelerates the training step |
| **Vectorisation** | `jax.vmap` efficiently batches residual computation |

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `model.py` | Defines the `PINN` Flax module |
| `physics.py` | Computes Navier–Stokes residuals via `jax.grad` / `jax.hessian` |
| `utils.py` | Samples interior collocation and boundary condition points |
| `train.py` | Full training pipeline; saves `params.pkl` |
| `call_visualize.py` | Loads `params.pkl` and calls the visualization routine |
| `visualize.ipynb` | Interactive notebook with grid prediction and all three plots |
| `params.pkl` | Serialised trained model weights (generated by `train.py`) |

---

## 📚 References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** *Journal of Computational Physics*, 378, 686–707.
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
