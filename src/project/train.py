"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior

def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    # Define a jitted step that takes current params/state and an IC batch
    @jit
    def step(params, state, ic_batch):
        def loss_fn(p):
            l_data = data_loss(p, sensor_data, cfg)
            l_ic = ic_loss(p, ic_batch, cfg)
            return cfg.lambda_data * l_data + cfg.lambda_ic * l_ic, (l_data, l_ic)

        (total_val, (l_data_val, l_ic_val)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        new_params, new_state = adam_step(params, grads, state, lr=cfg.learning_rate)

        return new_params, new_state, total_val, l_data_val, l_ic_val

    # Training loop
    for _ in range(cfg.num_epochs):
        ic_epoch, key = sample_ic(key, cfg)

        nn_params, adam_state, l_total, l_data, l_ic = step(nn_params, adam_state, ic_epoch)

        losses["total"].append(l_total)
        losses["data"].append(l_data)
        losses["ic"].append(l_ic)

    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}

def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################

    def loss_fn(p, ic_batch, bc_batch):
        l_data = data_loss(p['nn'], sensor_data, cfg)
        print('l_data rekna')
        l_ic = ic_loss(p['nn'], ic_batch, cfg)
        print('l_ic rekna')
        l_ph = physics_loss(p, ic_batch, cfg)
        print('l_ph rekna')
        l_bc = bc_loss(p, bc_batch, cfg)
        print('l_bc rekna')
        return cfg.lambda_data * l_data + cfg.lambda_ic * l_ic + cfg.lambda_physics*l_ph + cfg.lambda_bc*l_bc, (l_data, l_ic, l_ph, l_bc)
    # Update the nn_params and losses dictionary
    @jit
    def step(params, state, ic_batch, bc_batch):
        # Dette vil anten sprenga RAM-en min vekk og gjera PC-en min ubrukeleg
        # eller gi feil modell, alt etter om eg oppdaterer new_params som ein sideeffekt
        # eller returnerer det. 


        (total_val, (l_data_val, l_ic_val, l_ph_val, l_bc_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ic_batch, bc_batch)


        new_params, new_state = adam_step(params, grads, state, lr=cfg.learning_rate)

        return new_params, new_state, total_val, l_data_val, l_ic_val, l_ph_val, l_bc_val

    # Training loop
    for e in range(cfg.num_epochs):
        print(f'Epoke {e} av {cfg.num_epochs}')
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)

        pinn_params, opt_state, l_total, l_data, l_ic, l_ph, l_bc = step(pinn_params, opt_state, ic_epoch, bc_epoch)

        losses["total"].append(float(l_total))
        losses["data"].append(float(l_data))
        losses["ic"].append(float(l_ic))
        losses["physics"].append(float(l_ph))
        losses["bc"].append(float(l_bc))
    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
