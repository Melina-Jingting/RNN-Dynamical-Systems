# Equinox and JAX-related imports
import equinox as eqx
import jax
import jax.numpy as jnp
from src.model_utils import save_loss, save_model


# File handling and data processing
import os
import pandas as pd

@eqx.filter_value_and_grad
def compute_loss(model, input, target, dt):
    hs, outputs = jax.vmap(model, in_axes=(0, None))(input, dt)
    return jnp.mean((outputs - target)**2)

@eqx.filter_jit
def make_step(model, optim, input, target, dt, opt_state):
    loss, grads = compute_loss(model, input, target, dt)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model(model, iter_data, optim, opt_state, model_id, dt=0.1, n_training_steps=10000, log_every=1000):
    """Run the training loop."""

    save_dir = f"saved_models/{model_id}"
    loss_file_path = os.path.join(save_dir, "loss.csv")

    # Continue from previous checkpoint if trained before
    step = 0
    if os.path.exists(loss_file_path):
      losses = pd.read_csv(loss_file_path)
      step = losses.iloc[-1]["step"]

    for step, (input, target) in zip(range(n_training_steps), iter_data):
        loss, model, opt_state = make_step(model, optim, input, target, dt, opt_state)
        loss = loss.item()
        if step % log_every == 0:
            print(f"step={step}, loss={loss}")
            save_loss(model_id, step, loss)
            save_model(model_id, model)

    return model