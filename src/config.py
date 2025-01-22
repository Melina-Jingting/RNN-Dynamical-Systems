import jax 
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SAVED_MODELS_FOLDER_SUBPATH = "saved_models"

def get_run_config(seed=42, n_bits=3):
    """Initialize random keys and other run-specific settings."""
    return {
        "seed": seed,
        "n_bits": n_bits
    }

def get_data_hyperparameters(config, T=25, dt=0.1, p_ticks=0.333, batch_size=256):
    """Define hyperparameters for data preprocessing and loader."""
    n_timesteps = int(T / dt)
    data_key = jax.random.split(jax.random.PRNGKey(config["seed"]), 2)[0]
    return {
        "n_bits": config["n_bits"],
        "n_coarse_steps": T,
        "upsampling_rate": int(1 / dt),
        "p_ticks": p_ticks,
        "batch_size": batch_size,
        "key": data_key,
    }

def get_model_hyperparameters(config, n_hidden=100, h0_scale=0.1, wIn_factor=1.0, wRec_factor=0.9, wOut_factor=1.0, tau=1.0):
    """Define hyperparameters for the model."""
    model_key = jax.random.split(jax.random.PRNGKey(config["seed"]), 2)[1]
    return {
        "key": model_key,
        "n_input": config["n_bits"],
        "n_hidden": n_hidden,
        "n_output": config["n_bits"],
        "h0_scale": h0_scale,
        "wIn_factor": wIn_factor,
        "wRec_factor": wRec_factor,
        "wOut_factor": wOut_factor,
        "tau": tau
    }

def get_optimizer_hyperparameters(learning_rate=3e-3):
    """Define training hyperparameters."""
    return {
        "learning_rate": learning_rate,
    }