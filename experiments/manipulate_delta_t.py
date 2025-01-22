import os, optax
from src.config import get_run_config, get_data_hyperparameters, get_model_hyperparameters, get_optimizer_hyperparameters
from src.data_utils import nbatch_nbit_memory_dataloader, build_nbatch_nbit_memory
from src.model_utils import compile_metadata, generate_model_id, save_model_metadata, load_model, save_model, clear_model_cache
from src.training import train_model
from src.plotting import plot_dts_experiment
from src.models import ContinuousTimeRNN

def run_experiment(experiment_dts, n_neurons=2, n_training_steps=10000, log_every=1000, use_existing_models=False):
    """Run the n-bit memory task experiment."""
    config = get_run_config()
    models = []

    for dt in experiment_dts:
        # Hyperparameters
        data_hyperparameters = get_data_hyperparameters(config, dt=dt)
        model_hyperparameters = get_model_hyperparameters(config)
        optimizer_hyperparameters = get_optimizer_hyperparameters()

        # Compile metadata and generate model ID
        metadata = compile_metadata(config, model_hyperparameters, data_hyperparameters, optimizer_hyperparameters)
        model_id = generate_model_id(metadata)
        save_model_metadata(metadata, model_id)

        # Check if the model already exists
        model_exists = os.path.exists(f"saved_models/{model_id}/model.eqx")
        if use_existing_models and model_exists:
            model = load_model(model_id, ContinuousTimeRNN)
        else:
            clear_model_cache(model_id)
            model = ContinuousTimeRNN(**model_hyperparameters)

        # Data loader
        iter_data = nbatch_nbit_memory_dataloader(**data_hyperparameters)

        # Train the model
        optimizer = optax.adam(**optimizer_hyperparameters)
        opt_state = optimizer.init(model)
        model = train_model(model, iter_data, optimizer, opt_state, model_id=model_id, dt=dt, n_training_steps=n_training_steps, log_every=log_every)
        models.append(model)

    # Final evaluation and plotting
    config["seed"] = 0  # Reset seed for evaluation data
    data_hyperparameters = get_data_hyperparameters(config, dt=1, batch_size=1)
    input_coarse, target_coarse = build_nbatch_nbit_memory(**data_hyperparameters)
    plot_dts_experiment(models, experiment_dts, input_coarse[0], target_coarse[0], T=25, n_neurons=n_neurons, plot_input=True, plot_dynamics=True, plot_output=True)

if __name__ == "__main__":
    experiment_dts = [1, 0.1, 0.01]
    run_experiment(experiment_dts, n_neurons=2, n_training_steps=10000, log_every=1000, use_existing_models=False)
