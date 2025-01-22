# Equinox and JAX-related imports
import equinox as eqx
import jax

# General-purpose libraries
import numpy as np
# File handling and data processing
import os
import json
import hashlib
import pandas as pd
import csv


def keygen(key, nkeys):
  keys = jax.random.split(key, nkeys+1)
  return keys[0], (k for k in keys[1:])

def compile_metadata(config, model_hyperparameters, data_hyperparameters, optimizer_hyperparameters):
    """Collect session metadata."""
    model_hyperparameters = model_hyperparameters.copy()
    data_hyperparameters = data_hyperparameters.copy()
    model_hyperparameters.pop("key")
    data_hyperparameters.pop("key")
    return {
        "config": config,
        "model_hyperparameters": model_hyperparameters,
        "data_hyperparameters": data_hyperparameters,
        "optimizer_hyperparameters": optimizer_hyperparameters
    }

def generate_model_id(metadata):
  model_id = hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
  save_model_metadata(metadata, model_id)
  return model_id

def check_model_exists(model_id, folder_path="saved_models"):
  os.path.join(folder_path, model_id, "model.eqx")
  return os.path.exists("model.eqx")

def save_model_metadata(metadata, model_id, metadata_filename="metadata.json", model_filename="model.eqx", folder_path="saved_models"):
  """
  Save the model and metadata to the specified folder.
  Returns True if metadata already exists, False otherwise.
  """
  save_dir = f"{folder_path}/{model_id}"
  os.makedirs(save_dir, exist_ok=True)

  metadata_path = os.path.join(save_dir, metadata_filename)
  model_path = os.path.join(save_dir, model_filename)
  if os.path.exists(model_path):
    try:
      loss_file_path = os.path.join(save_dir, "loss.csv")
      loss_final_row = pd.read_csv(loss_file_path).iloc[-1]
      loss = loss_final_row["loss"]
      steps = loss_final_row["step"]
    except:
      loss = 0
      steps = 0
    print(f"A model with the same metadata already exists with id: {model_id}. Trained steps = {steps}, Loss = {loss}")

  if os.path.exists(metadata_path):
    pass
  else:
    with open(metadata_path, "w") as f:
      json.dump(metadata, f, indent=4)


def save_model(model_id, model, model_filename="model.eqx", folder_path="saved_models"):
    """
    Save the model and metadata to the specified folder.
    """
    save_dir = f"{folder_path}/{model_id}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, model_filename)
    eqx.tree_serialise_leaves(model_path, model) #Ref: https://docs.kidger.site/equinox/api/serialisation/


def load_model(model_id, model_class, model_filename="model.eqx", metadata_filename="metadata.json", folder_path="saved_models"):
    """
    Load the model and metadata from the specified folder.
    """
    save_dir = f"{folder_path}/{model_id}"
    metadata_path = os.path.join(save_dir, metadata_filename)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    key = jax.random.split(jax.random.PRNGKey(metadata["config"]["seed"]), 2)[1]
    model_hyperparameters = metadata["model_hyperparameters"].copy()
    model_hyperparameters["key"] = key

    model_path = os.path.join(save_dir, model_filename)
    model = eqx.filter_eval_shape(model_class, **model_hyperparameters)
    model = eqx.tree_deserialise_leaves(model_path, model)

    return model

def save_loss(model_id, step, loss, folder_path="saved_models"):
  """
  Saves the loss value into a document "loss.csv"
  """
  save_dir = f"{folder_path}/{model_id}"
  loss_file_path = os.path.join(save_dir, "loss.csv")
  with open(loss_file_path, mode="a", newline="") as loss_file:
    loss_writer = csv.writer(loss_file)
    if step == 0:
      loss_writer.writerow(["step", "loss"])
    loss_writer.writerow([step, loss])

def clear_model_cache(model_id, folder_path="saved_models" ):
  """
  Deletes files created by a previous training run of the same model
  """
  save_dir = f"{folder_path}/{model_id}"
  loss_file_path = os.path.join(save_dir, "loss.csv")
  model_path = os.path.join(save_dir, "model.eqx")
  if os.path.exists(loss_file_path):
    os.remove(loss_file_path)
  if os.path.exists(model_path):
    os.remove(model_path)