import matplotlib.pyplot as plt
import numpy as np 
import jax

def plot_examples(ntimesteps, rnn_internals, key, nexamples=1):
  """Plot some input/hidden/output triplets."""
  nplots = 3
  ridx = 0
  batch_size = rnn_internals['inputs'].shape[0]
  example_idxs = jax.random.randint(key, shape=(nexamples,),minval=0, maxval=batch_size)
  fig = plt.figure(figsize=(nexamples*5, 20))
  input_dim = rnn_internals['inputs'].shape[2]
  expand_val = 3 * np.std(rnn_internals['inputs'][:, :, 0])
  expander = expand_val * np.expand_dims(np.arange(input_dim), axis=0)
  for eidx, bidx in enumerate(example_idxs):
    plt.subplot(nplots, nexamples, ridx*nexamples + eidx + 1)
    plt.plot(rnn_internals['inputs'][bidx, :] + expander, 'k')
    plt.xlim([0, ntimesteps])
    plt.title('Example %d' % (bidx))
    if eidx == 0:
      plt.ylabel('Input')
  ridx += 1

  ntoplot = 10
  closeness = 0.25
  for eidx, bidx in enumerate(example_idxs):
    plt.subplot(nplots, nexamples, ridx * nexamples + eidx + 1)
    plt.plot(rnn_internals['hiddens'][bidx, :, 0:ntoplot] +
               closeness * np.arange(ntoplot), 'm-.')

    plt.xlim([0, ntimesteps])
    if eidx == 0:
      plt.ylabel('Hidden')
  ridx += 1

  ntoplot = 10
  closeness = 0.25

  target_dim = rnn_internals['targets'].shape[2]
  expand_val = 3 * np.std(rnn_internals['targets'][:, :, 0])
  expander = expand_val * np.expand_dims(np.arange(target_dim), axis=0)
  for eidx, bidx in enumerate(example_idxs):
    plt.subplot(nplots, nexamples, ridx * nexamples + eidx + 1)
    plt.plot(rnn_internals['outputs'][bidx, :, :] + expander, 'm-.')
    plt.plot(rnn_internals['targets'][bidx, :, :] + expander, 'k')
    plt.xlim([0, ntimesteps])
    plt.xlabel('Timesteps')
    if eidx == 0:
      plt.ylabel('Output')
  ridx += 1

  plt.plot(0,0, 'm-.', label='RNN output')
  plt.plot(0,0, 'k', label='Target')
  fig.legend(loc=[.91,.1])

def plot_dts_experiment(models, dts, input_coarse, target_coarse, T, n_neurons, model_names=None, plot_input=True, plot_dynamics=True, plot_output=True):
  """Plot some input/hidden/output triplets."""
  n_rows = int(plot_input) + int(n_neurons) + int(plot_output)
  n_columns = len(models)
  colormap = plt.get_cmap('viridis', len(dts))
  
  fig = plt.figure(figsize=(n_columns*5, n_rows*3))
  input_dim = input_coarse.shape[1]
  
  # Plot Input
  
  for midx in range(len(models)):
    ridx = 0
    plt.subplot(n_rows, n_columns, ridx*n_columns + midx + 1)
    plt.title(f'Model {model_names[midx]}')
    model = models[midx]
    for didx, dt in enumerate(dts):
      color = colormap(didx)
      upsampling_rate = int(1 / dt)
      input_upsampled = np.repeat(input_coarse, upsampling_rate, axis=0)
      h, output = model(input_upsampled, dt)


      # Plot Inputs
      ridx = 0
      if didx == midx:
        expander = 3 * np.expand_dims(np.arange(input_dim), axis=0)
        plt.subplot(n_rows, n_columns, ridx*n_columns + midx + 1)
        plt.plot(np.arange(0, T, dt), input_upsampled + expander, 'k')
        plt.xlim([0, T])
        if midx == 0:
          plt.ylabel('Input at training $\Delta t$')
      

      # Plot unit activity
      for nidx in range(n_neurons):
        ridx += 1
        plt.subplot(n_rows, n_columns, ridx * n_columns + midx + 1)
        plt.plot(
                        np.arange(0, T, dt), 
                        h[:, nidx], 
                        linestyle='-', 
                        color=color, 
                        label=f'dt={dt}' if midx == 0 and nidx == 0 else None
                    )
        plt.xlim([0, T])
        if midx == 0:
          plt.ylabel(f'Hidden unit {nidx}')
  

      # Plot outputs
      ridx += 1
      target_dim = input_dim
      expander = 3 * np.expand_dims(np.arange(target_dim), axis=0)

      target_upsampled = np.repeat(target_coarse, upsampling_rate, axis=0)
      plt.subplot(n_rows, n_columns, ridx * n_columns + midx + 1)
      plt.plot(
          np.arange(0, T, dt), 
          output[:, :] + expander, 
          linestyle='-', 
          color=color)
      if didx == len(dts)-1:
        plt.plot(np.arange(0, T, dt), target_upsampled[:, :] + expander, 'k')
      plt.xlim([0, T])
      plt.ylim([np.min(expander)-2, np.max(expander)+2])
      plt.xlabel('Timesteps')
      if midx == 0:
        plt.ylabel('Output')

      # plt.plot(0,0, 'm-.', label='RNN output')
      # plt.plot(0,0, 'k', label='Target')
      fig.legend(loc=[.91,.1])
      
      
def plot_eigenspectra(matrices, matrix_names):
    """
    Plots the eigenspectra of a list of matrices side by side.

    Parameters:
    - matrices (list of np.ndarray): List of matrices to compute eigenspectra for.
    - matrix_names (list of str): List of names for the matrices to be used in plot titles.
    """
    # Validate inputs
    if len(matrices) != len(matrix_names):
        raise ValueError("The number of matrices must match the number of matrix names.")

    num_matrices = len(matrices)
    
    # Create subplots
    fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 6))
    
    # Ensure axes is iterable even for a single plot
    if num_matrices == 1:
        axes = [axes]
    
    # Plot eigenspectra for each matrix
    for i, (matrix, name) in enumerate(zip(matrices, matrix_names)):
        eigenvalues, _ = np.linalg.eig(matrix)
        axes[i].scatter(eigenvalues.real, eigenvalues.imag, color='blue', alpha=0.7)
        axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')  # Imaginary axis
        axes[i].axvline(0, color='black', linewidth=0.8, linestyle='--')  # Real axis
        axes[i].set_title(f"Eigenspectrum of {name}")
        axes[i].set_xlabel("Real Part")
        axes[i].set_ylabel("Imaginary Part")
        axes[i].grid(alpha=0.3)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()