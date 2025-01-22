import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def plot_dts_experiment(models, dts, input_coarse, target_coarse, T, n_neurons, plot_input=True, plot_dynamics=True, plot_output=True):
  """Plot some input/hidden/output triplets."""
  n_rows = int(plot_input) + int(n_neurons) + int(plot_output)
  n_columns = len(models)
  colormap = cm.get_cmap('viridis', len(dts))
  
  fig = plt.figure(figsize=(n_columns*5, n_rows*3))
  input_dim = input_coarse.shape[1]
  
  # Plot Input
  
  for midx in range(len(models)):
    ridx = 0
    plt.subplot(n_rows, n_columns, ridx*n_columns + midx + 1)
    plt.title(f'Model {midx+1}')
    model = models[midx]
    for didx, dt in enumerate(dts):
      color = colormap(didx)
      upsampling_rate = int(1 / dt)
      input_upsampled = np.repeat(input_coarse, upsampling_rate, axis=0)
      h, output = model(input_upsampled, dt)


      # Plot Inputs
      ridx = 0
      if didx == len(dts)-1:
        expander = 3 * np.expand_dims(np.arange(input_dim), axis=0)
        plt.subplot(n_rows, n_columns, ridx*n_columns + midx + 1)
        plt.plot(np.arange(0, T, dt), input_upsampled + expander, 'k')
        plt.xlim([0, T])
        if midx == 0:
          plt.ylabel('Input')
      

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
          plt.ylabel(f'Hidden (nidx)')
  

      # Plot outputs
      ridx += 1
      target_dim = input_dim
      expander = 3 * np.expand_dims(np.arange(target_dim), axis=0)

      target_upsampled = np.repeat(target_coarse, upsampling_rate, axis=0)
      plt.subplot(n_rows, n_columns, ridx * n_columns + midx + 1)
      plt.plot(
          np.arange(0, T, dt), 
          output[:, :] + expander, 
          linestyle='-.', 
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