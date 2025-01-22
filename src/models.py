
# Equinox and JAX-related imports
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

from src.model_utils import keygen


class ContinuousTimeRNN(eqx.Module):
  params: PyTree[Array]
  tau: int

  def __init__(self, key, n_input, n_hidden, n_output, h0_scale=0.1, wIn_factor=1.0, wRec_factor=0.9, wOut_factor=1.0, tau=1.0):
    key, skeys = keygen(key, 5)
    self.params = {
        'x0': jnp.zeros((n_hidden,)),
        'wIn': jax.random.normal(next(skeys), (n_hidden, n_input)) * wIn_factor,
        'wRec': jax.random.normal(next(skeys), (n_hidden, n_hidden)) *  wRec_factor / jnp.sqrt(n_hidden),
        'bRec': jnp.zeros([n_hidden]),
        'wOut': jax.random.normal(next(skeys), (n_output, n_hidden)) * wOut_factor,
        'bOut': jnp.zeros((n_output,))}
    self.tau = tau


  def __call__(self, input, dt, return_h=True):
    def _nonlinear_function(x_t):
      return jnp.tanh(x_t)

    def _euler_solver(x_tm1, input_t):
      alpha = dt / self.tau
      pre_activation = self.params['wIn'] @ input_t + self.params['wRec'] @ _nonlinear_function(x_tm1) + self.params['bRec']
      x_t = (1-alpha) * x_tm1 + alpha * pre_activation
      return x_t, x_t

    def _linear_readout(h_t):
      return self.params['wOut'] @ h_t + self.params['bOut']

    _, x = jax.lax.scan(_euler_solver, self.params['x0'], input)
    h = jax.vmap(_nonlinear_function)(x)
    output = jax.vmap(_linear_readout)(h)
    if return_h:
      return h, output
    else:
      return x, output