import os
from jax import numpy as jnp, config as jax_config

on_rtd = os.getenv('READTHEDOCS', False)

shone_dir = os.path.expanduser(os.path.join("~", ".shone"))
tiny_archives_dir = os.path.join(shone_dir, 'tiny_archives')
float_dtype = jnp.float64 if jax_config.read('jax_enable_x64') else jnp.float32

if on_rtd:
    # use a temporary directory on readthedocs:
    shone_dir = os.path.abspath('./.')
