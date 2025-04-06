import os
from jax import numpy as jnp

on_rtd = os.getenv('READTHEDOCS', False)

shone_dir = os.path.expanduser(os.path.join("~", ".shone"))
tiny_archives_dir = os.path.join(shone_dir, 'tiny_archives')
float_dtype = jnp.float64

if on_rtd:
    # use a temporary directory on readthedocs:
    shone_dir = os.path.abspath('./.')
