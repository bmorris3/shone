import os

on_rtd = os.getenv('READTHEDOCS', False)

shone_dir = os.path.expanduser(os.path.join("~", ".shone"))

if on_rtd:
    # use a temporary directory on readthedocs:
    shone_dir = os.path.abspath('./.')
