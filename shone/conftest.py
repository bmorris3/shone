import os
import pytest
import numpy as np


@pytest.fixture
def mihalas_h_minus_continuum():
    return np.loadtxt(
        os.path.join(os.path.dirname(__file__), 'data', 'h-_opacity.csv'),
        delimiter=','
    ).T
