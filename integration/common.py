import os

import numpy as np
import pytest


@pytest.fixture
def expected_predictions():
    pwd = os.path.realpath(os.path.dirname(__file__))
    return np.load(os.path.join(pwd, 'data', 'scores.npy'))
