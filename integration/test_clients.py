import os
import time
import threading
from xgbatch.client import score_pandas

from xgbatch import serve_xgb_batch, score_numpy
from xgbatch import score_table
from .train_model import load_array

import dask.dataframe as dd
import pyarrow.parquet as pq
import pytest
import numpy as np
import pandas as pd


def _run_server(model_uri):
    serve_xgb_batch("127.0.0.1", "8989", model_uri)


def serve_background(model_uri):
    th = threading.Thread(group=None, target=_run_server, args=(model_uri,))
    th.daemon = True
    th.start()
    time.sleep(1)
    return th


def setup_module():
    """
    Starts the server in a separate thread. Should address the
    ergonomics of not being able to easily stop the server in the future.
    """
    pwd = os.path.realpath(os.path.dirname(__file__))
    model_uri = os.path.join(pwd, 'model', 'iris.xgb')
    th = serve_background(model_uri)


def teardown_module(module):
    """
    Boilerplate to make sure pytest ends the tests instead
    of waiting for the server thread to stop.
    """
    pass


def test_numpy(expected_predictions):
    model_input = load_array('xtest')
    scores = score_numpy(model_input, "127.0.0.1", "8989")
    assert np.array_equal(scores, expected_predictions)


def test_dask(expected_predictions):
    pwd = os.path.realpath(os.path.dirname(__file__))
    ddf = dd.read_parquet(os.path.join(pwd, 'data', 'xtest.parquet'))
    ddf = ddf.map_partitions(lambda x: score_pandas(x, "127.0.0.1", "8989"))
    scores = ddf.compute()
    assert np.array_equal(scores, expected_predictions)


def test_pandas(expected_predictions):
    pwd = os.path.realpath(os.path.dirname(__file__))
    df = pd.read_parquet(os.path.join(pwd, 'data', 'xtest.parquet'))
    scores = score_pandas(df, "127.0.0.1", "8989")
    assert np.array_equal(scores, expected_predictions)


def test_arrow(expected_predictions):
    pwd = os.path.realpath(os.path.dirname(__file__))
    table = pq.read_table(os.path.join(pwd, 'data', 'xtest.parquet'))
    scores = score_table(table, "127.0.0.1", "8989").to_pandas()
    assert np.array_equal(scores, expected_predictions)


@pytest.fixture
def expected_predictions():
    pwd = os.path.realpath(os.path.dirname(__file__))
    return np.load(os.path.join(pwd, 'data', 'scores.npy'))
