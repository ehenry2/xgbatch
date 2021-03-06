import threading

from .common import gen_names
from .middleware import RequestLoggingMiddlewareFactory
from .middleware import TracingMiddlewareFactory

import ujson
import numpy as np
import pyarrow
from pyarrow.flight import FlightServerBase
import xgboost as xgb


class ModelServer(FlightServerBase):
    """
    Simple implementation of base flight server for 
    xgboost based model.
    """
    def __init__(self, host, port, model,
                 logger, tracer, model_name):
        """
        Constructor.

        :param host: Host to serve on.
        :param port: Port to serve on.
        :param model: XGBoost booster object.
        :param logger: Logger object to write server logs to.
        :param tracer: Opentracing compatible tracer.
        :param model_name: Name of the model for tracing.
        """
        # Initialize the base class.
        middleware = {}
        if logger is not None:
            middleware['logger'] = RequestLoggingMiddlewareFactory(logger, ujson)
        if tracer is not None:
            middleware['tracer'] = TracingMiddlewareFactory(tracer, model_name)
        location = f"grpc://{host}:{port}"
        super().__init__(location=location, middleware=middleware)
        self.lock = threading.Lock()
        self.model = model

    def _score_data(self, batch):
        # Convert from record batch -> table -> numpy -> to xgb format.
        table = pyarrow.Table.from_batches([batch])
        arr2d = np.array([col.to_numpy() for col in table])
        dmatrix = xgb.DMatrix(arr2d.T)
        # Acquire the lock before scoring, as the predict function is
        # apparently not threadsafe.
        # TODO: Add parameter for wait time to acquire lock/release
        # if the model hangs.
        try:
            self.lock.acquire()
            result = self.model.predict(dmatrix)
        finally:
            self.lock.release()
        table = pyarrow.Table.from_arrays(result.T,
                                          names=gen_names(result.shape[1]))

        return table.to_batches()

    def do_exchange(self, context, descriptor, reader, writer):
        # Process each message.
        is_first_batch = True
        for batch in reader.read_chunk():
            if batch is None:
                break
            result = self._score_data(batch)
            if is_first_batch:
                writer.begin(result[0].schema)
                is_first_batch = False
            for batch in result:
                writer.write_batch(batch)
        writer.close()


def _load_model(model_obj_uri, storage_options):
    opts = storage_options or {}
    bst = xgb.Booster()
    if model_obj_uri.startswith("s3://"):
        # TODO implement for s3.
        pass
    elif model_obj_uri.startswith("gcs://"):
        # TODO implement for gcs.
        pass
    # if no uri prefix, must be a local path
    elif '://' not in model_obj_uri:
        bst.load_model(model_obj_uri)
        return bst
    else:
        raise ValueError("Unsupported file system for model object uri")
        

def serve_xgb_batch(host, port, model_obj_uri,
                    logger=None,
                    storage_options=None, tracer=None,
                    model_name="xgbatch_model"):
    """
    Serve an xgboost model to accept batch inputs via arrow flight.
    This function will load the model artifact and initialize a flight server
    to score against the xgboost model.

    :param host: IP or hostname to serve on. If you want to serve on all
                 interfaces, set it to '0.0.0.0' or if only locally on the
                 loopback interface, use '127.0.0.1'.
    :param port: Port to serve on.
    :param model_obj_uri: URI of the xgboost model object to load. This can
                          be a local path or a remote path to object storage.
                          Use 's3://' prefix for AWS S3 or 'gcs://' for Google Cloud Storage.
                          Model object must have been saved with save_model(), NOT
                          with pickle.
    :param storage_options: Dict of storage backend specific configuration. Corresponds 
                            to the storage_options parameter for s3fs and gcsfs.
    :param logger: A python Logger object compatible with the 'logging' standard library.
                   If logger is not None, a middleware component adding request logging in json format
                   will be enabled on the server.
    :param tracer: An OpenTracing compatible tracer (such as Jaeger). If tracer is not None,
                   a middleware component will be added to enable tracing on the server.
    :param model_name: Name of the model (used in the tracing to specify the operation name).
                       If no tracer is provided, this value is ignored.
    """
    # Load the model object.
    model = _load_model(model_obj_uri, storage_options)
    # Run the service.
    server = ModelServer(host, port, model,
                         logger, tracer, model_name)
    server.serve()
