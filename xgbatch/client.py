import threading

import numpy as np
from pyarrow.flight import FlightClient, FlightDescriptor
import pyarrow

from .common import gen_names


def _in_memory_read(reader, sink):
    table = reader.read_all()
    writer = pyarrow.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()


def score_numpy(arr, host, port):
    """
    Score a 2darray of numpy arrays against a remote
    xgbatch enabled model.

    :param arr:  2d numpy array
    :param host:  remote hostname or ip address
    :param port:  remote port the server is listening on
    
    :returns: a 2d numpy array with the results.
    """
    table = pyarrow.Table.from_arrays(arr.T,
                                      names=gen_names(arr.shape[1]))
    result = score_table(table, host, port)
    return np.array([col.to_numpy() for col in result]).T


def score_pandas(df, host, port):
    """
    Score a pandas dataframe against a remote
    xgbatch enabled model.

    :param df: pandas DataFrame
    :param host:  remote hostname or ip address
    :param port:  remote port the server is listening on
    
    :returns: A pandas DataFrame with the results.
    """
    table = score_table(df.to_pandas(), host, port)
    return table.to_pandas()


def score_table(table, host, port):
    """
    Score a pyarrow table against a remote
    xgbatch enabled model.

    :param table: pyarrow Table
    :param host:  remote hostname or ip address
    :param port:  remote port the server is listening on
    
    :returns: A pyarrow Table with the results.
    """
    sink = pyarrow.BufferOutputStream()
    descriptor = FlightDescriptor.for_command(b"")
    client = FlightClient(f"grpc://{host}:{port}")
    writer, reader = client.do_exchange(descriptor)
    th = threading.Thread(target=_in_memory_read, args=(reader, sink))
    th.start()
    writer.begin(table.schema)
    for batch in table.to_batches():
        writer.write_batch(batch)
        writer.done_writing()
    th.join()
    writer.close()
    buf_reader = pyarrow.RecordBatchStreamReader(sink.getvalue())
    return buf_reader.read_all()
