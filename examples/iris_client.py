import logging
import os
from xgbatch.client import score_pandas
import pandas as pd

# Figure out the path to our example iris dataset.
examples_dir = os.path.realpath(os.path.dirname(__file__))
data_path = os.path.join(os.path.dirname(examples_dir), 'integration', 'data', 'xtest.csv')

# Read the data from csv. Drop the index column as that's
# not part of the data we want to score.
df = pd.read_csv(data_path, index_col=0)
df = df.reset_index(drop=True)

# Run the scoring. Make sure you started the iris_server_with_middleware.py in another terminal first!
output = score_pandas(df, "127.0.0.1", "8989")
print(output)
