import xgboost as xgb
from xgbatch import score_numpy
import datetime

from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
In another terminal, start the Flight server:

from xgbatch.server import serve_xgb_batch
serve_xgb_batch("127.0.0.1", "8989", "model.model")
"""

# Load some data.
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("scoring remotely...")
start = datetime.datetime.now()
result = score_numpy(X_test, "127.0.0.1", "8989")
end = datetime.datetime.now()
elapsed = end - start
print(result)
print(f"took {elapsed.microseconds} microseconds")

print("scoring xgboost model directly...")
model = xgb.Booster()
# Change to the name of whatever your model is.
model.load_model('model.model')
start = datetime.datetime.now()
result = model.predict(xgb.DMatrix(X_test))
end = datetime.datetime.now()
elapsed = end - start
print(result)
print(f"took {elapsed.microseconds} microseconds")

print("scoring on different data...")
print(score_numpy(X_train, "127.0.0.1", "8989"))
