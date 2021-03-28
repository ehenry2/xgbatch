"""
Trains a new model from the iris data and saves
test and train data as parquet for future use.
"""
import os
import numpy

import pyarrow
import pyarrow.parquet as pq
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb


PWD = os.path.realpath(os.path.dirname(__file__))


def gen_names(n):
    return [f"c{i}" for i in range(n)]


def save_array(arr, name):
    outpath = os.path.join(PWD, 'data', f"{name}.parquet")
    # Handle 1D array, e.g. y values.
    if len(arr.shape) == 1:
        table = pyarrow.Table.from_arrays([arr], names=["c0"])
    else:
        table = pyarrow.Table.from_arrays(arr,
                                          names=gen_names(arr.shape[1]))
    pq.write_table(table, outpath)


def load_array(name):
    inpath = os.path.join(PWD, 'data', f"{name}.parquet")  
    table = pq.read_table(inpath)
    arrays = [col.to_numpy() for col in table]
    return numpy.array(arrays).T


def load_data():
    if os.path.exists(os.path.join(PWD, 'data', 'ytrain.parquet')):
        X_train = load_array('xtrain')
        y_train = load_array('ytrain')
        return X_train, y_train
    else:
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)
        save_array(X_train.T, 'xtrain')
        save_array(X_test.T, 'xtest')
        save_array(y_train, 'ytrain')
        save_array(y_test, 'ytest')
        return X_train, y_train


def train_model(X_train, y_train):
    params = {
        'max_depth': 3,
        'eta': 0.3,
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 3
    }
    num_round = 20
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, dtrain, num_round)
    return bst


def main():
    model_out_path = os.path.join(PWD, 'model', 'iris.xgb')
    X_train, y_train = load_data()
    bst = train_model(X_train, y_train)
    bst.save_model(model_out_path)


if __name__ == '__main__':
    main()
