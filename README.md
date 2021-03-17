# XGBATCH
A high performance microservice framework for XGBoost Models

## What problem we are trying to solve?
Most solutions for productionizing ML have separate deployment methods
for real time model needs (e.g. an API or GRPC service, etc) vs batch scoring 
(e.g. some form of Spark or Dask based solution). This makes sense because
methods that are optimized for one are generally not optimized for the other.
Take Sagemaker as an example. It runs its batch scoring by simply calling your
REST API a million times. This is nice because it's easy to set up, but what
if your model takes a while to make its prediction? If you have a large amount
of data, your batch process ends up being really slow unless you throw a large
amount of parallelism/resources at it. To summarize, real time use cases need
low latency, while batch processes need to take advantage of vectorization
to process through large volumes of data quickly. Enter XGBatch.


