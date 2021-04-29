import logging
import os
from xgbatch.server import serve_xgb_batch

# Figure out the path to our example iris model.
examples_dir = os.path.realpath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(examples_dir), 'integration', 'model', 'iris.xgb')

# Enable logging.
logger = logging.getLogger("xgbatch")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt="%(message)s", datefmt=None, style='%', validate=True)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Enable tracing. Set TRACING_ENABLED = True to run with tracing.
TRACING_ENABLED = False

# To use it, just run the all-in-one jaeger docker container in another terminal with the following command:
# docker run --rm -p 6831:6831/udp -p 6832:6832/udp -p 16686:16686 jaegertracing/all-in-one:1.7 --log-level=debug
# And in this terminal, install the dependency with pip:
# pip3 install jaeger-python

tracer = None
if TRACING_ENABLED:
    from jaeger_client import Config
    def get_tracer():
        config = Config(
            config={
                'sampler': {
                    'type': 'const',
                    'param': 1
                },
                'logging': True
            },
            service_name='XGBATCH_DEMO',
            validate=True
        )
        return config.initialize_tracer()
    tracer = get_tracer()


# Start the server.
serve_xgb_batch("127.0.0.1", "8989", model_path,
                logger=logger, tracer=tracer)
