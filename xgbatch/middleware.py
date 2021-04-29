import datetime
import json
import uuid

from opentracing import Format, Span
from pyarrow.flight import ServerMiddlewareFactory
from pyarrow.flight import ServerMiddleware


class RequestLoggingMiddlewareFactory(ServerMiddlewareFactory):

    def __init__(self, logger, json_parser=json):
        super().__init__()
        self.logger = logger
        self.json = json_parser

    def start_call(self, info, headers):
        request_id = str(uuid.uuid4())
        return RequestLoggingMiddleware(
            request_id, self.logger, self.json
        )


class RequestLoggingMiddleware(ServerMiddleware):
    def __init__(self, request_id, logger, json_parser):
        super().__init__()
        self.request_id = request_id
        self.logger = logger
        self.json = json_parser

    def sending_headers(self):
        return {
            "x-request-id": self.request_id
        }

    def call_completed(self, exception):
        now = datetime.datetime.now().isoformat()
        code = 0
        desc = "OK"
        msg = ""
        if exception is not None:
            if isinstance(exception, ValueError):
                code = 3
                desc = "INVALID_ARGUMENT"
                msg = f"{exception}"
            else:
                code = 2
                desc = "UNKNOWN"
                msg = f"{exception}"
        msg = self.json.dumps({
            "id": self.request_id,
            "ts": now,
            "code": code,
            "desc": desc,
            "error": msg
        })
        self.logger.info(msg)


class TracingMiddlewareFactory(ServerMiddlewareFactory):

    def __init__(self, tracer, operation_name):
        super().__init__()
        self.tracer = tracer
        self.operation_name = operation_name

    def start_call(self, info, headers):
        span_context = self.tracer.extract(
            format=Format.HTTP_HEADERS,
            carrier=headers
        )
        span: Span = self.tracer.start_span(
            operation_name=f'{self.operation_name}_scoring',
            child_of=span_context
        )
        return TracingMiddleware(span)


class TracingMiddleware(ServerMiddleware):
    def __init__(self, span: Span):
        self.span = span
    
    def sending_headers(self):
        self.span.finish()
        return {}
