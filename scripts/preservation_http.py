"""Public HF HTTP-client hook: bounded outbound bodies and no silent retries.

Xet must be disabled before SDK import so uploads use this metered HTTP path.
The counter is conservative payload/framing accounting, not a billing meter.
"""
import math
from scripts.preservation_transfer import GateFailure, require


def client_factory(limit, transport=None):
    import httpx
    require(type(limit) is int and limit>0,'positive HTTP byte reservation required')
    class BudgetTransport(httpx.BaseTransport):
        def __init__(self):
            self.inner=transport or httpx.HTTPTransport(retries=0)
            self.used=0
        def consume(self, amount):
            require(self.used+amount<=limit,'outbound HTTP byte reservation exhausted')
            self.used+=amount
        def handle_request(self, request):
            self.consume(8192)  # conservative headers/connection allowance per request
            parent=self
            original=request.stream
            class Stream(httpx.SyncByteStream):
                def __iter__(self):
                    for chunk in original:
                        parent.consume(math.ceil(len(chunk)*1.02))
                        yield chunk
                def close(self):original.close()
            try:
                content=request.content
            except httpx.RequestNotRead:
                request.stream=Stream()
            else:
                self.consume(math.ceil(len(content)*1.02))
            try:
                response=self.inner.handle_request(request)
            except httpx.HTTPError:
                raise GateFailure('HTTP transport failure; diagnosed recovery required') from None
            if response.status_code in {408,429,500,502,503,504}:
                response.close()
                raise GateFailure(f'HTTP transient status {response.status_code}; diagnosed recovery required')
            return response
        def close(self):self.inner.close()
    return lambda: httpx.Client(transport=BudgetTransport(),follow_redirects=True,timeout=60,trust_env=False)


def configure(limit):
    from huggingface_hub import constants,set_client_factory
    require(constants.HF_HUB_DISABLE_XET,'Xet must be disabled before SDK import for bounded transport')
    set_client_factory(client_factory(limit))
