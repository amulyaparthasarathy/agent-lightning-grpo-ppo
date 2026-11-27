"""Bound SDK 0.40.1 HTTP submissions, including transport retries, in this CLI process."""
from contextlib import contextmanager
from threading import Lock
from unittest.mock import patch

import httpx


@contextmanager
def request_limits(reload, journal):
    # The high-level SDK creates its own HTTPX clients and retries below its public futures.
    # Its pinned transport calls AsyncClient.send for each submission. No request bodies,
    # auth headers, query strings or account inventory enter the journal.
    remaining = dict(asample=4 if reload else 20, forward_backward=0 if reload else 2,
                     optim_step=0 if reload else 2, save_weights_for_sampler=0 if reload else 3,
                     create_model=0 if reload else 1, create_sampling_session=1 if reload else 2,
                     create_session=1)
    total = 100 if reload else 400
    failed = False
    busy = False
    lock = Lock()
    send = httpx.AsyncClient.send

    async def bounded_send(client, request, **kwargs):
        nonlocal total, failed, busy
        route = request.url.path.rsplit("/", 1)[-1]
        counted = route in remaining
        with lock:
            if total <= 0 or (counted and (failed or busy or remaining[route] <= 0)):
                raise RuntimeError("Hosted submission budget exhausted or previous submission failed")
            total -= 1
            if counted:
                remaining[route] -= 1
                journal(route)
                busy = True
        # Do not allow HTTPX to replay a submission at a redirect destination.
        kwargs["follow_redirects"] = False
        success = False
        try:
            response = await send(client, request, **kwargs)
            success = response.is_success
            return response
        finally:
            if counted:
                with lock:
                    failed |= not success
                    busy = False

    with patch.object(httpx.AsyncClient, "send", bounded_send):
        yield
