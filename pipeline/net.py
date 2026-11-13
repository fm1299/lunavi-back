"""HTTP with retries, so one failed request in a whole-Earth run does not stop the build."""

import ssl
import time

import certifi
import httpx

_client = httpx.Client(
    follow_redirects=True,
    timeout=120,
    verify=ssl.create_default_context(cafile=certifi.where()),
    limits=httpx.Limits(max_connections=48, max_keepalive_connections=24),
)
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _delay(response: httpx.Response | None, attempt: int) -> float:
    header = response.headers.get("Retry-After") if response is not None else None
    try:
        return float(header)
    except (TypeError, ValueError):
        return float(2**attempt)


def fetch(url: str, method: str = "GET", attempts: int = 6, **kwargs) -> httpx.Response:
    """Send a request, waiting and retrying on rate limits, server errors and network errors."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        response = None
        try:
            response = _client.request(method, url, **kwargs)
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response
            last_error = httpx.HTTPStatusError(
                f"{response.status_code} from {url}", request=response.request, response=response
            )
        except httpx.TransportError as error:
            last_error = error
        time.sleep(_delay(response, attempt))
    raise RuntimeError(f"giving up on {url} after {attempts} attempts") from last_error
