from urllib.parse import urlparse

class PolicyError(Exception):
    pass

class PolicyGate:
    """Fail-closed policy gate for the local demonstration."""
    def __init__(self, target_url: str, authorization_confirmed: bool, max_requests: int):
        self.target_url = target_url
        self.max_requests = max_requests
        self.used = 0
        parsed = urlparse(target_url)
        if not authorization_confirmed:
            raise PolicyError("Written/explicit authorization confirmation is required.")
        if parsed.scheme not in ("http", "https"):
            raise PolicyError("Only HTTP(S) localhost targets are allowed.")
        host = (parsed.hostname or "").lower()
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise PolicyError("This MVP only permits localhost targets.")
        if not parsed.port:
            raise PolicyError("Specify an explicit localhost port.")

    def authorize_request(self, url: str):
        if url != self.target_url:
            raise PolicyError("Target URL differs from the approved target.")
        if self.used >= self.max_requests:
            raise PolicyError("Request budget exhausted.")
        self.used += 1
