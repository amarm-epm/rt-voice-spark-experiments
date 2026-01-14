"""
Launch the voice assistant with SSL and disabled certificate verification.
This is needed because Gradio's internal httpx client fails on self-signed certs.
"""

import ssl

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
import httpcore

# Patch httpx to use verify=False
_orig_client = httpx.Client

class PatchedClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs["verify"] = False
        super().__init__(*args, **kwargs)

httpx.Client = PatchedClient

# Also patch the module-level functions
_orig_get = httpx.get
_orig_request = httpx.request

def patched_get(url, **kwargs):
    kwargs.pop("verify", None)  # Remove if present, we set it on Client
    with _orig_client(verify=False) as client:
        return client.get(url, **kwargs)

def patched_request(method, url, **kwargs):
    kwargs.pop("verify", None)
    with _orig_client(verify=False) as client:
        return client.request(method, url, **kwargs)

httpx.get = patched_get
httpx.request = patched_request

# Now import and run the app
from src.ui.app import main

if __name__ == "__main__":
    main()
