"""Transport between the thin client and the daemon.

The container runs Linux, so AF_UNIX is the real path. Some Windows Python
builds ship without socket.AF_UNIX at all, which makes the daemon untestable
locally, so this falls back to a loopback TCP port when the unix family is
missing. The fallback never engages inside the image.
"""

import os
import socket
import time

SOCKET_PATH = os.environ.get("MC3_SOCKET", "/app/run.sock")
TCP_PORT = int(os.environ.get("MC3_TCP_PORT", "8931"))


def use_unix():
    return hasattr(socket, "AF_UNIX")


def client_connect(deadline):
    """Connect, retrying until deadline. Returns a connected socket."""
    last = None
    while True:
        try:
            if use_unix():
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.connect(SOCKET_PATH)
                except OSError:
                    sock.close()
                    raise
            else:
                remaining = max(0.1, deadline - time.time())
                sock = socket.create_connection(("127.0.0.1", TCP_PORT), timeout=remaining)
            return sock
        except OSError as exc:
            last = exc
            if time.time() >= deadline:
                raise last
            time.sleep(0.25)


def server_listen():
    """Bind and listen. Returns a socket ready to accept."""
    if use_unix():
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        parent = os.path.dirname(SOCKET_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", TCP_PORT))
    sock.listen(64)
    return sock


def describe():
    return f"unix:{SOCKET_PATH}" if use_unix() else f"tcp:127.0.0.1:{TCP_PORT}"
