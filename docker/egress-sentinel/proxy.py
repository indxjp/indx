"""Egress sentinel — a stand-in proxy that records and rejects every outbound request.

Belt-and-suspenders for environments where `network_mode: none` is unavailable
(testing-strategy §4.4c). Point HTTP(S)_PROXY at this process; any attempt to leave the
container is logged to /tmp/egress.log and answered with 599. A conftest then fails the
test if the log is non-empty.
"""

from __future__ import annotations

import http.server
import os

LOG = os.environ.get("EGRESS_LOG", "/tmp/egress.log")


class Sentinel(http.server.BaseHTTPRequestHandler):
    def _trip(self) -> None:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{self.command} {self.path}\n")
        self.send_response(599, "Egress blocked by indx air-gap sentinel")
        self.end_headers()

    do_GET = do_POST = do_CONNECT = do_PUT = do_HEAD = _trip  # type: ignore[assignment]

    def log_message(self, *args: object) -> None:  # silence default logging
        pass


if __name__ == "__main__":
    port = int(os.environ.get("EGRESS_PORT", "8888"))
    http.server.HTTPServer(("0.0.0.0", port), Sentinel).serve_forever()  # noqa: S104
