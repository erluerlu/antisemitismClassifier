from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class _TeeStream:
    """Mirror writes to console and log file."""

    def __init__(self, console: TextIO, log_file: TextIO):
        self._console = console
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._console.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._console.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._console.isatty()

    def __getattr__(self, name: str):
        return getattr(self._console, name)


@contextmanager
def redirect_output_to_log(log_path: str, mode: str = "a") -> Iterator[None]:
    """Send stdout/stderr to both terminal and UTF-8 log file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode, encoding="utf-8", buffering=1) as log_file:
        header = f"\n[{datetime.now().isoformat(timespec='seconds')}] Logging started\n"
        log_file.write(header)

        import sys

        tee_out = _TeeStream(sys.stdout, log_file)
        tee_err = _TeeStream(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            yield
