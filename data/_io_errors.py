"""Shared mapping of low-level OS errors to friendly, path-naming exceptions."""

from __future__ import annotations

import errno
import os
from pathlib import Path

_TEMPLATES: dict[int, tuple[type[OSError], str]] = {
    errno.ENOENT: (FileNotFoundError, "Path not found while {action}: {label}"),
    errno.EISDIR: (IsADirectoryError, "Expected a file, got a directory while {action}: {label}"),
    errno.ENOTDIR: (NotADirectoryError, "Expected a directory while {action}: {label}"),
    errno.EEXIST: (FileExistsError, "A file blocks the directory path while {action}: {label}"),
    errno.EACCES: (PermissionError, "Permission denied while {action}: {label}"),
    errno.EPERM: (PermissionError, "Operation not permitted while {action}: {label}"),
    errno.ENOSPC: (OSError, "Insufficient disk space while {action}: {label}"),
}
_FALLBACK = (OSError, "Filesystem error while {action}: {label}")


def map_os_error(err: OSError, path: str | os.PathLike[str], action: str) -> OSError:
    """Return a typed OSError whose message and .filename both name the path.

    Windows reports opening a directory as a file with EACCES, so EACCES on an existing
    directory is mapped to IsADirectoryError.
    """
    label = os.fspath(path)
    code = err.errno
    if code == errno.EACCES and label and Path(label).is_dir():
        code = errno.EISDIR
    cls, template = _TEMPLATES.get(code or 0, _FALLBACK)
    return cls(code, template.format(action=action, label=label), label)


def describe_os_error(exc: OSError) -> str:
    """One-line description of an OS error that always names the path when the error carries one."""
    detail = exc.strerror or str(exc) or "I/O failure"
    if not exc.filename or str(exc.filename) in detail:
        return detail
    return f"{detail}: {exc.filename}"
