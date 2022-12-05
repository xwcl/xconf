import logging
import os.path
import threading
from urllib.parse import urlparse
import fsspec

log = logging.getLogger(__name__)

_LOCAL = threading.local()
_LOCAL.filesystems = {}

def join(*args):
    """URL-aware path joining, working equally well on file paths and
    URLs with schemes like https://"""
    res = urlparse(args[0])
    new_path = os.path.join(res.path, *args[1:])
    return args[0].replace(res.path, new_path)


def basename(path):
    """URL-aware path basename, working equally well on file paths and
    URLs with schemes like https://"""
    res = urlparse(path)
    return os.path.basename(res.path)


def get_fs(path):
    """Obtain a concrete fsspec filesystem from a path using
    the protocol string (if any; defaults to 'file:///') and
    `_get_kwargs_from_urls` on the associated fsspec class. The same
    instance will be returned if the same kwargs are used multiple times
    in the same thread.

    Note: Won't work when kwargs are required but not encoded in the
    URL.
    """
    scheme = urlparse(path).scheme
    proto = scheme if scheme != "" else "file"
    cls = fsspec.get_filesystem_class(proto)
    if hasattr(cls, "_get_kwargs_from_urls"):
        kwargs = cls._get_kwargs_from_urls(path)
    else:
        kwargs = {}
    key = (proto,) + tuple(kwargs.items())
    if not hasattr(_LOCAL, "filesystems"):
        _LOCAL.filesystems = (
            {}
        )  # unclear why this is not init at import in dask workers
    if key not in _LOCAL.filesystems:
        fs = cls(**kwargs)
        _LOCAL.filesystems[key] = fs
    return _LOCAL.filesystems[key]

