"""
Module to enable cryptographic hashing of pickle files or dill files.
The idea is that in normal use, users can verify that a known key was used to
sign the files. If not, user has to actively opt in to running code from an
untrusted source.

We want this security check on any recipes being unpickled.
"""

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any

import dill

try:
    from platformdirs import user_config_path
except ImportError:
    user_config_path = None  # type: ignore[assignment]

_MAGIC_HEADER = b"MASS2DL\x01"
_DIGEST_SIZE = 32  # SHA-256 produces 32 raw bytes
_HEADER_SIZE = len(_MAGIC_HEADER) + _DIGEST_SIZE


class UntrustedPayloadError(RuntimeError):
    """Raised when attempting to load an unsigned or invalidly signed dill payload."""


def default_config_dir(app_name: str = "mass2") -> Path:
    """Return the standard per-user configuration directory for mass2."""
    if user_config_path is not None:
        return user_config_path(app_name, ensure_exists=True)
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base / app_name
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _default_signing_key_path(app_name: str = "mass2") -> Path:
    """Return the path to the user's primary local HMAC signing key."""
    return default_config_dir(app_name) / "hmac_signing.key"


def _default_trusted_keys_dir(app_name: str = "mass2") -> Path:
    """Return the directory holding additional trusted lab/instrument keys."""
    trusted_dir = default_config_dir(app_name) / "trusted_keys"
    trusted_dir.mkdir(parents=True, exist_ok=True)
    return trusted_dir


def get_or_create_signing_key(key_path: str | Path | None = None) -> bytes:
    """Load the local 32-byte HMAC key, creating it with 0600 permissions if absent."""
    path = Path(key_path) if key_path is not None else _default_signing_key_path()
    if path.exists():
        return path.read_bytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)
    except FileExistsError:
        return path.read_bytes()
    return key


def iter_verification_keys(
    key_path: str | Path | None = None,
    trusted_keys_dir: str | Path | None = None,
) -> list[bytes]:
    """Return all candidate keys: primary signing key first, then any *.key files in trusted_keys/."""
    keys: list[bytes] = []
    seen: set[bytes] = set()

    primary_path = Path(key_path) if key_path is not None else _default_signing_key_path()
    if primary_path.exists():
        primary_key = primary_path.read_bytes()
        if primary_key:
            keys.append(primary_key)
            seen.add(primary_key)

    extra_dir = Path(trusted_keys_dir) if trusted_keys_dir is not None else _default_trusted_keys_dir()
    if extra_dir.is_dir():
        for candidate in sorted(extra_dir.glob("*.key")):
            if candidate.is_file():
                data = candidate.read_bytes()
                if data and data not in seen:
                    keys.append(data)
                    seen.add(data)

    return keys


def _compute_hmac(key: bytes, payload: bytes) -> bytes:
    """Compute a 32-byte HMAC-SHA256 digest over header + payload."""
    return hmac.digest(key, _MAGIC_HEADER + payload, hashlib.sha256)


def dumps_signed(obj: Any, *, key_path: str | Path | None = None, **dill_kwargs: Any) -> bytes:
    """Serialize `obj` with dill and prepend an HMAC-SHA256 signature using the primary key."""
    key = get_or_create_signing_key(key_path)
    payload = dill.dumps(obj, **dill_kwargs)
    mac = _compute_hmac(key, payload)
    return _MAGIC_HEADER + mac + payload


def loads_signed(
    blob: bytes,
    *,
    allow_untrusted: bool = False,
    key_path: str | Path | None = None,
    trusted_keys_dir: str | Path | None = None,
    **dill_kwargs: Any,
) -> Any:
    """Verify HMAC-SHA256 signature against all trusted keys and deserialize `blob`.

    Checks the user's primary signing key first, followed by any `*.key` files
    in the `trusted_keys/` configuration subdirectory.
    """
    has_header = len(blob) >= _HEADER_SIZE and blob[: len(_MAGIC_HEADER)] == _MAGIC_HEADER

    if not has_header:
        if allow_untrusted:
            return dill.loads(blob, **dill_kwargs)
        raise UntrustedPayloadError(
            "Refusing to load unsigned dill payload because it may execute arbitrary code. "
            "If you trust the source of this file, pass `allow_untrusted=True`."
        )

    expected_mac = blob[len(_MAGIC_HEADER) : _HEADER_SIZE]
    payload = blob[_HEADER_SIZE:]

    for candidate_key in iter_verification_keys(key_path=key_path, trusted_keys_dir=trusted_keys_dir):
        actual_mac = _compute_hmac(candidate_key, payload)
        if hmac.compare_digest(actual_mac, expected_mac):
            return dill.loads(payload, **dill_kwargs)

    if allow_untrusted:
        return dill.loads(payload, **dill_kwargs)

    raise UntrustedPayloadError(
        "HMAC signature verification failed against your local signing key and all keys in "
        f"`{trusted_keys_dir or _default_trusted_keys_dir()}`. "
        "Copy the creator's `.key` file into `trusted_keys/` or pass `allow_untrusted=True`."
    )


def dump_signed(obj: Any, filename: str | Path, *, key_path: str | Path | None = None, **dill_kwargs: Any) -> None:
    """Write a signed dill payload to `filename`."""
    Path(filename).write_bytes(dumps_signed(obj, key_path=key_path, **dill_kwargs))


def load_signed(
    filename: str | Path,
    *,
    allow_untrusted: bool = False,
    key_path: str | Path | None = None,
    trusted_keys_dir: str | Path | None = None,
    **dill_kwargs: Any,
) -> Any:
    """Read and verify a signed dill payload from `filename`."""
    return loads_signed(
        Path(filename).read_bytes(),
        allow_untrusted=allow_untrusted,
        key_path=key_path,
        trusted_keys_dir=trusted_keys_dir,
        **dill_kwargs,
    )


def install_trusted_key(
    source_path: str | Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
    trusted_keys_dir: str | Path | None = None,
) -> Path:
    """Install a lab/instrument HMAC key into the user's trusted_keys directory.

    Parameters
    ----------
    source_path
        Path to the key file copied from another machine (e.g. `hmac_signing.key`
        or `my-daq-computer.key`).
    name
        Optional filename to save under in `trusted_keys/` (e.g. `"my-daq-computer"`
        or `"my-daq-computer.key"`). Defaults to `source_path.name`.
    overwrite
        If False (default), raises FileExistsError when a key with the same
        destination name already exists.
    trusted_keys_dir
        Override the target directory (defaults to the standard `mass2` config dir).

    Returns
    -------
    Path
        The installed key path inside `trusted_keys/`.
    """
    src = Path(source_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Key file not found: {src}")

    key_bytes = src.read_bytes()
    if len(key_bytes) != 32:
        raise ValueError(
            f"Expected a 32-byte HMAC key file, got {len(key_bytes)} bytes from {src}. "
            "Make sure you are installing an `hmac_signing.key` file and not a recipe file."
        )

    dest_dir = Path(trusted_keys_dir) if trusted_keys_dir is not None else _default_trusted_keys_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_name = name if name is not None else src.name
    if not dest_name.endswith(".key"):
        dest_name = f"{dest_name}.key"

    # Prevent path traversal if a user passes something like "../foo" as `name`
    dest_path = dest_dir / Path(dest_name).name

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Trusted key `{dest_path.name}` already exists at {dest_path}. Pass `overwrite=True` or provide a distinct `name=`."
        )

    # Write with restrictive 0600 permissions before writing secret bytes
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    fd = os.open(dest_path, flags, 0o600)
    try:
        os.write(fd, key_bytes)
    finally:
        os.close(fd)
    os.chmod(dest_path, 0o600)

    return dest_path


def list_trusted_keys(
    key_path: str | Path | None = None,
    trusted_keys_dir: str | Path | None = None,
) -> list[Path]:
    """Return paths to all active verification keys (primary key + installed trusted keys)."""
    paths: list[Path] = []
    primary = Path(key_path) if key_path is not None else _default_signing_key_path()
    if primary.is_file():
        paths.append(primary)

    extra_dir = Path(trusted_keys_dir) if trusted_keys_dir is not None else _default_trusted_keys_dir()
    if extra_dir.is_dir():
        paths.extend(sorted(p for p in extra_dir.glob("*.key") if p.is_file()))
    return paths
