"""
Glob and pattern utilities for S3 and local file selections.

This module provides a small, explicit glob layer used by the public S3 API
to expand user-facing path patterns into concrete file paths or S3 object keys.

Supported glob syntax
---------------------
- ``*``  : matches zero or more characters within a single path segment
           (does not cross path separators)
- ``?``  : matches exactly one character within a single path segment
           (does not cross path separators)
- ``**`` : matches across path segments recursively

Examples
--------
- ``data/*.csv``       -> files directly under ``data/``
- ``data/**/*.csv``    -> all CSV files recursively under ``data/``
- ``**/*.parquet``     -> all parquet files recursively
- ``raw/*``            -> all direct children under ``raw/``
"""

from typing import Iterable
from pathlib import Path
import re

_GLOB_CHARS = ("*", "?", "[")


def normalize_path_like(value: str) -> str:
    """
    Normalize a local or S3-like path string.

    This helper converts backslashes to forward slashes, removes repeated
    separators, and strips leading ``./`` sequences.

    Parameters
    ----------
    value: str
        Input path or key.

    Returns
    -------
    str
        Normalized path string.
    """
    value = str(value).replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    value = re.sub(r"^\./+", "", value)
    return value


def has_glob(pattern: str) -> bool:
    """
    Return whether a path string contains glob syntax.

    Parameters
    ----------
    pattern: str
        Candidate path or key.

    Returns
    -------
    bool
        True if the pattern contains glob tokens.
    """
    return any(ch in pattern for ch in _GLOB_CHARS)


def split_segments(pattern: str) -> list[str]:
    """
    Split a normalized pattern into path segments.

    Parameters
    ----------
    pattern: str
        Path or glob pattern.

    Returns
    -------
    list[str]
        Path segments.
    """
    pattern = normalize_path_like(pattern).strip("/")
    if not pattern:
        return []
    return pattern.split("/")


def is_recursive_pattern(pattern: str) -> bool:
    """
    Return whether a glob pattern contains ``**``.

    Parameters
    ----------
    pattern: str
        Glob pattern.

    Returns
    -------
    bool
        True if the pattern is recursive.
    """
    return "**" in split_segments(pattern)


def glob_listing_prefix(pattern: str) -> str:
    """
    Derive the widest safe static prefix for S3 listing from a glob pattern.

    The returned prefix is the longest leading part of the pattern that does
    not contain any glob token. It is intended to minimize S3 listing scope
    before regex filtering.

    Examples
    --------
    ``data/**/*.csv``     -> ``data/``
    ``data/raw/*.csv``    -> ``data/raw/``
    ``**/*.parquet``      -> ``""``
    ``foo/bar.csv``       -> ``foo/bar.csv``

    Parameters
    ----------
    pattern: str
        Normalized glob pattern or concrete key.

    Returns
    -------
    str
        Prefix suitable for S3 ``Prefix=...`` listing.
    """
    pattern = normalize_path_like(pattern)
    parts = split_segments(pattern)

    static_parts: list[str] = []
    for part in parts:
        if any(ch in part for ch in _GLOB_CHARS):
            break
        static_parts.append(part)

    if not static_parts:
        return ""

    prefix = "/".join(static_parts)
    if len(static_parts) < len(parts):
        return prefix + "/"
    return prefix


def _translate_segment(segment: str) -> str:
    """
    Translate a single glob path segment into regex.

    Supported tokens inside a segment:
    - ``*`` -> zero or more non-separator characters
    - ``?`` -> exactly one non-separator character

    Parameters
    ----------
    segment: str
        Single path segment.

    Returns
    -------
    str
        Regex fragment.
    """
    out: list[str] = []
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """
    Convert a glob pattern into a compiled regex.

    Semantics:
    - ``*`` matches within a segment only
    - ``?`` matches one character within a segment only
    - ``**`` matches recursively across path segments

    Parameters
    ----------
    pattern: str
        Glob pattern to compile.

    Returns
    -------
    re.Pattern[str]
        Compiled regex matching normalized path strings.
    """
    pattern = normalize_path_like(pattern).strip("/")
    parts = split_segments(pattern)

    if not parts:
        return re.compile(r"^$")

    regex_parts: list[str] = []

    for idx, part in enumerate(parts):
        if part == "**":
            if idx == len(parts) - 1:
                regex_parts.append(r"(?:[^/]+(?:/[^/]+)*)?")
            else:
                regex_parts.append(r"(?:[^/]+/)*")
            continue

        translated = _translate_segment(part)
        regex_parts.append(translated)

        if idx < len(parts) - 1:
            regex_parts.append("/")

    regex = "^" + "".join(regex_parts) + "$"
    return re.compile(regex)


def match_glob(pattern: str, value: str) -> bool:
    """
    Test whether a normalized path/key matches a glob pattern.

    Parameters
    ----------
    pattern: str
        Glob pattern.
    value: str
        Candidate local path or S3 key.

    Returns
    -------
    bool
        True if the value matches the pattern.
    """
    value = normalize_path_like(value).strip("/")
    return bool(glob_to_regex(pattern).match(value))


def expand_local_pattern(pattern: str) -> list[Path]:
    """
    Expand a local filesystem glob pattern into explicit file paths.

    This function uses pathlib glob/rglob semantics for local files but
    normalizes the pattern first so that users can pass either ``/`` or ``\\``.

    Parameters
    ----------
    pattern: str
        Local filesystem path or glob pattern.

    Returns
    -------
    list[Path]
        Sorted list of matching file paths. Existing concrete files are returned
        as a single-item list.
    """
    pattern = normalize_path_like(pattern)

    if not has_glob(pattern):
        path = Path(pattern)
        return [path] if path.exists() else []

    parts = split_segments(pattern)

    static_parts: list[str] = []
    dynamic_parts: list[str] = []
    dynamic_started = False

    for part in parts:
        if not dynamic_started and not any(ch in part for ch in _GLOB_CHARS):
            static_parts.append(part)
        else:
            dynamic_started = True
            dynamic_parts.append(part)

    root = Path("/".join(static_parts)) if static_parts else Path(".")
    subpattern = "/".join(dynamic_parts)

    if not root.exists():
        return []

    matches = [p for p in root.glob(subpattern) if p.is_file()]
    return sorted(matches)


def expand_s3_pattern(
    client,
    *,
    bucket: str,
    pattern: str,
) -> list[str]:
    """
    Expand an S3 glob pattern into explicit object keys.

    The function first derives a safe static prefix, lists objects under that
    prefix, then filters returned keys with the compiled glob regex.

    Parameters
    ----------
    client: boto3 S3 client
        boto3 S3 client.
    bucket: str
        Bucket name.
    pattern: str
        S3 key or S3 glob pattern.

    Returns
    -------
    list[str]
        Sorted list of matching S3 object keys.
    """
    pattern = normalize_path_like(pattern).strip("/")

    if not has_glob(pattern):
        return [pattern]

    prefix = glob_listing_prefix(pattern)
    regex = glob_to_regex(pattern)

    paginator = client.get_paginator("list_objects_v2")
    results: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = normalize_path_like(item["Key"]).strip("/")
            if regex.match(key):
                results.append(key)

    return sorted(results)


def expand_pattern(
    *,
    value: str,
    location: str,
    client=None,
    bucket: str | None = None,
) -> list[str]:
    """
    Expand a local or S3 path/pattern into explicit string paths.

    Parameters
    ----------
    value:
        Path or glob pattern.
    location:
        Either ``"local"`` or ``"s3"``.
    client:
        boto3 S3 client, required for S3 expansion.
    bucket:
        S3 bucket, required for S3 expansion.

    Returns
    -------
    list[str]
        Expanded paths / keys as strings.

    Raises
    ------
    ValueError
        If required arguments are missing or location is unsupported.
    """
    if location == "local":
        return [str(p) for p in expand_local_pattern(value)]

    if location == "s3":
        if client is None:
            raise ValueError("client is required for S3 pattern expansion.")
        if not bucket:
            raise ValueError("bucket is required for S3 pattern expansion.")
        return expand_s3_pattern(client, bucket=bucket, pattern=value)

    raise ValueError(f"Unsupported location: {location!r}")


def common_static_root(pattern: str) -> str:
    """
    Return the static, non-glob root of a pattern.

    This is useful when preserving relative structure during multi-file
    operations such as transfer, upload, or download.

    Examples
    --------
    ``data/raw/*.csv``    -> ``data/raw``
    ``data/**/*.parquet`` -> ``data``
    ``**/*.csv``          -> ``""``

    Parameters
    ----------
    pattern:
        Glob pattern or concrete path.

    Returns
    -------
    str
        Static root without trailing slash.
    """
    prefix = glob_listing_prefix(pattern).rstrip("/")
    return prefix


def relative_to_root(value: str, root: str) -> str:
    """
    Compute a normalized relative path from a static root.

    Parameters
    ----------
    value:
        Concrete path or S3 key.
    root:
        Static root used as reference.

    Returns
    -------
    str
        Relative normalized path.

    Notes
    -----
    If ``root`` is empty, the input value is returned normalized.
    """
    value = normalize_path_like(value).strip("/")
    root = normalize_path_like(root).strip("/")

    if not root:
        return value

    if value == root:
        return ""

    prefix = root + "/"
    if value.startswith(prefix):
        return value[len(prefix):]

    return value


def map_preserving_structure(
    *,
    sources: Iterable[str],
    source_root: str,
    destination_root: str,
) -> list[str]:
    """
    Map source paths/keys to destination paths/keys while preserving structure.

    This helper is useful when:
    - uploading many local files to one S3 prefix
    - downloading many S3 keys to one local directory
    - transferring one S3 subtree to another prefix

    Parameters
    ----------
    sources:
        Concrete source paths or keys.
    source_root:
        Static root used to compute relative paths.
    destination_root:
        Destination root/prefix.

    Returns
    -------
    list[str]
        Destination paths/keys preserving relative structure.
    """
    destination_root = normalize_path_like(destination_root).rstrip("/")
    results: list[str] = []

    for src in sources:
        rel = relative_to_root(src, source_root).lstrip("/")
        if destination_root:
            results.append(f"{destination_root}/{rel}" if rel else destination_root)
        else:
            results.append(rel)

    return results


def ensure_non_empty_selection(
    values: list[str],
    *,
    original_pattern: str,
) -> list[str]:
    """
    Ensure that a selection is not empty.

    Parameters
    ----------
    values:
        Expanded values.
    original_pattern:
        Original user-facing path or glob pattern.

    Returns
    -------
    list[str]
        The input values if non-empty.

    Raises
    ------
    FileNotFoundError
        If no file/key matched the provided pattern.
    """
    if not values:
        raise FileNotFoundError(f"No match found for pattern: {original_pattern}")
    return values