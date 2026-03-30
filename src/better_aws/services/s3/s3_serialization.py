"""
Serialization, deserialization, and upload-preparation utilities for S3 objects.

This module centralizes all format-related logic used by the public S3 API:
- serialize Python objects before upload
- deserialize S3 payloads after download/load
- prepare upload sources in an execution-friendly representation
- map file extensions to the appropriate pandas / polars / binary handlers

Main goals
-------------
1. Serialization:
   Convert a Python object into bytes or a temporary file representation.

2. Deserialization:
   Convert raw bytes loaded from S3 into a Python object based on file extension
   and output preferences.

3. Upload preparation:
   Decide whether an object should be uploaded as:
   - an existing local file path
   - raw in-memory bytes
   - a temporary file created from serialization
"""

from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path
import tempfile
import pickle
import json
import io

try:
    import joblib
except ImportError:
    joblib = None

try:
    import skops.io as skops_io
except ImportError:
    skops_io = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


PreparedUploadMode = Literal["file_path", "bytes", "temp_file"]
OutputType = Literal["pandas", "polars"]


@dataclass(slots=True)
class PreparedUploadSource:
    """
    Prepared upload source ready to be consumed by the execution engine.

    Attributes
    ----------
    mode:
        Representation mode used by the execution engine.
        Supported values:
        - ``"file_path"``: upload from an existing local file
        - ``"bytes"``: upload directly from in-memory bytes
        - ``"temp_file"``: upload from a temporary file created during
          serialization
    payload:
        The actual payload associated with the selected mode.
        - file path string for ``"file_path"`` and ``"temp_file"``
        - raw bytes for ``"bytes"``
    extension:
        Final extension associated with the prepared payload.
    size_hint:
        Optional payload size in bytes when known.
    cleanup:
        Whether the payload should be cleaned up after upload.
        Typically True for ``"temp_file"``.
    """

    mode: PreparedUploadMode
    payload: str | bytes
    extension: str | None = None
    size_hint: int | None = None
    cleanup: bool = False


def normalize_extension(value: str | None, *, default: str | None = None) -> str | None:
    """
    Normalize a file extension.

    Parameters
    ----------
    value:
        Extension string such as ``"csv"`` or ``".csv"``.
    default:
        Default extension to use if ``value`` is missing.

    Returns
    -------
    str | None
        Normalized extension including the leading dot, or None.
    """
    ext = value or default
    if not ext:
        return None
    ext = str(ext).strip().lower()
    if not ext:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def infer_extension_from_key(key: str) -> str | None:
    """
    Infer a file extension from an S3 key or local path.

    Parameters
    ----------
    key:
        File name, local path, or S3 key.

    Returns
    -------
    str | None
        Lowercase extension including the leading dot, or None.
    """
    suffix = Path(str(key)).suffix.lower()
    return suffix or None


def resolve_extension(
    *,
    key: str | None = None,
    explicit_file_type: str | None = None,
    default_file_type: str | None = None,
) -> str | None:
    """
    Resolve the target file extension for serialization.

    Resolution order:
    1. explicit_file_type
    2. inferred extension from key
    3. default_file_type

    Parameters
    ----------
    key:
        Optional destination key/path.
    explicit_file_type:
        Explicit file type requested by the caller.
    default_file_type:
        Default file type from configuration.

    Returns
    -------
    str | None
        Final normalized extension, or None.
    """
    ext = normalize_extension(explicit_file_type)
    if ext:
        return ext

    ext = infer_extension_from_key(key) if key else None
    if ext:
        return ext

    return normalize_extension(default_file_type)


def is_pandas_dataframe(obj: Any) -> bool:
    """Return whether the object is a pandas DataFrame."""
    return pd is not None and isinstance(obj, pd.DataFrame)


def is_polars_dataframe(obj: Any) -> bool:
    """Return whether the object is a polars DataFrame."""
    return pl is not None and isinstance(obj, pl.DataFrame)


def is_tabular(obj: Any) -> bool:
    """
    Return whether the object is a supported tabular dataframe.

    Supported types:
    - pandas.DataFrame
    - polars.DataFrame
    """
    return is_pandas_dataframe(obj) or is_polars_dataframe(obj)


def _require_pandas() -> None:
    if pd is None:
        raise ImportError("pandas is required for this operation.")


def _require_polars() -> None:
    if pl is None:
        raise ImportError("polars is required for this operation.")


def _serialize_json_bytes(
    obj: Any,
    *,
    encoding: str = "utf-8",
    json_kwargs: dict[str, Any] | None = None,
) -> bytes:
    """
    Serialize an object to JSON bytes.

    Parameters
    ----------
    obj:
        Python object serializable to JSON.
    encoding:
        Output encoding.
    json_kwargs:
        Optional keyword arguments passed to ``json.dumps``.

    Returns
    -------
    bytes
        JSON-encoded payload.
    """
    kwargs = {"ensure_ascii": False}
    if json_kwargs:
        kwargs.update(json_kwargs)
    return json.dumps(obj, **kwargs).encode(encoding)


def _serialize_pickle_bytes(obj: Any, *, protocol: int | None = None) -> bytes:
    """
    Serialize an object to pickle bytes.

    Parameters
    ----------
    obj:
        Python object to pickle.
    protocol:
        Optional pickle protocol.

    Returns
    -------
    bytes
        Pickle payload.
    """
    return pickle.dumps(obj, protocol=protocol)


def _serialize_joblib_bytes(obj: Any, *, compress: int | bool = 0) -> bytes:
    """
    Serialize an object to joblib bytes.

    Parameters
    ----------
    obj:
        Python object to serialize.
    compress:
        Compression level or boolean.

    Returns
    -------
    bytes
        Joblib payload.

    Raises
    ------
    ImportError
        If joblib is not installed.
    """
    if joblib is None:
        raise ImportError("joblib is required for .joblib serialization." \
        "To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")

    bio = io.BytesIO()
    joblib.dump(obj, bio, compress=compress)
    return bio.getvalue()


def _serialize_skops_bytes(obj: Any) -> bytes:
    """
    Serialize an object to skops bytes.

    Parameters
    ----------
    obj:
        Python object to serialize.

    Returns
    -------
    bytes
        skops payload.

    Raises
    ------
    ImportError
        If skops is not installed.
    """
    if skops_io is None:
        raise ImportError("skops is required for .skops serialization." \
        "To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")

    bio = io.BytesIO()
    skops_io.dump(obj, bio)
    return bio.getvalue()


def _serialize_pandas_bytes(
    df,
    *,
    extension: str,
    csv_sep: str = ",",
    csv_index: bool = False,
    parquet_index: bool | None = None,
    excel_index: bool = False,
    encoding: str = "utf-8",
) -> bytes:
    """
    Serialize a pandas DataFrame to bytes according to the target extension.
    """
    _require_pandas()

    if extension == ".csv":
        text = df.to_csv(index=csv_index, sep=csv_sep)
        return text.encode(encoding)

    if extension == ".parquet":
        bio = io.BytesIO()
        df.to_parquet(bio, index=parquet_index)
        return bio.getvalue()

    if extension in {".xlsx", ".xls"}:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=excel_index)
        return bio.getvalue()

    if extension == ".json":
        text = df.to_json(orient="records")
        return text.encode(encoding)

    raise ValueError(f"Unsupported pandas serialization extension: {extension}")


def _serialize_polars_bytes(
    df,
    *,
    extension: str,
    csv_sep: str = ",",
    encoding: str = "utf-8",
) -> bytes:
    """
    Serialize a polars DataFrame to bytes according to the target extension.
    """
    _require_polars()

    if extension == ".csv":
        return df.write_csv(separator=csv_sep).encode(encoding)

    if extension == ".parquet":
        bio = io.BytesIO()
        df.write_parquet(bio)
        return bio.getvalue()

    if extension == ".json":
        return df.write_json().encode(encoding)

    if extension in {".xlsx", ".xls"}:
        bio = io.BytesIO()
        df.write_excel(bio)
        return bio.getvalue()

    raise ValueError(f"Unsupported polars serialization extension: {extension}")


def serialize_object_to_bytes(
    obj: Any,
    *,
    extension: str,
    csv_sep: str = ",",
    csv_index: bool = False,
    parquet_index: bool | None = None,
    excel_index: bool = False,
    encoding: str = "utf-8",
    pickle_protocol: int | None = None,
    joblib_compress: int | bool = 0,
    json_kwargs: dict[str, Any] | None = None,
) -> bytes:
    """
    Serialize a supported Python object to bytes.

    Supported object families:
    - bytes / bytearray
    - str
    - dict / list / JSON-serializable objects via .json
    - pandas DataFrame
    - polars DataFrame
    - generic Python objects via .pkl / .pickle / .joblib / .skops

    Parameters
    ----------
    obj:
        Object to serialize.
    extension:
        Target extension including leading dot.
    csv_sep, csv_index, parquet_index, excel_index, encoding:
        Format-specific options.
    pickle_protocol:
        Optional pickle protocol.
    joblib_compress:
        Optional joblib compression level.
    json_kwargs:
        Optional kwargs passed to json.dumps.

    Returns
    -------
    bytes
        Serialized payload.

    Raises
    ------
    ValueError
        If the object/extension combination is unsupported.
    """
    extension = normalize_extension(extension)
    if extension is None:
        raise ValueError("A target extension is required for serialization.")

    if isinstance(obj, bytes):
        return obj

    if isinstance(obj, bytearray):
        return bytes(obj)

    if isinstance(obj, str):
        if extension in {".txt", ".csv", ".json"}:
            return obj.encode(encoding)
        return obj.encode(encoding)

    if is_pandas_dataframe(obj):
        return _serialize_pandas_bytes(
            obj,
            extension=extension,
            csv_sep=csv_sep,
            csv_index=csv_index,
            parquet_index=parquet_index,
            excel_index=excel_index,
            encoding=encoding,
        )

    if is_polars_dataframe(obj):
        return _serialize_polars_bytes(
            obj,
            extension=extension,
            csv_sep=csv_sep,
            encoding=encoding,
        )

    if extension == ".json":
        return _serialize_json_bytes(
            obj,
            encoding=encoding,
            json_kwargs=json_kwargs,
        )

    if extension in {".pkl", ".pickle"}:
        return _serialize_pickle_bytes(obj, protocol=pickle_protocol)

    if extension == ".joblib":
        return _serialize_joblib_bytes(obj, compress=joblib_compress)

    if extension == ".skops":
        return _serialize_skops_bytes(obj)

    raise ValueError(
        f"Unsupported serialization combination for object type "
        f"{type(obj)!r} and extension {extension!r}, please provide a supported extension or bytes payload."
    )


def write_temp_payload(
    payload: bytes,
    *,
    extension: str | None = None,
) -> tuple[str, int]:
    """
    Write bytes payload to a temporary file.

    Parameters
    ----------
    payload:
        Serialized bytes payload.
    extension:
        Optional extension for the temp file.

    Returns
    -------
    tuple[str, int]
        Temporary file path and payload size in bytes.
    """
    suffix = normalize_extension(extension) or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        path = tmp.name
    return path, len(payload)


def prepare_upload_source(
    obj: Any,
    *,
    key: str | None = None,
    explicit_file_type: str | None = None,
    default_file_type: str | None = None,
    small_payload_threshold: int = 8 * 1024 * 1024,
    csv_sep: str = ",",
    csv_index: bool = False,
    parquet_index: bool | None = None,
    excel_index: bool = False,
    encoding: str = "utf-8",
    pickle_protocol: int | None = None,
    joblib_compress: int | bool = 0,
    json_kwargs: dict[str, Any] | None = None,
) -> PreparedUploadSource:
    """
    Prepare an object for upload according to a simple performance policy.

    Strategy
    --------
    1. Existing local file path:
       returned as ``mode="file_path"``
    2. Object serialized to bytes:
       - if small enough, returned as ``mode="bytes"``
       - otherwise written to a temp file and returned as ``mode="temp_file"``

    Parameters
    ----------
    obj:
        Object to upload.
    key:
        Destination key, used to infer the extension when possible.
    explicit_file_type:
        Explicit requested file type.
    default_file_type:
        Fallback file type from configuration.
    small_payload_threshold:
        Max payload size kept in memory before switching to temp file.
    csv_sep, csv_index, parquet_index, excel_index, encoding:
        Format-specific options.
    pickle_protocol:
        Optional pickle protocol.
    joblib_compress:
        Optional joblib compression level.
    json_kwargs:
        Optional kwargs passed to json.dumps.

    Returns
    -------
    PreparedUploadSource
        Prepared upload representation ready for the execution engine.
    """
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if path.exists() and path.is_file():
            return PreparedUploadSource(
                mode="file_path",
                payload=str(path),
                extension=infer_extension_from_key(str(path)),
                size_hint=path.stat().st_size,
                cleanup=False,
            )

    extension = resolve_extension(
        key=key,
        explicit_file_type=explicit_file_type,
        default_file_type=default_file_type,
    )

    payload = serialize_object_to_bytes(
        obj,
        extension=extension,
        csv_sep=csv_sep,
        csv_index=csv_index,
        parquet_index=parquet_index,
        excel_index=excel_index,
        encoding=encoding,
        pickle_protocol=pickle_protocol,
        joblib_compress=joblib_compress,
        json_kwargs=json_kwargs,
    )

    if len(payload) <= small_payload_threshold:
        return PreparedUploadSource(
            mode="bytes",
            payload=payload,
            extension=extension,
            size_hint=len(payload),
            cleanup=False,
        )

    temp_path, size = write_temp_payload(payload, extension=extension)
    return PreparedUploadSource(
        mode="temp_file",
        payload=temp_path,
        extension=extension,
        size_hint=size,
        cleanup=True,
    )


def deserialize_payload(
    payload: bytes,
    *,
    key: str | None = None,
    file_type: str | None = None,
    output: OutputType = "pandas",
    encoding: str = "utf-8",
    pandas_read_kwargs: dict[str, Any] | None = None,
    polars_read_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Deserialize a raw payload loaded from S3.

    Parameters
    ----------
    payload:
        Raw bytes payload.
    key:
        Optional source key used to infer extension.
    file_type:
        Explicit file type override.
    output:
        Preferred tabular backend: ``"pandas"`` or ``"polars"``.
    encoding:
        Text decoding encoding.
    pandas_read_kwargs:
        Optional kwargs passed to pandas readers.
    polars_read_kwargs:
        Optional kwargs passed to polars readers.

    Returns
    -------
    Any
        Deserialized Python object.

    Notes
    -----
    Fallback behavior:
    - if no extension is known, raw bytes are returned
    - unsupported extensions also fall back to raw bytes
    """
    extension = resolve_extension(
        key=key,
        explicit_file_type=file_type,
        default_file_type=None,
    )

    if extension is None:
        return payload

    pandas_read_kwargs = pandas_read_kwargs or {}
    polars_read_kwargs = polars_read_kwargs or {}

    if extension in {".txt", ".csv", ".json"} and output not in {"pandas", "polars"}:
        return payload.decode(encoding)

    if extension == ".txt":
        return payload.decode(encoding)

    if extension == ".json":
        try:
            return json.loads(payload.decode(encoding))
        except json.JSONDecodeError:
            pass

    bio = io.BytesIO(payload)

    if output == "pandas":
        _require_pandas()

        if extension == ".csv":
            return pd.read_csv(bio, **pandas_read_kwargs)
        if extension == ".parquet":
            return pd.read_parquet(bio, **pandas_read_kwargs)
        if extension in {".xlsx", ".xls"}:
            return pd.read_excel(bio, **pandas_read_kwargs)

    if output == "polars":
        _require_polars()

        if extension == ".csv":
            return pl.read_csv(bio, **polars_read_kwargs)
        if extension == ".parquet":
            return pl.read_parquet(bio, **polars_read_kwargs)
        if extension in {".xlsx", ".xls"}:
            return pl.read_excel(bio, **polars_read_kwargs)

    if extension in {".pkl", ".pickle"}:
        return pickle.loads(payload)

    if extension == ".joblib":
        if joblib is None:
            raise ImportError("joblib is required for .joblib deserialization." \
            "To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")
        return joblib.load(io.BytesIO(payload))

    if extension == ".skops":
        if skops_io is None:
            raise ImportError("skops is required for .skops deserialization." \
            "To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")
        return skops_io.load(io.BytesIO(payload))

    return payload


def cleanup_prepared_upload(prepared: PreparedUploadSource) -> None:
    """
    Cleanup resources associated with a prepared upload source.

    Parameters
    ----------
    prepared:
        Prepared upload source previously returned by
        ``prepare_upload_source``.
    """
    if prepared.cleanup and prepared.mode == "temp_file":
        Path(str(prepared.payload)).unlink(missing_ok=True)