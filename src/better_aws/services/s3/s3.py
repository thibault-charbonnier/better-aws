from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from rich.console import Console
from pathlib import Path
import pandas as pd
import polars as pl
import pickle

from .s3_erros import _err_code, _raise_s3
from .s3_exec import S3ExecutionEngine
from .s3_pattern import (
    common_static_root,
    ensure_non_empty_selection,
    expand_pattern,
    has_glob,
    map_preserving_structure,
    normalize_path_like,
)
from .s3_planner import (
    build_delete_plan,
    build_download_plan,
    build_transfer_plan,
    build_load_plan,
    build_upload_plan
)
from .s3_serialization import (
    PreparedUploadSource,
    deserialize_payload,
    is_tabular,
    normalize_extension,
    prepare_upload_source,
    resolve_extension,
)
from .s3_tree import _build_tree_from_objects, _compute_folder_sizes, _render_tree, list_s3_objects

TabularOutput = Literal["pandas", "polars"]
ObjectFormat = Literal["pickle", "joblib", "skops"]
KeyLike = Union[str, Sequence[str]]
PathLike = Union[str, Path]
TabularFileType = Literal[
    "csv", "parquet", "xlsx", "xls", "pkl", "pickle", "joblib", "jl", "skops"
]
UploadInput = Union[pd.DataFrame, pl.DataFrame, dict, PathLike, bytes, bytearray, Any]

_UNSAFE_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".jl", ".skops"}


class S3:
    """
    Class responsible for S3 interactions using boto3 client.

    It provides the following methods:
    - list: List objects in an S3 bucket with optional filtering and metadata.
    - exists: Check if one or more keys exist in the bucket.
    - delete: Delete one or more keys from the bucket.
    - download: Download one or more keys to a local path.
    - load: Load one or more keys into Python objects.
    - upload: Upload one or more local files or Python objects to the bucket.
    - transfer: Transfer one or more files between local paths and S3.
    - tree: Pretty-print an S3 prefix as a tree structure.
    """

    def __init__(self, aws):
        self.aws = aws
        self.logger = aws.logger.getChild("s3")
        self.client_cache = None
        self.engine_cache = None
        self.bucket = None
        self.key_prefix = None
        self.output_type = None
        self.file_type = None
        self.overwrite = None
        self.encoding = None
        self.csv_sep = None
        self.csv_index = None
        self.parquet_index = None
        self.excel_index = None
        self.allow_unsafe_serialization = None
        self.pickle_protocol = None
        self.joblib_compress = None
        self.object_base_format = None
        self.small_payload_threshold = None
        self.transfer_config = None
        self.delete_batch_size = None

    # --------------------------------------------------------
    # |                  Internal Helpers                    |
    # --------------------------------------------------------

    def _resolve_bucket(self, bucket: Optional[str]) -> str:
        """
        Resolve the bucket name to use for an operation, checking the provided
        bucket parameter, and the instance's bucket override.

        Parameters:
        -----------
        bucket: optional str
            The bucket name provided directly to the method call, which takes
            precedence over the instance's default bucket.

        Returns:
        --------
        str
            The resolved bucket name to use for the S3 operation.
        """
        b = bucket or self.bucket
        if not b:
            raise ValueError("Bucket is not set. Provide bucket=... or configure a default bucket.")
        return b

    def _get_client(self):
        """
        Create and return a boto3 S3 client using the AWS session and configuration.

        Returns:
        --------
        boto3.client
            A configured S3 client for making API calls to S3.
        """
        if self.client_cache is None:
            self.client_cache = self.aws._session().client("s3", config=self.aws._config())
        return self.client_cache

    def _get_engine(self) -> S3ExecutionEngine:
        """
        Returns a S3ExecutionEngine instance.
        If the engine is not already created and cached, it initializes a new one with the current configuration.
        
        Returns
        -------
        S3ExecutionEngine
            An instance of S3ExecutionEngine for executing transfer plans.
        """
        if self.engine_cache is None:
            self.engine_cache = S3ExecutionEngine(
                client=self._get_client(),
                transfer_config=self.transfer_config,
                prepare_upload_source=self._prepare_upload_source_for_get_engine,
                delete_batch_size=self.delete_batch_size,
            )
        return self.engine_cache

    def _normalize_keys(self, key: KeyLike) -> List[str]:
        """
        Normalize one or many S3 keys into a list of prefixed, normalized keys.

        This helper:
        - accepts either a scalar key or a sequence of keys
        - normalizes path separators
        - strips leading slashes
        - applies the configured key_prefix when needed
        - avoids duplicating the prefix if it is already present

        Parameters
        ----------
        key : str or sequence of str
            One or many S3 keys.

        Returns
        -------
        list[str]
            Normalized S3 keys.
        """
        keys = list(key) if isinstance(key, (list, tuple)) else [key]

        prefix = normalize_path_like(self.key_prefix).strip("/") if self.key_prefix else ""

        out: List[str] = []
        for k in keys:
            kk = normalize_path_like(str(k)).lstrip("/")

            if prefix:
                if kk == prefix or kk.startswith(prefix + "/"):
                    out.append(kk)
                else:
                    out.append(f"{prefix}/{kk}" if kk else prefix)
            else:
                out.append(kk)

        return out

    def _normalize_s3_prefix(self, prefix: str = "") -> str:
        """
        Normalize an S3 prefix for listing-style operations.

        This helper:
        - normalizes path separators
        - strips leading slashes
        - applies the configured key_prefix when needed
        - avoids duplicating the prefix if it is already present

        Unlike key resolution helpers, this function does not expand glob patterns
        and does not return a list. It is intended for prefix-based operations such
        as ``list()`` and ``tree()``.

        Parameters
        ----------
        prefix : str, optional
            Prefix to normalize.

        Returns
        -------
        str
            Normalized S3 prefix.
        """
        pref = normalize_path_like(str(prefix)).lstrip("/")

        if self.key_prefix:
            base = normalize_path_like(self.key_prefix).strip("/")
            if pref == base or pref.startswith(base + "/"):
                return pref
            return f"{base}/{pref}" if pref else base

        return pref
    
    def _resolve_s3_keys(
        self,
        key: KeyLike,
        *,
        bucket: str,
        require_match: bool = True,
    ) -> List[str]:
        """
        Resolve S3 keys according to the following rules:
            1. Normalize scalar or list input into a list of keys.
            2. Normalize the path (handle slashes, backslashes, redundant separators).
            3. Add S3 prefix is needed.
            4. Resolve glob patterns and expand to matching keys in S3.

        Parameters
        ----------
        key : str or sequence of str
            One or many S3 keys or patterns to resolve.
        bucket : str
            The S3 bucket to resolve keys in.
        require_match : bool
            If True, require that glob patterns match at least one key in S3. If False, allow patterns that match no keys.

        Returns
        -------
        list[str]
            A list of resolved S3 keys matching the input patterns and normalization rules.
        """
        keys_in = list(key) if isinstance(key, (list, tuple)) else [key]

        normalized: List[str] = []
        prefix = normalize_path_like(self.key_prefix).strip("/") if self.key_prefix else ""

        for k in keys_in:
            kk = normalize_path_like(str(k)).lstrip("/")
            if prefix:
                if kk == prefix or kk.startswith(prefix + "/"):
                    normalized.append(kk)
                else:
                    normalized.append(f"{prefix}/{kk}" if kk else prefix)
            else:
                normalized.append(kk)

        resolved: List[str] = []
        client = self._get_client()

        for kk in normalized:
            matches = expand_pattern(
                value=kk,
                location="s3",
                client=client,
                bucket=bucket,
            )
            if has_glob(kk):
                if require_match:
                    ensure_non_empty_selection(matches, original_pattern=kk)
                resolved.extend(matches)
            else:
                resolved.extend(matches or [kk])

        return list(dict.fromkeys(resolved))

    def _resolve_local_paths(
        self,
        value,
        *,
        require_match: bool = True,
    ) -> List[str]:
        """
        Resolve local file paths according to the following rules:
            1. Normalize scalar or list input into a list of paths.
            2. Normalize the path (handle slashes, backslashes, redundant separators).
            3. Resolve glob patterns and expand to matching file paths.
        
        Parameters
        ----------
        value : str or sequence of str
            One or many local file paths or patterns to resolve.
        require_match : bool
            If True, require that glob patterns match at least one file. If False, allow patterns that match no files.

        Returns
        -------
        list[str]
            A list of resolved local file paths matching the input patterns and normalization rules.
        """
        values = list(value) if isinstance(value, (list, tuple)) else [value]

        resolved: List[str] = []
        for v in values:
            vv = normalize_path_like(str(v))
            matches = expand_pattern(value=vv, location="local")

            if has_glob(vv):
                if require_match:
                    ensure_non_empty_selection(matches, original_pattern=vv)
                resolved.extend(matches)
            else:
                if matches:
                    resolved.extend(matches)
                else:
                    path = Path(vv)
                    if require_match and not path.exists():
                        raise FileNotFoundError(str(path))
                    resolved.append(str(path))

        return list(dict.fromkeys(str(p) for p in resolved))

    def _resolve_upload_items(
        self,
        *,
        src: Union[UploadInput, Sequence[UploadInput]],
        key: KeyLike,
        bucket: str,
        overwrite: bool,
    ) -> List[Dict[str, Any]]:
        """
        Resolve upload inputs into planner-ready items.

        This helper converts the user-facing upload inputs into a flat list of
        dictionaries of the form:

            {"src": <local path or PreparedUploadSource>, "key": <final s3 key>}

        It handles two main cases:
        - local file paths / local glob patterns
        - Python objects that must be serialized/prepared before upload

        Parameters
        ----------
        src : UploadInput or sequence of UploadInput
            Source object(s) to upload.
        key : str or sequence of str
            Destination S3 key(s).
        bucket : str
            Target S3 bucket, used for overwrite checks.
        overwrite : bool
            Whether existing destination objects may be overwritten.

        Returns
        -------
        list[dict[str, Any]]
            Planner-ready upload items.

        Raises
        ------
        ValueError
            If the src/key cardinalities are inconsistent.
        FileNotFoundError
            If a local file path does not exist.
        """
        srcs = list(src) if isinstance(src, (list, tuple)) else [src]
        keys = self._normalize_keys(key)

        items: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Case 1: local file paths / glob patterns
        # ------------------------------------------------------------------
        if all(isinstance(s, (str, Path)) for s in srcs):
            resolved_sources: List[str] = []
            source_roots: List[str] = []
            any_glob = False

            for s in srcs:
                s_str = normalize_path_like(str(s))
                matches = expand_pattern(value=s_str, location="local")

                if has_glob(s_str):
                    any_glob = True
                    ensure_non_empty_selection(matches, original_pattern=s_str)
                    root = common_static_root(s_str)
                else:
                    path = Path(s_str)
                    if not path.exists():
                        raise FileNotFoundError(str(path))
                    matches = [str(path)]
                    root = str(path.parent)

                for m in matches:
                    resolved_sources.append(str(m))
                    source_roots.append(root)

            if len(keys) == 1 and (len(resolved_sources) > 1 or any_glob):
                dst_root = keys[0].rstrip("/")
                final_keys = [
                    self._normalize_keys(
                        map_preserving_structure(
                            sources=[src_item],
                            source_root=src_root,
                            destination_root=dst_root,
                        )[0]
                    )[0]
                    for src_item, src_root in zip(resolved_sources, source_roots, strict=True)
                ]
            elif len(keys) == len(resolved_sources):
                final_keys = [self._normalize_keys(k)[0] for k in keys]
            else:
                raise ValueError(
                    "src and key must have the same length, unless a single destination "
                    "prefix is provided for multiple local files."
                )

            if not overwrite:
                for k in final_keys:
                    if self.exists(k, bucket=bucket):
                        raise ValueError(f"Refusing to overwrite existing object: s3://{bucket}/{k}")

            return [{"src": src_item, "key": final_key} for src_item, final_key in zip(resolved_sources, final_keys, strict=True)]

        # ------------------------------------------------------------------
        # Case 2: Python objects
        # ------------------------------------------------------------------
        if len(srcs) == 1 and len(keys) > 1:
            srcs = srcs * len(keys)

        if len(srcs) != len(keys):
            raise ValueError(
                f"src and key must have the same length (got {len(srcs)} vs {len(keys)})."
            )

        if not overwrite:
            for k in keys:
                if self.exists(k, bucket=bucket):
                    raise ValueError(f"Refusing to overwrite existing object: s3://{bucket}/{k}")

        for obj, raw_key in zip(srcs, keys, strict=True):
            normalized_key = self._normalize_keys(raw_key)[0]

            ext = self._effective_extension(obj=obj, key=normalized_key)
            self._guard_unsafe_extension(ext)

            prepared = prepare_upload_source(
                obj,
                key=normalized_key,
                explicit_file_type=None,
                default_file_type=(
                    self.file_type
                    if is_tabular(obj)
                    else self._object_default_extension().lstrip(".")
                ),
                small_payload_threshold=self.small_payload_threshold,
                csv_sep=self.csv_sep,
                csv_index=self.csv_index,
                parquet_index=self.parquet_index,
                excel_index=self.excel_index,
                encoding=self.encoding,
                pickle_protocol=self.pickle_protocol,
                joblib_compress=self.joblib_compress,
            )

            final_key = normalized_key
            if prepared.extension and not Path(final_key).suffix:
                final_key = final_key + prepared.extension

            items.append(
                {
                    "src": prepared,
                    "key": final_key,
                }
            )

        return items

    def _infer_transfer_mode(
        self,
        src: str,
        dst: str,
        *,
        bucket_src: Optional[str] = None,
        bucket_dst: Optional[str] = None,
    ) -> str:
        """
        Infer the transfer mode from source and destination specifications.

        The inference supports both explicit S3 URIs and implicit S3 keys relative
        to the configured/default bucket.

        Rules
        -----
        - If src is an explicit S3 URI, source is S3.
        - Otherwise, if src matches an existing local path or local glob, source is local.
        - Otherwise, source is treated as S3.
        - If dst is an explicit S3 URI, destination is S3.
        - Otherwise, if bucket_dst is provided, destination is treated as S3.
        - Otherwise, if source is S3 and dst looks like a local path, destination is local.
        - Otherwise, destination is treated as S3.

        Returns
        -------
        str
            One of:
            - "local_to_s3"
            - "s3_to_local"
            - "s3_to_s3"
        """
        src_str = str(src)
        dst_str = str(dst)

        if self._is_s3_uri(src_str):
            src_is_local = False
        else:
            local_matches = expand_pattern(value=src_str, location="local")
            src_is_local = bool(local_matches) or Path(src_str).exists()

        if self._is_s3_uri(dst_str):
            dst_is_s3 = True
        elif bucket_dst is not None:
            dst_is_s3 = True
        else:
            dst_is_s3 = not self._looks_like_local(dst_str)

        if src_is_local and dst_is_s3:
            return "local_to_s3"

        if not src_is_local and dst_is_s3:
            return "s3_to_s3"

        if not src_is_local and not dst_is_s3:
            return "s3_to_local"

        raise ValueError(
            f"Unsupported transfer combination: src={src!r}, dst={dst!r}"
        )

    def _is_s3_uri(self, value: str) -> bool:
        return str(value).startswith("s3://")

    @staticmethod
    def _looks_like_local(path_str: str) -> bool:
        p = Path(path_str)
        if p.is_absolute():
            return True
        if p.drive:
            return True
        if "\\" in path_str:
            return True
        if path_str.startswith(("./", "../", "~/")):
            return True
        if p.exists():
            return True
        return False

    def _parse_s3_uri(self, value: str) -> tuple[str, str]:
        raw = str(value)
        if not raw.startswith("s3://"):
            return self._resolve_bucket(None), self._normalize_key(raw)
        no_scheme = raw[5:]
        bucket, _, key = no_scheme.partition("/")
        if not bucket:
            raise ValueError(f"Invalid S3 URI: {value}")
        return bucket, normalize_path_like(key).strip("/")

    def _effective_extension(self, *, obj: Any, key: str | None = None, file_type: str | None = None) -> str | None:
        default_type = self.file_type if is_tabular(obj) else self._object_default_extension().lstrip(".")
        ext = resolve_extension(
            key=key,
            explicit_file_type=file_type,
            default_file_type=default_type,
        )
        if ext == ".jl":
            return ".joblib"
        return ext

    def _object_default_extension(self) -> str:
        if self.object_base_format == "joblib":
            return ".joblib"
        if self.object_base_format == "skops":
            return ".skops"
        return ".pkl"

    def _guard_unsafe_extension(self, extension: str | None) -> None:
        ext = normalize_extension(extension)
        if ext in _UNSAFE_EXTENSIONS and not self.allow_unsafe_serialization:
            raise ValueError(
                f"{ext} serialization is not allowed for security reasons. "
                "Enable allow_unsafe_serialization in `config()` to override this behavior."
            )

    def _prepare_upload_source_for_get_engine(self, src: Any) -> PreparedUploadSource:
        if isinstance(src, PreparedUploadSource):
            return src
        return prepare_upload_source(
            src,
            default_file_type=self.file_type,
            small_payload_threshold=self.small_payload_threshold,
            csv_sep=self.csv_sep,
            csv_index=self.csv_index,
            parquet_index=self.parquet_index,
            excel_index=self.excel_index,
            encoding=self.encoding,
            pickle_protocol=self.pickle_protocol,
            joblib_compress=self.joblib_compress,
        )

    def _filter_by_extensions(
        self,
        values: List[str],
        *,
        include_extensions: Optional[Sequence[str]] = None,
        exclude_extensions: Optional[Sequence[str]] = None,
    ) -> List[str]:
        inc = {normalize_extension(x) for x in include_extensions} if include_extensions else None
        exc = {normalize_extension(x) for x in exclude_extensions} if exclude_extensions else None
        out: List[str] = []
        for value in values:
            ext = normalize_extension(Path(value).suffix)
            if inc is not None and ext not in inc:
                continue
            if exc is not None and ext in exc:
                continue
            out.append(value)
        return out

    def _expand_s3_keys(self, keys: List[str], *, bucket: str, require_match: bool = False) -> List[str]:
        out: List[str] = []
        for key in keys:
            expanded = expand_pattern(value=key, location="s3", client=self._get_client(), bucket=bucket)
            if require_match:
                ensure_non_empty_selection(expanded, original_pattern=key)
            out.extend(expanded)
        return sorted(dict.fromkeys(out))

    # --------------------------------------------------------
    # |                     Configuration                    |
    # --------------------------------------------------------

    def config(
        self,
        bucket: str = None,
        *,
        key_prefix: str = "",
        output_type: TabularOutput = "pandas",
        file_type: TabularFileType = "parquet",
        overwrite: bool = True,
        encoding: str = "utf-8",
        csv_sep: str = ",",
        csv_index: bool = False,
        parquet_index: bool | None = None,
        excel_index: bool = False,
        allow_unsafe_serialization: bool = False,
        object_base_format: ObjectFormat = "pickle",
        pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
        joblib_compress: int = 3,
        small_payload_threshold: int = 5 * 1024 * 1024,
        multipart_threshold_mb: int = 5,
        multipart_chunksize_mb: int = 5,
        max_concurrency: int = 8,
        use_threads: bool = True,
        delete_batch_size: int = 1000,
    ) -> None:
        """
        Mandatory configuration method to set up S3 parameters.

        Parameters:
        -----------
        bucket: str
            Default S3 bucket name to operate on.
        key_prefix: str
            Optional prefix to prepend to all object keys.
        output_type: TabularOutput
            Default output type for loaded tabular objects (if applicable).
            Supported values: "pandas", "polars".
        file_type: TabularFileType
            Default file type for tabular uploads when the key extension is missing.
            Supported values: "csv", "parquet", "xlsx", "xls", "pkl", "pickle", "joblib", "jl", "skops".
        overwrite: bool
            Whether to allow overwriting existing objects when uploading.
        encoding: str
            Default encoding for text-based operations (e.g., JSON, CSV).
        csv_sep: str
            Default separator for CSV files (if output_type is "csv").
        csv_index: bool
            Whether to include the index when uploading pandas DataFrames as CSV (if output_type is "csv").
        parquet_index: bool | None
            Whether to persist the pandas index when uploading pandas DataFrames as parquet.
            ``None`` keeps pandas' native default behavior.
        excel_index: bool
            Whether to include the index when uploading pandas DataFrames as Excel.
        allow_unsafe_serialization: bool
            Whether to allow unsafe serialization of objects (pickle or alike).
        object_base_format: ObjectFormat
            The default serialization format for Python objects when the destination key does not have a recognized tabular extension.
            Supported values: "pickle", "joblib", "skops".
        pickle_protocol: int
            Pickle protocol version to use when serializing with pickle (if allowed).
        joblib_compress: int
            Compression level to use when serializing with joblib (if allowed).
        small_payload_threshold: int
            Max in-memory payload size before switching to a temp-file upload strategy.
        multipart_threshold_mb, multipart_chunksize_mb, max_concurrency, use_threads:
            Managed transfer configuration forwarded to boto3 TransferConfig.
        delete_batch_size: int
            Maximum number of S3 objects deleted per delete_objects call.
        """
        self.bucket = bucket
        self.key_prefix = key_prefix
        self.output_type = output_type
        self.file_type = file_type
        self.overwrite = overwrite
        self.encoding = encoding
        self.csv_sep = csv_sep
        self.csv_index = csv_index
        self.parquet_index = parquet_index
        self.excel_index = excel_index
        self.allow_unsafe_serialization = allow_unsafe_serialization
        self.object_base_format = object_base_format
        self.pickle_protocol = pickle_protocol
        self.joblib_compress = joblib_compress
        self.small_payload_threshold = small_payload_threshold
        self.delete_batch_size = delete_batch_size
        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold_mb * 1024**2,
            multipart_chunksize=multipart_chunksize_mb * 1024**2,
            max_concurrency=max_concurrency,
            use_threads=use_threads,
        )
        self.engine_cache = None
        self.aws.info(
            "S3 configured with bucket=%s, key_prefix=%s, output_type=%s, file_type=%s, overwrite=%s, allow_unsafe_serialization=%s",
            bucket,
            key_prefix,
            output_type,
            file_type,
            overwrite,
            allow_unsafe_serialization,
        )

    # --------------------------------------------------------
    # |                     External API                     |
    # --------------------------------------------------------

    def list(
        self,
        prefix: str = "",
        *,
        bucket: Optional[str] = None,
        limit: Optional[int] = None,
        recursive: bool = True,
        with_meta: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket.

        Parameters
        -----------
        prefix: optional str
            Filter objects by prefix.
        bucket: optional str
            Override default bucket from config.
        limit: optional int
            Maximum number of objects to return.
        recursive: bool
            If False, only list objects directly under the prefix (no deeper levels).
        with_meta: bool
            Whether to include metadata (size, last modified, etc.) in the output.

        Returns
        --------
        List of dictionaries containing object information.
        """
        b = self._resolve_bucket(bucket)
        pref = self._normalize_s3_prefix(prefix)

        return list_s3_objects(
            client=self._get_client(),
            bucket=b,
            prefix=pref,
            limit=limit,
            recursive=recursive,
            with_meta=with_meta,
        )

    def tree(
        self,
        prefix: str = "",
        *,
        bucket: Optional[str] = None,
        show_full_path: bool = True,
        max_depth: Optional[int] = None,
        max_children: Optional[int] = None,
        folders_first: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        """
        Display a tree view of S3 objects under a given prefix.

        Parameters:
        -----------
        prefix: optional str
            Filter objects by prefix.
        bucket: optional str
            Override default bucket from config.
        show_full_path: bool
            If True, display the full path for each node.
        max_depth: optional int
            Maximum depth to display.
        max_children: optional int
            Maximum number of children to display per folder.
        folders_first: bool
            Whether to display folders before files.
        limit: optional int
            Maximum number of objects to include in the tree.
        """
        b = self._resolve_bucket(bucket)
        pref = self._normalize_s3_prefix(prefix)

        objs = list_s3_objects(
            client=self._get_client(),
            bucket=b,
            prefix=pref,
            limit=limit,
            recursive=True,
            with_meta=True,
        )

        root_label = pref or "/"
        root = _build_tree_from_objects(objs, root_label=root_label)
        _compute_folder_sizes(root)

        rich_tree = _render_tree(
            root,
            show_full_path=show_full_path,
            max_depth=max_depth,
            max_children=max_children,
            folders_first=folders_first,
        )
        Console().print(rich_tree)

    def exists(self, key: KeyLike, *, bucket: Optional[str] = None) -> Union[bool, List[bool]]:
        """
        Check if one or more keys exist in the S3 bucket.

        Parameters
        -----------
        key: str or list of str
            The object key(s) to check for existence.
        bucket: optional str
            Override default bucket from config.

        Returns
        --------
        bool or list of bool
            A single bool if one key was provided, or a list of booleans (one
            per key, in the same order as the input) if multiple keys were given.
        """
        b = self._resolve_bucket(bucket)
        keys = self._normalize_keys(key)
        s3 = self._get_client()
        results = []
        for k in keys:
            try:
                s3.head_object(Bucket=b, Key=k)
                self.aws.info("File exists: s3://%s/%s", b, k)
                results.append(True)
            except ClientError as e:
                code = _err_code(e)
                if code in {"404", "NoSuchKey", "NotFound"}:
                    self.aws.info("File does not exist: s3://%s/%s", b, k)
                    results.append(False)
                else:
                    _raise_s3(e, bucket=b, key=k)
        return results[0] if len(results) == 1 else results

    def delete(self, key: KeyLike, *, force: bool = False, bucket: Optional[str] = None) -> None:
        """
        Delete one or more keys from the S3 bucket.

        Key can be :
        - A single key (str)
        - A list of keys (List[str])
        - A pattern with glob syntax (as "folder/*") to delete all matching keys. Use with force=True.

        Parameters
        -----------
        key: str or list of str
            The object key(s) to delete.
        force: bool
            Allow deleting of a tree of objects by pattern (as "folder/*").
            Use with caution as this will delete all matching objects.
        bucket: optional str
            Override default bucket from config.
        """
        b = self._resolve_bucket(bucket)

        normalized = self._normalize_keys(key)
        if any(has_glob(k) for k in normalized) and not force:
            raise ValueError(
                "Refusing to delete by pattern without force=True. Enable force=True as: s3.delete('research/*', force=True)"
            )

        keys = self._resolve_s3_keys(key, bucket=b, require_match=False)
        if not keys:
            self.aws.info("No objects matched for deletion.")
            return

        plan = build_delete_plan(keys=keys, bucket=b)
        self._get_engine().execute(plan)

    def download(
        self,
        key: KeyLike,
        to: PathLike = None,
        *,
        preserve_prefix: bool = False,
        bucket: Optional[str] = None,
    ) -> Union[Path, List[Path]]:
        """
        Download one or more keys from S3 to a local path.

        Parameters
        -----------
        key: str or list of str
            The object key(s) to download. Glob patterns are supported.
        to: str or Path
            Local file path or directory to download to. If downloading multiple keys, this should be a directory.
        preserve_prefix: bool
            If True, preserve the S3 key prefix structure in the local download path.
            If False, preserve structure relative to the static root of each glob/prefix selection.
            Ex:
            - key="data/2023/*.csv", to="downloads/", preserve_prefix=True -> downloads/data/2023/file.csv
            - key="data/2023/*.csv", to="downloads/", preserve_prefix=False -> downloads/file.csv
        bucket: optional str
            Override default bucket from config.

        Returns
        -------
        Path or list of Paths
            The local path(s) where the object(s) were downloaded.
        """
        b = self._resolve_bucket(bucket)
        dest_base = Path(".") if to is None else Path(to)

        normalized_inputs = self._normalize_keys(key)
        keys = self._resolve_s3_keys(key, bucket=b, require_match=False)

        if not keys:
            self.aws.info("No objects matched for download.")
            return []

        multi = len(keys) > 1 or any(has_glob(k) for k in normalized_inputs)

        if multi and dest_base.suffix and not str(dest_base).endswith(("/", "\\")):
            raise ValueError("When downloading multiple keys (or patterns), `to` must be a directory path.")

        if multi or dest_base.is_dir() or str(dest_base).endswith(("/", "\\")):
            source_root = (
                ""
                if preserve_prefix
                else common_static_root(normalized_inputs[0]) if len(normalized_inputs) == 1 else ""
            )
            rels = map_preserving_structure(
                sources=keys,
                source_root=source_root,
                destination_root="",
            )
            dsts = [dest_base / rel for rel in rels]
        else:
            dsts = [dest_base]

        plan = build_download_plan(
            keys=keys,
            dsts=[str(p) for p in dsts],
            bucket=b,
        )
        self._get_engine().execute(plan)

        out = [Path(p) for p in dsts]
        return out[0] if len(out) == 1 else out

    def load(
        self,
        key: KeyLike,
        *,
        bucket: Optional[str] = None,
        output_type: Optional[TabularOutput] = None,
    ) -> Any:
        """
        Load one or more keys from S3 into Python objects.

        Parameters
        -----------
        key: str or list of str
            The object key(s) to load. Glob patterns are supported.
        bucket: optional str
            Override default bucket from config.
        output_type: optional str
            Override default output type from config for this load operation.
            Supported values: "pandas", "polars".

        Returns
        --------
        The loaded object(s). The type depends on the output_type and file format:
        - For tabular formats (CSV, Parquet), returns a DataFrame-like object based on output_type.
        - For text-based formats (JSON), returns a dict or list.
        - For bytes output, returns raw bytes. It is also the fallback for unknown formats.
        """
        b = self._resolve_bucket(bucket)
        keys = self._resolve_s3_keys(key, bucket=b, require_match=False)

        if not keys:
            self.aws.info("No objects matched for load.")
            return []

        plan = build_load_plan(keys=keys, bucket=b)
        raw_items = self._get_engine().execute(plan)

        tabular_output = output_type or self.output_type
        out: List[Any] = []

        for item in raw_items:
            k = item["key"]
            ext = normalize_extension(Path(k).suffix)
            if ext in _UNSAFE_EXTENSIONS:
                self._guard_unsafe_extension(ext)

            out.append(
                deserialize_payload(
                    item["payload"],
                    key=k,
                    output=tabular_output,
                    encoding=self.encoding,
                )
            )

        return out[0] if len(out) == 1 else out

    def upload(
        self,
        src: Union[UploadInput, Sequence[UploadInput]],
        key: KeyLike,
        *,
        bucket: Optional[str] = None,
        overwrite: Optional[bool] = None,
    ) -> Union[str, List[str]]:
        """
        Upload one or more local files or Python objects to S3.

        Parameters
        -----------
        src: UploadInput or list of UploadInput
            The source object(s) to upload. Can be a single object or a list of objects.
            Supported types:
            - Local file paths (str or Path), including glob patterns for batch uploads
            - Python dicts (will be serialized as JSON)
            - Tabular objects (pandas or polars DataFrames, will be serialized based on key extension or default file_type)
            - Raw bytes payloads
        key: str or list of str
            The destination key(s) for the upload. Can be a single key or a list of keys.
            If uploading multiple local files matched from a glob, a single destination prefix is accepted and
            the relative structure is preserved.
        bucket: optional str
            Override default bucket from config.
        overwrite: optional bool
            Whether to allow overwriting existing objects. Overrides default overwrite setting from config for this operation.

        Returns
        --------
        str or list of str
            The key(s) of the uploaded object(s) in S3.
        """
        b = self._resolve_bucket(bucket)
        ow = self.overwrite if overwrite is None else overwrite

        items = self._resolve_upload_items(
            src=src,
            key=key,
            bucket=b,
            overwrite=ow,
        )

        if not items:
            self.aws.info("No objects matched for upload.")
            return []

        plan = build_upload_plan(
            items=items,
            bucket=b,
        )
        self._get_engine().execute(plan)

        keys = [item["key"] for item in items]
        return keys[0] if len(keys) == 1 else keys

    def transfer(
        self,
        src: str,
        dst: str,
        *,
        move: bool = True,
        bucket_src: Optional[str] = None,
        bucket_dst: Optional[str] = None,
    ) -> Union[str, List[str], Path, List[Path]]:
        """
        Transfer one or more files between local paths and S3.

        Parameters
        ----------
        src: str
            Source path, key, S3 URI, or glob pattern.
        dst: str
            Destination path, prefix, or S3 URI.
        move: bool
            If True, remove the source after a successful transfer when supported.
        bucket_src: optional str
            Override source bucket for S3 sources.
        bucket_dst: optional str
            Override destination bucket for S3 destinations.

        Returns
        -------
        str | list[str] | Path | list[Path]
            Destination(s) created by the transfer.
        """
        src_str = str(src)
        dst_str = str(dst)

        mode = self._infer_transfer_mode(
            src_str,
            dst_str,
            bucket_src=bucket_src,
            bucket_dst=bucket_dst,
        )

        # ------------------------------------------------------------------
        # local -> s3
        # ------------------------------------------------------------------
        if mode == "local_to_s3":
            srcs = self._resolve_local_paths(src_str, require_match=True)

            dst_bucket = (
                bucket_dst
                or (self._parse_s3_uri(dst_str)[0] if self._is_s3_uri(dst_str) else self._resolve_bucket(None))
            )
            dst_root = (
                self._parse_s3_uri(dst_str)[1]
                if self._is_s3_uri(dst_str)
                else self._normalize_keys(dst_str)[0]
            )

            source_root = (
                common_static_root(src_str)
                if has_glob(src_str)
                else str(Path(src_str).parent)
            )

            dsts = map_preserving_structure(
                sources=srcs,
                source_root=source_root,
                destination_root=dst_root,
            )

            plan = build_transfer_plan(
                mode=mode,
                srcs=srcs,
                dsts=dsts,
                bucket_dst=dst_bucket,
                move=move,
            )
            self._get_engine().execute(plan)

            return dsts[0] if len(dsts) == 1 else dsts

        # ------------------------------------------------------------------
        # s3 source resolution
        # ------------------------------------------------------------------
        src_bucket_resolved = (
            bucket_src
            or (self._parse_s3_uri(src_str)[0] if self._is_s3_uri(src_str) else self._resolve_bucket(None))
        )
        src_pattern = (
            self._parse_s3_uri(src_str)[1]
            if self._is_s3_uri(src_str)
            else src_str
        )
        srcs = self._resolve_s3_keys(src_pattern, bucket=src_bucket_resolved, require_match=True)

        # ------------------------------------------------------------------
        # s3 -> s3
        # ------------------------------------------------------------------
        if mode == "s3_to_s3":
            dst_bucket_resolved = (
                bucket_dst
                or (self._parse_s3_uri(dst_str)[0] if self._is_s3_uri(dst_str) else self._resolve_bucket(None))
            )
            dst_root = (
                self._parse_s3_uri(dst_str)[1]
                if self._is_s3_uri(dst_str)
                else self._normalize_keys(dst_str)[0]
            )

            src_root = common_static_root(
                self._normalize_keys(src_pattern)[0] if isinstance(src_pattern, str) else str(src_pattern)
            )
            dsts = map_preserving_structure(
                sources=srcs,
                source_root=src_root,
                destination_root=dst_root,
            )

            plan = build_transfer_plan(
                mode=mode,
                srcs=srcs,
                dsts=dsts,
                bucket_src=src_bucket_resolved,
                bucket_dst=dst_bucket_resolved,
                move=move,
            )
            self._get_engine().execute(plan)

            return dsts[0] if len(dsts) == 1 else dsts

        # ------------------------------------------------------------------
        # s3 -> local
        # ------------------------------------------------------------------
        dst_base = Path(dst_str)

        if len(srcs) == 1 and dst_base.suffix and not str(dst_base).endswith(("/", "\\")):
            dsts = [str(dst_base)]
        else:
            normalized_src = self._normalize_keys(src_pattern)[0] if isinstance(src_pattern, str) else str(src_pattern)
            src_root = common_static_root(normalized_src)

            if not has_glob(normalized_src):
                src_root = normalize_path_like(str(Path(normalized_src).parent))

            rels = map_preserving_structure(
                sources=srcs,
                source_root=src_root,
                destination_root="",
            )
            dsts = [str(dst_base / rel) for rel in rels]

        plan = build_transfer_plan(
            mode=mode,
            srcs=srcs,
            dsts=dsts,
            bucket_src=src_bucket_resolved,
            move=move,
        )
        self._get_engine().execute(plan)

        out = [Path(p) for p in dsts]
        return out[0] if len(out) == 1 else out
