from __future__ import annotations
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    Callable
)
from io import BytesIO
TabularOutput = Literal["pandas", "polars"]
KeyLike = Union[str, Sequence[str]]
PathLike = Union[str, Path]
TabularFileType = Literal["csv", "parquet", "xlsx", "xls"]
from botocore.exceptions import ClientError
from .errors import _raise_s3, _err_code
import json
import pandas as pd
import polars as pl
UploadInput = Union[pd.DataFrame, pl.DataFrame, dict, PathLike]


class S3:
    """
    Class responsible for S3 interactions using boto3 client.

    It provides the following methods:
        - list: List objects in an S3 bucket with optional filtering and metadata.
        - exists: Check if a specific key exists in the bucket.
        - delete: Delete one or more keys from the bucket.
        - download: Download one or more keys to a local path.
        - load: Load one or more keys into Python objects.
        - upload: Upload one or more local files or Python objects to the bucket.

    Attributes:
        aws (AWS): An instance of the AWS class for session management and configuration.
        cfg (S3Config): Configuration for S3 operations, including bucket name, encoding, overwrite behavior, and output type.
    """

    def __init__(self, aws):
        self.aws = aws
        self.logger = aws.logger.getChild("s3")

        self.client_cache = None
        self.bucket = None
        self.key_prefix = None
        self.output_type = None
        self.file_type = None
        self.overwrite = None
        self.encoding = None
        self.csv_sep = None
        self.csv_index = None

    # --------------------------------------------------------
    # |                   Internal Helpers                   |
    # --------------------------------------------------------

    def _bucket(self, bucket: Optional[str]) -> str:
        """
        Resolve the bucket name to use for an operation, checking the provided bucket parameter,
        and the instance's bucket override.

        Parameters:
        -----------
        bucket: optional str
            The bucket name provided directly to the method call, which takes precedence over the instance's default bucket.

        Returns:
        --------
        str
            The resolved bucket name to use for the S3 operation.
        """
        b = bucket or self.bucket
        if not b:
            raise ValueError("Bucket is not set. Provide bucket=... or configure a default bucket.")
        
        return b
    
    def _client(self):
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
    
    def _resolve_key(self, key: str) -> str:
        """
        Build the full S3 path by combining the base prefix from configuration with the provided key.

        Parameters:
        -----------
        key: str
            The object key to resolve.

        Returns:
        str
            The resolved S3 key with the configured prefix applied.
        """
        key = key.lstrip("/")
        if self.key_prefix:
            base = self.key_prefix.strip("/")
            if key.startswith(base + "/") or key == base:
                return key
            return f"{base}/{key}"
        return key
    
    def _normalize_keys(self, key: KeyLike) -> List[str]:
        """
        Normalize the input key(s) to a list of resolved S3 keys.

        Parameters:
        -----------
        key: str or list of str
            The object key(s) to normalize.

        Returns:
        --------
        List[str]
            A list of resolved S3 keys.
        """
        if isinstance(key, (list, tuple)):
            return [self._resolve_key(str(k)) for k in key]
        return [self._resolve_key(str(key))]
    
    def _normalize_upload(self, src: Union[UploadInput, Sequence[UploadInput]], key: KeyLike) -> Tuple[List[UploadInput], List[str]]:
        """
        Normalize the upload inputs and keys after checking for consistency.

        Parameters:
        -----------
        src: UploadInput or list of UploadInput
            The source object(s) to upload. Can be a single object or a list of objects.
        key: KeyLike
            The destination key(s) for the upload. Can be a single key or a list of keys.

        Returns:
        Tuple[List[UploadInput], List[str]]
            A tuple containing the list of source objects and the corresponding list of resolved keys.
        """
        srcs: List[UploadInput] = list(src) if isinstance(src, (list, tuple)) else [src]
        keys_raw: List[str] = list(key) if isinstance(key, (list, tuple)) else [key]

        if len(srcs) != len(keys_raw):
            raise ValueError(f"src and key must have the same length (got {len(srcs)} vs {len(keys_raw)}).")

        keys: List[str] = [self._resolve_key(str(k)) for k in keys_raw]

        for k in keys:
            if k.endswith("/"):
                raise ValueError(f"Destination key must be a full object key (not a prefix ending with '/'): {k}")

        return srcs, keys
    
    def _map_loader(self, file_extension: str, output_tabular: str) -> Callable[[bytes], Any]:
        """
        Define the appropriate loader function based on file extension and desired output type.

        The strategy is the following:
            - If the file is a JSON, we trivially decode it to Python dict.
            - For tabular formats (CSV, Parquet, Excel), we choose the loader based on the output type:
                - If output_tabular is "pandas", we use pandas' read_csv, read_parquet, or read_excel.
                - If output_tabular is "polars", we use polars' read_csv, read_parquet, or read_excel
            - For unknown formats we return raw bytes as a fallback.

        Parameters:
        -----------
        file_extension: str
            The file extension (e.g., ".csv", ".json") to determine the format of the data.
        output_tabular: str
            The desired output type for tabular data (e.g., "pandas", "polars").

        Returns:
        Callable[[bytes], Any]
            A function that takes raw bytes and returns the loaded object in the desired format.
        """
        if file_extension == ".json":
            return lambda raw: json.loads(raw.decode(self.encoding))

        if file_extension in {".csv", ".parquet", ".xlsx", ".xls"}:

            if output_tabular == "pandas":

                pandas_map: dict[str, Callable[[bytes], Any]] = {
                    ".csv": lambda raw: pd.read_csv(BytesIO(raw), sep=self.csv_sep, encoding=self.encoding),
                    ".parquet": lambda raw: pd.read_parquet(BytesIO(raw)),
                    ".xlsx": lambda raw: pd.read_excel(BytesIO(raw)),
                    ".xls": lambda raw: pd.read_excel(BytesIO(raw)),
                }
                return pandas_map.get(file_extension, lambda raw: raw)
            elif output_tabular == "polars":

                polars_map: dict[str, Callable[[bytes], Any]] = {
                    ".csv": lambda raw: pl.read_csv(BytesIO(raw), separator=self.csv_sep),
                    ".parquet": lambda raw: pl.read_parquet(BytesIO(raw)),
                    ".xlsx": lambda raw: pl.read_excel(BytesIO(raw)),
                    ".xls": lambda raw: pl.read_excel(BytesIO(raw)),
                }
                return polars_map.get(file_extension, lambda raw: raw)
            else:
                raise ValueError(f"Unsupported output type for tabular data: {output_tabular}")

        return lambda raw: raw

    def _map_uploader(self,
                      s3_client,
                      bucket: str,
                      obj: UploadInput,
                      dest_key: str,
                      overwrite: bool) -> Tuple[str, Callable]:
        """
        Define the appropriate upload action based on the type of the input object and the destination key.

        The strategy is the following:
            - If the object is a local file path, we use s3.upload_file.
            - If the object is a dict, we serialize it as JSON and use s3.put_object.
            - If the object is a tabular DataFrame (pandas or polars), we serialize it
              based on the destination key extension (or default file type) and use s3.put_object.
        Parameters:
        -----------
        s3_client: boto3 S3 client
            The S3 client to use for performing the upload action.
        bucket: str
            The S3 bucket where the object should be uploaded.
        obj: UploadInput
            The source object to upload. Can be a local file path, a dict, or a tabular DataFrame.
        dest_key: str
            The destination key in S3 where the object should be uploaded.
        overwrite: bool
            Whether to allow overwriting an existing object at the destination key.

        Returns:
        --------
        Tuple[str, Callable]
            A tuple containing the final resolved key for the upload and a callable action that performs the upload when called.
        """
        b = bucket

        if isinstance(obj, (str, Path)):
            p = Path(obj)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(str(p))

            final_key = dest_key

            def action():
                if not overwrite and self.exists(final_key, bucket=b):
                    raise ValueError(f"Refusing to overwrite existing object: s3://{b}/{final_key}")
                else:
                    s3_client.upload_file(str(p), b, final_key)

            return final_key, action

        if isinstance(obj, dict):
            final_key = dest_key
            if Path(final_key).suffix.lower() == "":
                final_key = final_key + ".json"

            data = json.dumps(obj, ensure_ascii=False, indent=2).encode(self.encoding)

            def action() -> None:
                if not overwrite and self.exists(final_key, bucket=b):
                    raise ValueError(f"Refusing to overwrite existing object: s3://{b}/{final_key}")
                else:
                    s3_client.put_object(Bucket=b, Key=final_key, Body=data, ContentType="application/json")
                    
            return final_key, action

        if isinstance(obj, (pd.DataFrame, pl.DataFrame)):
            final_key = dest_key
            ext = Path(final_key).suffix.lower()
            if ext == "":
                ext = "." + self.file_type if self.file_type else ".parquet"
                final_key = final_key + ext

            if obj.__class__.__module__.startswith("pandas"):
                if ext == ".csv":
                    s = obj.to_csv(sep=self.csv_sep, index=self.csv_index)
                    data, ct = s.encode(self.encoding), "text/csv"
                elif ext == ".parquet":
                    bio = BytesIO()
                    obj.to_parquet(bio, index=False)
                    data, ct = bio.getvalue(), "application/octet-stream"
                elif ext in {".xlsx", ".xls"}:
                    bio = BytesIO()
                    obj.to_excel(bio, index=False)
                    ct = (
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        if ext == ".xlsx"
                        else "application/vnd.ms-excel"
                    )
                    data, ct =  bio.getvalue(), ct
                else:
                    raise ValueError(f"Unsupported tabular extension for pandas: {ext}")

            if obj.__class__.__module__.startswith("polars"):
                if ext == ".csv":
                    s = obj.write_csv()
                    data, ct = s.encode(self.encoding), "text/csv"
                elif ext == ".parquet":
                    bio = BytesIO()
                    obj.write_parquet(bio)
                    data, ct = bio.getvalue(), "application/octet-stream"
                elif ext in {".xlsx", ".xls"}:
                    bio = BytesIO()
                    obj.to_pandas().to_excel(bio, index=False)
                    ct = (
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        if ext == ".xlsx"
                        else "application/vnd.ms-excel"
                    )
                    data, ct = bio.getvalue(), ct
                else:
                    raise ValueError(f"Unsupported tabular extension for polars: {ext}")

            def action() -> None:
                if not overwrite and self.exists(final_key, bucket=b):
                    raise ValueError(f"Refusing to overwrite existing object: s3://{b}/{final_key}")
                else:
                    s3_client.put_object(Bucket=b, Key=final_key, Body=data, ContentType=ct)

            return final_key, action
        
    # --------------------------------------------------------
    # |                     Configuration                    |
    # --------------------------------------------------------

    def config(self,
               bucket: str = None,
               key_prefix: str = "",
               output_type: TabularOutput = "pandas",
               file_type: TabularFileType = "parquet",
               overwrite: bool = True,
               encoding: str = "utf-8",
               csv_sep: str = ",",
               csv_index: bool = False):
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
            Supported values: "csv", "parquet", "xlsx", "xls".
        overwrite: bool
            Whether to allow overwriting existing objects when uploading.
        encoding: str
            Default encoding for text-based operations (e.g., JSON, CSV).
        csv_sep: str
            Default separator for CSV files (if output_type is "csv").
        csv_index: bool
            Whether to include the index when uploading pandas DataFrames as CSV (if output_type is "csv").
        """
        self.bucket = bucket
        self.key_prefix = key_prefix
        self.output_type = output_type
        self.file_type = file_type
        self.overwrite = overwrite
        self.encoding = encoding
        self.csv_sep = csv_sep
        self.csv_index = csv_index

        self.aws.info("S3 configured with bucket=%s, key_prefix=%s, output_type=%s, file_type=%s, overwrite=%s",
                         bucket, key_prefix, output_type, file_type, overwrite)

    # --------------------------------------------------------
    # |                    External API                      |
    # --------------------------------------------------------

    def list(self,
             prefix: str = "",
             *,
             bucket: Optional[str] = None,
             limit: Optional[int] = None,
             recursive: bool = True,
             with_meta: bool = True) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket.

        Parameters:
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
        
        Returns:
        --------
        List of dictionaries containing object information.
        """
        b = self._bucket(bucket)
        s3 = self._client()

        pref = self._resolve_key(prefix)
        if pref and not pref.endswith("/") and not recursive:
            pref = pref + "/"

        out: List[Any] = []
        try:
            paginator = s3.get_paginator("list_objects_v2")
            paginate_kwargs = {"Bucket": b, "Prefix": pref}
            if not recursive:
                paginate_kwargs["Delimiter"] = "/"

            for page in paginator.paginate(**paginate_kwargs):
                for obj in page.get("Contents", []):
                    if with_meta:
                        out.append(
                            {
                                "key": obj["Key"],
                                "size": int(obj.get("Size", 0)),
                                "last_modified": obj.get("LastModified"),
                                "etag": (obj.get("ETag") or "").strip('"') or None,
                                "storage_class": obj.get("StorageClass"),
                            }
                        )
                    else:
                        out.append(obj["Key"])

                    if limit is not None and len(out) >= limit:
                        return out
            return out
        except ClientError as e:
            _raise_s3(e, bucket=b)

    def exists(self, key: str, *, bucket: Optional[str] = None) -> bool:
        """
        Check if a specific key exists in the S3 bucket.

        Parameters:
        -----------
        key: str
            The object key to check for existence.
        bucket: optional str
            Override default bucket from config.
        
        Returns:
        --------
        bool
            True if the object exists, False if it does not exist.
        """
        b = self._bucket(bucket)
        k = self._resolve_key(key)
        s3 = self._client()

        try:
            s3.head_object(Bucket=b, Key=k)
            self.aws.info("File exists: s3://%s/%s", b, k)
            return True
        except ClientError as e:
            code = _err_code(e)
            if code in {"404", "NoSuchKey", "NotFound"}:
                self.aws.info("File does not exist: s3://%s/%s", b, k)
                return False
            _raise_s3(e, bucket=b, key=k)
    
    def delete(self, key: KeyLike, *, bucket: Optional[str] = None) -> None:
        """
        Delete one or more keys from the S3 bucket.

        Parameters:
        -----------
        key: str or list of str
            The object key(s) to delete.
        bucket: optional str
            Override default bucket from config.
        """
        b = self._bucket(bucket)
        keys = self._normalize_keys(key)
        s3 = self._client()
        try:
            if len(keys) == 1:
                s3.delete_object(Bucket=b, Key=keys[0])
                self.aws.info("Deleted s3://%s/%s", b, keys[0])
            else:
                for i in range(0, len(keys), 1000):
                    chunk = keys[i: i + 1000]
                    s3.delete_objects(
                        Bucket=b,
                        Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
                    )
                    self.aws.info("Deleted s3://%s/%s (batch %d)", b, chunk, i // 1000 + 1)
        except ClientError as e:
            _raise_s3(e, bucket=b)
    
    def download(self,
                 key: KeyLike,
                 to: PathLike = None,
                 *,
                 bucket: Optional[str] = None) -> Union[Path, List[Path]]:
        """
        Download one or more keys from S3 to a local path.

        Parameters:
        -----------
        key: str or list of str
            The object key(s) to download.
        to: str or Path
            Local file path or directory to download to.
            If downloading multiple keys, this should be a directory.
        bucket: optional str
            Override default bucket from config.

        Returns:
        Path or list of Paths
            The local path(s) where the object(s) were downloaded.
        """
        b = self._bucket(bucket)
        keys = self._normalize_keys(key)
        s3 = self._client()

        if to is None:
            dest_base = Path(".")
        else:
            dest_base = Path(to)

        multi = len(keys) > 1
        if multi and dest_base.suffix:
            raise ValueError("When downloading multiple keys, `to` must be a directory path (not a file).")

        out: List[Path] = []
        for k in keys:

            if to is None:
                path = Path(Path(k).name)
            else:
                is_dir_semantics = dest_base.is_dir() or str(dest_base).endswith(("/", "\\"))
                if multi or is_dir_semantics:
                    path = dest_base / Path(k).name
                else:
                    path = dest_base

            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists() and not self.overwrite:
                out.append(path)
                continue

            try:
                s3.download_file(b, k, str(path))
                out.append(path)
                self.aws.info("Downloaded s3://%s/%s to %s", b, k, path)
            except ClientError as e:
                _raise_s3(e, bucket=b, key=k)

        return out[0] if len(out) == 1 else out
    
    def load(self,
             key: KeyLike,
             *,
             bucket: Optional[str] = None,
             output_type: Optional[TabularOutput] = None) -> Any:
        """
        Load one or more keys from S3 into Python objects.

        Parameters:
        -----------
        key: str or list of str
            The object key(s) to load.
        bucket: optional str
            Override default bucket from config.
        output_type: optional str
            Override default output type from config for this load operation.
            Supported values: "pandas", "polars", "pyarrow", "bytes", "text", "json".

        Returns:
        --------
        The loaded object(s). The type depends on the output_type and file format:
            - For tabular formats (CSV, Parquet), returns a DataFrame-like object based on output_type.
            - For text-based formats (JSON), returns a dict or list.
            - For bytes output, returns raw bytes. It is also the fallback for unknown formats.
        """
        b = self._bucket(bucket)
        keys = self._normalize_keys(key)
        s3 = self._client()
        tabular_output = output_type or self.output_type
    
        out: List[Any] = []
        for k in keys:
            ext = Path(k).suffix.lower()

            try:
                raw = s3.get_object(Bucket=b, Key=k)["Body"].read()
                self.aws.info("Loaded s3://%s/%s", b, k)
            except ClientError as e:
                _raise_s3(e, bucket=b, key=k)

            loader = self._map_loader(ext, tabular_output)
            out.append(loader(raw))

        return out[0] if len(out) == 1 else out
        
    def upload(self,
               src: Union[UploadInput, Sequence[UploadInput]],
               key: KeyLike,
               *,
               bucket: Optional[str] = None,
               overwrite: Optional[bool] = None) -> Union[str, List[str]]:
        """
        Upload one or more local files or Python objects to S3.

        Parameters:
        -----------
        src: UploadInput or list of UploadInput
            The source object(s) to upload. Can be a single object or a list of objects.
            Supported types:
                - Local file paths (str or Path)
                - Python dicts (will be serialized as JSON)
                - Tabular objects (pandas or polars DataFrames, will be serialized based on key extension or default file_type)
        key: str or list of str
            The destination key(s) for the upload. Can be a single key or a list of keys.
            If uploading multiple objects, this should be a list of keys with the same length as src.
        bucket: optional str
            Override default bucket from config.
        overwrite: optional bool
            Whether to allow overwriting existing objects. Overrides default overwrite setting from config for this operation.

        Returns:
        --------
        str or list of str
            The key(s) of the uploaded object(s) in S3.
        """
        b = self._bucket(bucket)
        ow = self.overwrite if overwrite is None else overwrite
        srcs, keys = self._normalize_upload(src, key)
        s3 = self._client()

        final_keys: List[str] = []
        for obj, k in zip(srcs, keys):
            final_key, action = self._map_uploader(s3, b, obj, k, ow)
            action()
            self.aws.info("Uploaded s3://%s/%s", b, final_key)
            final_keys.append(final_key)
        
        return final_keys[0] if len(final_keys) == 1 else final_keys