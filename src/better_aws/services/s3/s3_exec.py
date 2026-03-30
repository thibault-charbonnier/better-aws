"""
Execution engine for S3 transfer plans.

This module contains the low-level execution layer responsible for running
TransferPlan objects against AWS S3 using boto3 managed transfer primitives
whenever possible.

The engine is intentionally separated from the planning layer:
    public API -> resolve / normalize -> build TransferPlan -> execute plan

Responsibilities:
    - execute planned actions in order
    - dispatch each action to the appropriate boto3 primitive
    - centralize TransferConfig usage for managed transfers
    - batch S3 deletions efficiently with delete_objects
    - keep execution concerns isolated from planning and path resolution

For uploads, this engine expects a preparation function to have converted the
source object into one of the supported upload representations. By default,
the engine supports:
    - local file path uploads
    - raw bytes uploads

The engine is designed to be extended later with:
    - upload_fileobj support
    - temp file cleanup
    - retry / dry-run policies
    - execution reports
    - callbacks / progress bars
"""

from .s3_planner import TransferAction, TransferPlan
from .s3_serialization import PreparedUploadSource
from boto3.s3.transfer import TransferConfig
from typing import Any, Callable, Iterable
from botocore.client import BaseClient
from pathlib import Path

DEFAULT_DELETE_BATCH_SIZE = 1000


def default_prepare_upload_source(src: Any) -> PreparedUploadSource:
    """
    Default upload preparation strategy.

    This function keeps the execution engine usable with a minimal policy:
    - local paths are treated as managed file uploads
    - bytes are uploaded directly with put_object

    More advanced logic such as dataframe serialization, temp files, or size-
    based routing should be implemented in a dedicated preparation layer and
    injected into ``S3ExecutionEngine``.

    Parameters
    ----------
    src:
        Raw upload source from a planned action.

    Returns
    -------
    PreparedUploadSource
        Prepared upload source compatible with the execution engine.

    Raises
    ------
    TypeError
        If the source type is not supported by the default preparation policy.
    FileNotFoundError
        If a provided local path does not exist.
    """
    if isinstance(src, bytes):
        return PreparedUploadSource(
            mode="bytes",
            payload=src,
            size_hint=len(src),
        )

    if isinstance(src, (str, Path)):
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"Upload source does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Upload source must be a file: {path}")

        return PreparedUploadSource(
            mode="file_path",
            payload=str(path),
            size_hint=path.stat().st_size,
        )

    raise TypeError(
        "Unsupported upload source type for default_prepare_upload_source: "
        f"{type(src)!r}"
    )


class S3ExecutionEngine:
    """
    Execution engine for S3 transfer plans.

    This class is the low-level runtime component that takes a ``TransferPlan``
    and executes its actions using boto3 S3 primitives. Its role is strictly
    operational: the plan must already be resolved and normalized before being
    passed here.
    """

    def __init__(
        self,
        client: BaseClient,
        transfer_config: TransferConfig | None = None,
        prepare_upload_source: Callable[[Any], PreparedUploadSource] | None = None,
        delete_batch_size: int = DEFAULT_DELETE_BATCH_SIZE,
    ) -> None:
        """
        Attributes
        ----------
        client:
            boto3 S3 client.
        transfer_config:
            Optional boto3 TransferConfig used for managed transfers. If omitted,
            boto3 defaults are used.
        prepare_upload_source:
            Function converting a raw upload source into a
            ``PreparedUploadSource``. This makes the engine extensible without
            mixing serialization/performance policy into the planner.
        delete_batch_size:
            Maximum number of objects deleted in a single ``delete_objects`` call.
            S3 accepts up to 1000 objects per batch.
        """
        self.client = client
        self.transfer_config = transfer_config
        self.prepare_upload_source = (
            prepare_upload_source or default_prepare_upload_source
        )
        self.delete_batch_size = delete_batch_size

    def execute(self, plan: TransferPlan) -> list[dict[str, Any]]:
        """
        Execute a transfer plan in action order.

        Delete actions are buffered and flushed in batches for efficiency.
        Non-delete actions are executed immediately.

        Parameters
        ----------
        plan:
            Transfer plan to execute.

        Returns
        -------
        list[dict[str, Any]]
            Lightweight execution report, one entry per executed action or
            delete batch.
        """
        results: list[dict[str, Any]] = []
        delete_buffer: list[TransferAction] = []

        for action in plan.actions:
            if action.type == "delete_object":
                delete_buffer.append(action)
                if len(delete_buffer) >= self.delete_batch_size:
                    results.append(self._flush_delete_buffer(delete_buffer))
                    delete_buffer.clear()
                continue

            if delete_buffer:
                results.append(self._flush_delete_buffer(delete_buffer))
                delete_buffer.clear()

            results.append(self._execute_action(action))

        if delete_buffer:
            results.append(self._flush_delete_buffer(delete_buffer))

        return results

    def _execute_action(self, action: TransferAction) -> dict[str, Any]:
        """
        Execute a single non-batched action.

        Parameters
        ----------
        action:
            Planned action to execute.

        Returns
        -------
        dict[str, Any]
            Lightweight execution metadata.
        """
        if action.type == "load_object":
            return self._execute_load_object(action)

        if action.type == "upload_file":
            return self._execute_upload_file(action)

        if action.type == "upload_bytes":
            return self._execute_upload_bytes(action)

        if action.type == "download_file":
            return self._execute_download_file(action)

        if action.type == "copy_s3":
            return self._execute_copy_s3(action)

        if action.type == "delete_local":
             return self._execute_delete_local(action)

        raise ValueError(f"Unsupported action type: {action.type}")

    def _execute_delete_local(self, action: TransferAction) -> dict[str, Any]:
        """
        Delete a local file after a successful transfer.

        Parameters
        ----------
        action:
            Planned local deletion action.

        Returns
        -------
        dict[str, Any]
            Execution metadata describing the deleted local file.

        Raises
        ------
        FileNotFoundError
            If the local file does not exist.
        ValueError
            If the source path is not a file.
        """
        path = Path(str(action.src))

        if not path.exists():
            raise FileNotFoundError(f"Local file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Local deletion expects a file path, got: {path}")

        path.unlink()

        return {
            "type": action.type,
            "src": str(path),
        }

    def _execute_upload_file(self, action: TransferAction) -> dict[str, Any]:
        """
        Execute a managed file upload to S3.

        This method routes the upload through boto3's managed transfer system,
        which enables multipart uploads and concurrency according to the
        configured ``TransferConfig``.

        Parameters
        ----------
        action:
            Upload action whose source is expected to resolve to a local file.

        Returns
        -------
        dict[str, Any]
            Execution metadata describing the completed upload.
        """
        prepared = self.prepare_upload_source(action.src)
        if prepared.mode not in {"file_path", "temp_file"}:
            raise ValueError(
                "Action 'upload_file' requires prepared mode "
                "'file_path' or 'temp_file', "
                f"got {prepared.mode!r}"
            )

        extra_args = action.extra.get("extra_args")

        try:
            self.client.upload_file(
                Filename=str(prepared.payload),
                Bucket=action.bucket_dst,
                Key=action.dst,
                ExtraArgs=extra_args,
                Config=self.transfer_config,
            )
        finally:
            if prepared.cleanup and prepared.mode == "temp_file":
                Path(str(prepared.payload)).unlink(missing_ok=True)

        return {
            "type": action.type,
            "bucket": action.bucket_dst,
            "key": action.dst,
            "src": prepared.payload,
    }

    def _execute_upload_bytes(self, action: TransferAction) -> dict[str, Any]:
        """
        Execute an in-memory bytes upload to S3.

        This method uses ``put_object`` and is therefore best suited to small
        payloads already materialized in memory.

        Parameters
        ----------
        action:
            Upload action whose source is expected to resolve to bytes.

        Returns
        -------
        dict[str, Any]
            Execution metadata describing the completed upload.
        """
        prepared = self.prepare_upload_source(action.src)
        if prepared.mode != "bytes":
            raise ValueError(
                f"Action 'upload_bytes' requires prepared mode 'bytes', "
                f"got {prepared.mode!r}"
            )

        extra_args = action.extra.get("extra_args", {})
        body = prepared.payload

        self.client.put_object(
            Bucket=action.bucket_dst,
            Key=action.dst,
            Body=body,
            **extra_args,
        )

        return {
            "type": action.type,
            "bucket": action.bucket_dst,
            "key": action.dst,
            "size_hint": prepared.size_hint,
        }

    def _execute_load_object(self, action: TransferAction) -> dict[str, Any]:
        """
        Load a single S3 object into memory as raw bytes.

        Parameters
        ----------
        action:
            Planned S3 load action.

        Returns
        -------
        dict[str, Any]
            Execution metadata containing the source key, bucket, and raw payload.
        """
        response = self.client.get_object(
            Bucket=action.bucket_src,
            Key=action.src,
        )
        payload = response["Body"].read()

        return {
            "type": action.type,
            "bucket": action.bucket_src,
            "key": action.src,
            "payload": payload,
            "size_hint": len(payload),
        }

    def _execute_download_file(self, action: TransferAction) -> dict[str, Any]:
        """
        Execute a managed file download from S3.

        Parent directories are created automatically before downloading.

        Parameters
        ----------
        action:
            Download action mapping one S3 object to one local file path.

        Returns
        -------
        dict[str, Any]
            Execution metadata describing the completed download.
        """
        dst_path = Path(action.dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        self.client.download_file(
            Bucket=action.bucket_src,
            Key=action.src,
            Filename=str(dst_path),
            Config=self.transfer_config,
        )

        return {
            "type": action.type,
            "bucket": action.bucket_src,
            "key": action.src,
            "dst": str(dst_path),
        }

    def _execute_copy_s3(self, action: TransferAction) -> dict[str, Any]:
        """
        Execute a managed S3-to-S3 copy.

        This method uses boto3's managed copy operation, which can benefit from
        multipart copy behavior when object size exceeds the configured
        threshold.

        Parameters
        ----------
        action:
            Copy action mapping one S3 object to another S3 object.

        Returns
        -------
        dict[str, Any]
            Execution metadata describing the completed copy.
        """
        extra_args = action.extra.get("extra_args")

        copy_source = {
            "Bucket": action.bucket_src,
            "Key": action.src,
        }

        self.client.copy(
            CopySource=copy_source,
            Bucket=action.bucket_dst,
            Key=action.dst,
            ExtraArgs=extra_args,
            Config=self.transfer_config,
        )

        return {
            "type": action.type,
            "bucket_src": action.bucket_src,
            "key_src": action.src,
            "bucket_dst": action.bucket_dst,
            "key_dst": action.dst,
        }

    def _flush_delete_buffer(
        self,
        actions: Iterable[TransferAction],
    ) -> dict[str, Any]:
        """
        Execute a batch S3 deletion using ``delete_objects``.

        All actions in the batch must target the same source bucket.

        Parameters
        ----------
        actions:
            Delete actions to batch together.

        Returns
        -------
        dict[str, Any]
            Metadata summarizing the deletion batch.

        Raises
        ------
        ValueError
            If the batch is empty or spans multiple buckets.
        """
        actions = list(actions)
        if not actions:
            raise ValueError("Delete buffer is empty.")

        buckets = {action.bucket_src for action in actions}
        if len(buckets) != 1:
            raise ValueError(
                "All delete actions in a batch must target the same bucket."
            )

        bucket = actions[0].bucket_src
        objects = [{"Key": action.src} for action in actions]

        response = self.client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects, "Quiet": True},
        )

        deleted_count = len(response.get("Deleted", []))
        error_count = len(response.get("Errors", []))

        return {
            "type": "delete_batch",
            "bucket": bucket,
            "requested": len(actions),
            "deleted": deleted_count,
            "errors": error_count,
        }