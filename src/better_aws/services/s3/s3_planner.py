"""
Planning layer for S3 operations.

This module defines the internal planning objects used to transform already
normalized inputs into executable transfer plans.

The planner does not perform any I/O, does not call boto3, and does not
resolve paths, buckets, glob patterns, or serialization strategies.
Its sole responsibility is to describe *what* must be executed through a
sequence of atomic actions grouped in a TransferPlan.

Typical flow:
    public API -> resolve / normalize inputs -> build plan -> execution engine

Examples of responsibilities handled here:
    - map local sources to S3 destination keys for uploads
    - map S3 keys to local destination paths for downloads
    - build copy + optional delete actions for transfers
    - build deletion plans from an explicit list of resolved keys

This separation keeps the planning layer deterministic, lightweight, and
fully reusable across the public S3 API.
"""

from dataclasses import dataclass, field
from typing import Any, Literal
from typing import Iterable
from pathlib import Path

ActionType = Literal[
    "upload_file",
    "upload_bytes",
    "download_file",
    "copy_s3",
    "delete_object",
    "delete_local",
    "load_object",
]


@dataclass(slots=True)
class TransferAction:
    """
    Describes a single atomic action to be performed as part of a transfer plan.

    Attributes
    ----------
    type : ActionType
        The type of action to be performed (e.g., "upload_file", "download_file", etc.).
    src : Any
        The source of the action.
    dst : Any
        The destination of the action.
    bucket_src : str | None
        The source bucket for S3 operations.
    bucket_dst : str | None
        The destination bucket for S3 operations.
    extra : dict[str, Any]
        Additional metadata for the action.

    """
    type: ActionType
    src: Any = None
    dst: Any = None
    bucket_src: str | None = None
    bucket_dst: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransferPlan:
    """
    Represents a sequence of TransferAction objects that together form a complete transfer plan.

    Attributes
    ----------
    actions : list[TransferAction]
        A list of TransferAction objects that describe the steps to be executed in the transfer plan.
    """
    actions: list[TransferAction] = field(default_factory=list)

    def add(self, action: TransferAction) -> None:
        self.actions.append(action)

    def extend(self, actions: list[TransferAction]) -> None:
        self.actions.extend(actions)

    def __len__(self) -> int:
        return len(self.actions)

    def is_empty(self) -> bool:
        return not self.actions
    
# -------------------------------------------------------
# |                     Plan builders                   |
# -------------------------------------------------------

def build_load_plan(
    *,
    keys: Iterable[str],
    bucket: str,
) -> TransferPlan:
    """
    Build a TransferPlan for loading one or more S3 objects into memory.

    This planner does not deserialize payloads. It only describes the S3 read
    operations required to fetch raw bytes from the specified keys.

    Parameters
    ----------
    keys : Iterable[str]
        S3 object keys to load.
    bucket : str
        S3 bucket containing the objects.

    Returns
    -------
    TransferPlan
        A TransferPlan containing one ``load_object`` action per key.
    """
    plan = TransferPlan()

    for key in keys:
        plan.add(
            TransferAction(
                type="load_object",
                src=key,
                bucket_src=bucket,
            )
        )

    return plan

def build_upload_plan(
    *,
    items: Iterable[dict[str, Any]],
    bucket: str,
) -> TransferPlan:
    """
    Build a TransferPlan for uploading resolved items to S3.

    Parameters
    ----------
    items : Iterable[dict[str, Any]]
        Planner-ready upload items of the form:
            {"src": <local path or PreparedUploadSource>, "key": <final s3 key>}
    bucket : str
        Target S3 bucket.

    Returns
    -------
    TransferPlan
        Transfer plan containing upload actions.
    """
    plan = TransferPlan()

    for item in items:
        src = item["src"]
        key = item["key"]

        if hasattr(src, "mode"):
            action_type = "upload_bytes" if src.mode == "bytes" else "upload_file"
        else:
            action_type = "upload_file"

        plan.add(
            TransferAction(
                type=action_type,
                src=src,
                dst=key,
                bucket_dst=bucket,
            )
        )

    return plan

def build_download_plan(
    *,
    keys: Iterable[str],
    dsts: Iterable[str | Path],
    bucket: str,
) -> TransferPlan:
    """
    Generates a TransferPlan for downloading a collection of S3 objects to local destinations.

    Parameters
    ----------
    keys : Iterable[str]
        An iterable of S3 object keys to be downloaded.
    dsts : Iterable[str | Path]
        An iterable of local destination paths corresponding to the S3 keys.
    bucket : str
        The S3 bucket from which the objects will be downloaded.

    Returns
    -------
    TransferPlan
        A TransferPlan object containing the sequence of actions required to perform the downloads.
    """
    plan = TransferPlan()

    for key, dst in zip(keys, dsts, strict=True):
        plan.add(
            TransferAction(
                type="download_file",
                src=key,
                dst=str(dst),
                bucket_src=bucket,
            )
        )

    return plan

def build_delete_plan(
    *,
    keys: Iterable[str],
    bucket: str,
) -> TransferPlan:
    """
    Generates a TransferPlan for deleting a collection of S3 objects.

    Parameters
    ----------
    keys : Iterable[str]
        An iterable of S3 object keys to be deleted.
    bucket : str
        The S3 bucket from which the objects will be deleted.

    Returns
    -------
    TransferPlan
        A TransferPlan object containing the sequence of actions required to perform the deletions.
    """
    plan = TransferPlan()

    for key in keys:
        plan.add(
            TransferAction(
                type="delete_object",
                src=key,
                bucket_src=bucket,
            )
        )

    return plan

def build_s3_to_s3_transfer_plan(
    *,
    src_keys: Iterable[str],
    dst_keys: Iterable[str],
    bucket_src: str,
    bucket_dst: str,
    move: bool = True,
) -> TransferPlan:
    """
    Generates a TransferPlan for transferring a collection of S3 objects from a source bucket to a destination bucket,
    with an optional move (delete) action.

    Parameters
    ----------
    src_keys : Iterable[str]
        An iterable of S3 object keys to be transferred from the source bucket.
    dst_keys : Iterable[str]
        An iterable of S3 object keys corresponding to the destination keys in the destination bucket.
    bucket_src : str
        The S3 bucket from which the objects will be transferred.
    bucket_dst : str
        The S3 bucket to which the objects will be transferred.
    move : bool, optional
        Whether to include delete actions for the source objects after the transfer. Defaults to True.

    Returns
    -------
    TransferPlan
        A TransferPlan object containing the sequence of actions required to perform the transfers and optional deletions.
    """
    plan = TransferPlan()

    src_keys = list(src_keys)
    dst_keys = list(dst_keys)

    for src_key, dst_key in zip(src_keys, dst_keys, strict=True):
        plan.add(
            TransferAction(
                type="copy_s3",
                src=src_key,
                dst=dst_key,
                bucket_src=bucket_src,
                bucket_dst=bucket_dst,
            )
        )

    if move:
        for src_key in src_keys:
            plan.add(
                TransferAction(
                    type="delete_object",
                    src=src_key,
                    bucket_src=bucket_src,
                )
            )

    return plan

def build_local_to_s3_transfer_plan(
    *,
    srcs: Iterable[str | Path],
    dst_keys: Iterable[str],
    bucket_dst: str,
    move: bool = False,
) -> TransferPlan:
    """
    Generates a TransferPlan for transferring a collection of local sources to S3 destination keys,
    with an optional move (delete) action.

    Parameters
    ----------
    srcs : Iterable[str | Path]
        An iterable of local sources to be transferred. Each source should be a file path (str or Path).
    dst_keys : Iterable[str]
        An iterable of S3 object keys corresponding to the local sources.
    bucket_dst : str
        The S3 bucket to which the sources will be transferred.
    move : bool, optional
        Whether to include delete actions for the local sources after the transfer. Defaults to False.
    
    Returns
    -------
    TransferPlan
        A TransferPlan object containing the sequence of actions required to perform the transfers and optional deletions.
    """
    plan = TransferPlan()

    srcs = [str(s) for s in srcs]
    dst_keys = list(dst_keys)

    for src, dst_key in zip(srcs, dst_keys, strict=True):
        plan.add(
            TransferAction(
                type="upload_file",
                src=src,
                dst=dst_key,
                bucket_dst=bucket_dst,
            )
        )

    if move:
        for src in srcs:
            plan.add(
                TransferAction(
                    type="delete_local",
                    src=src,
                )
            )

    return plan

def build_s3_to_local_transfer_plan(
    *,
    src_keys: Iterable[str],
    dsts: Iterable[str | Path],
    bucket_src: str,
    move: bool = False,
) -> TransferPlan:
    """
    Generates a TransferPlan for transferring a collection of S3 objects to local destinations,
    with an optional move (delete) action.

    Parameters
    ----------
    src_keys : Iterable[str]
        An iterable of S3 object keys to be transferred.
    dsts : Iterable[str | Path]
        An iterable of local destination paths corresponding to the S3 keys.
    bucket_src : str
        The S3 bucket from which the objects will be transferred.
    move : bool, optional
        Whether to include delete actions for the S3 sources after the transfer. Defaults to False.

    Returns
    -------
    TransferPlan
        A TransferPlan object containing the sequence of actions required to perform the transfers and optional deletions.
    """
    plan = TransferPlan()

    src_keys = list(src_keys)
    dsts = [str(d) for d in dsts]

    for src_key, dst in zip(src_keys, dsts, strict=True):
        plan.add(
            TransferAction(
                type="download_file",
                src=src_key,
                dst=dst,
                bucket_src=bucket_src,
            )
        )

    if move:
        for src_key in src_keys:
            plan.add(
                TransferAction(
                    type="delete_object",
                    src=src_key,
                    bucket_src=bucket_src,
                )
            )

    return plan

def build_transfer_plan(
    *,
    mode: str,
    srcs: list[str],
    dsts: list[str],
    bucket_src: str | None = None,
    bucket_dst: str | None = None,
    move: bool = True,
) -> TransferPlan:
    """
    Builds a TransferPlan for tree-based transfers between S3 and local filesystems.
    This function serves as a high-level entry point for generating transfer plans based on the specified mode of transfer.

    Parameters
    ----------
    mode : str
        The transfer mode, which can be "s3_to_s3", "s3_to_local", or "local_to_s3".
    srcs : list[str]
        A list of source paths or keys, depending on the mode.
    dsts : list[str]
        A list of destination paths or keys, depending on the mode.
    bucket_src : str | None, optional
        The source bucket for S3 operations, required for "s3_to_s3" and "s3_to_local" modes. Defaults to None.
    bucket_dst : str | None, optional
        The destination bucket for S3 operations, required for "s3_to_s3" and "local_to_s3" modes. Defaults to None.
    move : bool, optional
        Whether to include delete actions for the sources after the transfer. Defaults to True.

    Returns
    -------
    TransferPlan
        A TransferPlan object containing the sequence of actions required to perform the transfer.
    """
    if mode == "s3_to_s3":
        return build_s3_to_s3_transfer_plan(
            src_keys=srcs,
            dst_keys=dsts,
            bucket_src=bucket_src or "",
            bucket_dst=bucket_dst or "",
            move=move,
        )

    if mode == "s3_to_local":
        return build_s3_to_local_transfer_plan(
            src_keys=srcs,
            dsts=dsts,
            bucket_src=bucket_src or "",
            move=move,
        )

    if mode == "local_to_s3":
        return build_local_to_s3_transfer_plan(
            srcs=srcs,
            dst_keys=dsts,
            bucket_dst=bucket_dst or "",
            move=move,
        )

    raise ValueError(f"Unsupported transfer mode: {mode}")