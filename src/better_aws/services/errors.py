from __future__ import annotations
from typing import Optional
from botocore.exceptions import ClientError

class BetterAWSError(RuntimeError): ...
class S3Error(BetterAWSError): ...
class S3NotFound(S3Error): ...
class S3AccessDenied(S3Error): ...
class S3UnsupportedFormat(S3Error): ...
class MissingOptionalDependency(S3Error): ...

def _err_code(e: ClientError) -> str:
    return e.response.get("Error", {}).get("Code", "Unknown")


def _raise_s3(e: ClientError, *, bucket: str, key: Optional[str] = None) -> None:
    code = _err_code(e)
    msg = e.response.get("Error", {}).get("Message", str(e))
    path = f"s3://{bucket}/{key}" if key else f"s3://{bucket}"

    if code in {"NoSuchKey", "404", "NotFound"}:
        raise S3NotFound(f"{path} not found ({code}): {msg}") from e
    if code in {"AccessDenied", "403"}:
        raise S3AccessDenied(f"Access denied to {path} ({code}): {msg}") from e

    raise S3Error(f"S3 error on {path} ({code}): {msg}") from e