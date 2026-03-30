"""
Integration tests for better-aws S3 V2.

These tests are designed to be run locally against a real S3 bucket.
They create only fake data under a dedicated test prefix, then clean up.

Usage with uv
-------------
1) Install test deps if needed:
   uv add --dev pytest pandas pyarrow openpyxl

2) Export AWS credentials as usual (profile / env vars).

3) Run:
   uv run pytest -s -vv tests/test_s3_integration.py --durations=20

Optional environment variables
------------------------------
BETTER_AWS_TEST_BUCKET   S3 bucket to use. Default: "thib-quant"
BETTER_AWS_TEST_PREFIX   Prefix root to use. Default: "research/test"
BETTER_AWS_TEST_REGION   Region hint if you want to pass one to AWS(...)

Notes
-----
- These are integration tests, not pure unit tests.
- They assume your public API roughly exposes:
      from better_aws import AWS
      aws = AWS(...)
      aws.s3.config(...)
      aws.s3.upload(...)
      aws.s3.load(...)
      aws.s3.download(...)
      aws.s3.transfer(...)
      aws.s3.delete(...)
      aws.s3.list(...)
      aws.s3.exists(...)
      aws.s3.tree(...)
- If your AWS(...) signature changed, only the fixture `aws_client()` should
  need a small adjustment.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


from better_aws import AWS


def _bucket() -> str:
    return os.getenv("BETTER_AWS_TEST_BUCKET", "thib-quant")


def _prefix_root() -> str:
    return os.getenv("BETTER_AWS_TEST_PREFIX", "research/test").strip("/")


def _region() -> str | None:
    return os.getenv("BETTER_AWS_TEST_REGION")


def _run_prefix() -> str:
    return f"{_prefix_root()}/pytest/{uuid.uuid4().hex[:10]}"


class StepTimer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"\n[START] {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        status = "OK" if exc is None else "FAIL"
        print(f"[{status}] {self.label} in {dt:.3f}s")


@pytest.fixture(scope="session")
def aws_client():
    region = _region()

    kwargs = {}
    if region:
        kwargs["region_name"] = region

    aws = AWS(**kwargs)

    aws.s3.config(
        bucket=_bucket(),
        key_prefix="",
        file_type="parquet",
        output_type="pandas",
        overwrite=True,
        csv_index=False,
        parquet_index=None,
        excel_index=False,
    )
    return aws


@pytest.fixture
def test_prefix(aws_client):
    prefix = _run_prefix()
    yield prefix
    try:
        aws_client.s3.delete(f"{prefix}/**", force=True)
    except Exception as exc:
        print(f"[WARN] Cleanup failed for prefix {prefix}: {exc}")


@pytest.fixture
def local_tmp(tmp_path: Path) -> Path:
    return tmp_path


def _s3_key(prefix: str, *parts: str) -> str:
    clean = [p.strip("/") for p in parts if p]
    return "/".join([prefix.strip("/")] + clean)


def _make_df():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, None, 4.5],
            "label": ["a", "b", "c", "d"],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    df.index.name = "observation_date"
    return df


def _make_big_binary(path: Path, size_mb: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = b"0123456789abcdef" * 65536
    with path.open("wb") as f:
        for _ in range(size_mb):
            f.write(chunk)
    return path


def test_upload_load_parquet_roundtrip_preserves_index(aws_client, test_prefix):
    pd = pytest.importorskip("pandas")
    df = _make_df()
    key = _s3_key(test_prefix, "roundtrip", "indexed.parquet")

    with StepTimer("upload parquet dataframe"):
        returned = aws_client.s3.upload(df, key=key)
        assert returned == key

    with StepTimer("load parquet dataframe"):
        loaded = aws_client.s3.load(key)

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.index.name == "observation_date"
    assert list(loaded.columns) == list(df.columns)
    pd.testing.assert_index_equal(loaded.index, df.index)
    pd.testing.assert_frame_equal(loaded, df)


def test_upload_same_object_to_multiple_keys(aws_client, test_prefix):
    pd = pytest.importorskip("pandas")
    df = _make_df()
    keys = [
        _s3_key(test_prefix, "multi", "a.parquet"),
        _s3_key(test_prefix, "multi", "b.parquet"),
        _s3_key(test_prefix, "multi", "c.parquet"),
    ]

    with StepTimer("broadcast one dataframe to multiple keys"):
        returned = aws_client.s3.upload(df, key=keys)

    assert returned == keys
    for k in keys:
        assert aws_client.s3.exists(k)


def test_upload_local_glob_to_single_prefix_and_load_via_glob(aws_client, test_prefix, local_tmp):
    pd = pytest.importorskip("pandas")

    src_dir = local_tmp / "glob_upload"
    src_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        df = pd.DataFrame({"x": [i, i + 1], "y": [10, 20]})
        df.to_csv(src_dir / f"file_{i}.csv", index=False)

    pattern = str(src_dir / "*.csv")
    dst_prefix = _s3_key(test_prefix, "glob_csv")

    with StepTimer("upload local glob to s3 prefix"):
        returned = aws_client.s3.upload(pattern, key=dst_prefix)

    assert isinstance(returned, list)
    assert len(returned) == 3

    with StepTimer("load from s3 glob"):
        loaded = aws_client.s3.load(f"{dst_prefix}/*.csv")

    assert isinstance(loaded, list)
    assert len(loaded) == 3


def test_download_single_file(aws_client, test_prefix, local_tmp):
    pd = pytest.importorskip("pandas")
    df = _make_df()
    key = _s3_key(test_prefix, "download", "single.parquet")
    out_path = local_tmp / "single_download.parquet"

    aws_client.s3.upload(df, key=key)

    with StepTimer("download single file"):
        downloaded = aws_client.s3.download(key, to=out_path)

    assert Path(downloaded).exists()
    assert Path(downloaded).is_file()


def test_download_glob_preserve_prefix(aws_client, test_prefix, local_tmp):
    pd = pytest.importorskip("pandas")
    prefix = _s3_key(test_prefix, "dl_tree")

    keys = [
        _s3_key(prefix, "a", "x.parquet"),
        _s3_key(prefix, "b", "y.parquet"),
    ]
    for i, k in enumerate(keys):
        aws_client.s3.upload(pd.DataFrame({"v": [i]}), key=k)

    out_dir = local_tmp / "downloads"

    with StepTimer("download glob preserve prefix"):
        downloaded = aws_client.s3.download(f"{prefix}/**/*.parquet", to=out_dir, preserve_prefix=True)

    assert isinstance(downloaded, list)
    assert len(downloaded) == 2
    assert (out_dir / prefix / "a" / "x.parquet").exists()
    assert (out_dir / prefix / "b" / "y.parquet").exists()


def test_transfer_local_to_s3_move(aws_client, test_prefix, local_tmp):
    local_file = local_tmp / "to_transfer" / "sample.bin"
    _make_big_binary(local_file, size_mb=6)
    dst_prefix = _s3_key(test_prefix, "transfer_local_to_s3")

    with StepTimer("transfer local to s3 with move"):
        result = aws_client.s3.transfer(str(local_file), dst_prefix, move=True)

    assert isinstance(result, str)
    assert aws_client.s3.exists(result)
    assert not local_file.exists()


def test_transfer_s3_to_local(aws_client, test_prefix, local_tmp):
    pd = pytest.importorskip("pandas")
    src_key = _s3_key(test_prefix, "transfer_s3_to_local", "one.parquet")
    aws_client.s3.upload(pd.DataFrame({"a": [1, 2]}), key=src_key)

    dst_dir = local_tmp / "s3_to_local"

    with StepTimer("transfer s3 to local"):
        result = aws_client.s3.transfer(src_key, str(dst_dir), move=False)

    if isinstance(result, list):
        result = result[0]
    assert Path(result).exists()


def test_transfer_s3_to_s3_move(aws_client, test_prefix):
    pd = pytest.importorskip("pandas")
    src_key = _s3_key(test_prefix, "transfer_s3_to_s3", "src.parquet")
    dst_root = _s3_key(test_prefix, "transfer_s3_to_s3_dst")
    aws_client.s3.upload(pd.DataFrame({"a": [42]}), key=src_key)

    with StepTimer("transfer s3 to s3 with move"):
        result = aws_client.s3.transfer(src_key, dst_root, move=True)

    if isinstance(result, list):
        result = result[0]

    assert aws_client.s3.exists(result)
    assert not aws_client.s3.exists(src_key)


def test_list_exists_tree_smoke(aws_client, test_prefix):
    pd = pytest.importorskip("pandas")
    key = _s3_key(test_prefix, "tree", "smoke.parquet")
    aws_client.s3.upload(pd.DataFrame({"a": [1]}), key=key)

    with StepTimer("list"):
        listed = aws_client.s3.list(prefix=test_prefix, recursive=True, with_meta=True)
    assert listed

    with StepTimer("exists"):
        assert aws_client.s3.exists(key)

    with StepTimer("tree"):
        aws_client.s3.tree(prefix=test_prefix, max_children=5)

def test_delete_glob_force(aws_client, test_prefix):
    pd = pytest.importorskip("pandas")
    base = _s3_key(test_prefix, "delete_glob")
    keys = [
        _s3_key(base, "a.parquet"),
        _s3_key(base, "b.parquet"),
    ]
    for i, k in enumerate(keys):
        aws_client.s3.upload(pd.DataFrame({"x": [i]}), key=k)

    with StepTimer("delete glob force"):
        aws_client.s3.delete(f"{base}/*.parquet", force=True)

    for k in keys:
        assert not aws_client.s3.exists(k)


def test_large_local_file_upload_and_download(aws_client, test_prefix, local_tmp):
    local_file = _make_big_binary(local_tmp / "large" / "big.bin", size_mb=12)
    key = _s3_key(test_prefix, "large", "big.bin")

    with StepTimer("upload large local file"):
        returned = aws_client.s3.upload(str(local_file), key=key)
    assert returned == key
    assert aws_client.s3.exists(key)

    out_dir = local_tmp / "large_download"

    with StepTimer("download large local file"):
        downloaded = aws_client.s3.download(key, to=out_dir)

    if isinstance(downloaded, list):
        downloaded = downloaded[0]

    downloaded_path = Path(downloaded)
    assert downloaded_path.exists()
    assert downloaded_path.stat().st_size == local_file.stat().st_size


if __name__ == "__main__":
    import sys
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-s", "-vv", "--durations=20", *sys.argv[1:]]))
