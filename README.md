# better-aws

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#)
[![PyPI](https://img.shields.io/pypi/v/better-aws.svg)](https://pypi.org/project/better-aws/)

A minimal, production-minded wrapper around `boto3` focused on **S3 and tabular data (CSV/Parquet/Excel)**.

- **S3-first**: the handful of operations you use 90% of the time
- **Batch-native**: same methods for single key or lists
- **Ergonomic I/O**: `load()` → Python objects, `download()` → files
- **Logging-friendly**: standalone “print-like” logs or plug into your app logger
- **Auth-ready**: designed to support multiple auth modes (profile, custom files, static creds, .env)

> EC2 API is **work in progress**.

---

## Install

```bash
pip install better-aws
```

---

## Development (uv)

```bash
git clone https://github.com/thibault-charbonnier/better-aws.git
cd better-aws
uv sync
```

---

## Quickstart

```python
from better_aws import AWS

# 1) Create a session (boto3 will use the default credential chain unless you add other auth modes)
aws = AWS(profile="s3admin", region="eu-west-3", verbose=True)

# Optional sanity check
aws.identity(print_info=True)

# 2) Configure S3 defaults
aws.s3.config(
    bucket="my-bucket",
    key_prefix="my-project",   # optional: all keys are relative to this prefix
    output_type="pandas",      # tabular loads -> pandas (or "polars")
    file_type="parquet",       # default tabular format for dataframe uploads without extension
    overwrite=True,
)

# 3) List / load / upload
keys = aws.s3.list(prefix="raw/", limit=10)

df = aws.s3.load("raw/prices.parquet")     # -> pandas DataFrame (by config)
df["ret"] = df["close"].pct_change()

aws.s3.upload(df, "processed/prices_with_returns")  # -> parquet by default (by config)

# 4) Verify existence
print(aws.s3.exists("processed/prices_with_returns.parquet"))
```

---

## Core features

### 1) Authentication (design goal)

`better-aws` is built to keep auth **clean and modular**:

- AWS profile / default chain (AWS CLI-style)
- static credentials (Python args)
- custom `credentials_file` / optional `config_file`
- `.env` (dotenv)

```python
# Static credentials
aws = AWS("s3admin", aws_access_key_id=AWS_ID_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# .env config
aws = AWS("s3admin", env_file="test.env")

# Custom location for credentials files
aws = AWS("s3admin", config_file=r"\...\credentials")

# Classic CLI-like auth (boto3 fallback)
aws = AWS("s3admin")
```

---

### 2) Configure your S3 “workspace”

The main arguments:

- `bucket`: default bucket
- `key_prefix`: optional “root folder”
- `output_type`: tabular `load()` output (`pandas` / `polars`)
- `file_type`: default format for dataframe uploads without extension (`parquet` / `csv` / `xlsx`)
- `overwrite`: default overwrite policy

```python
aws.s3.config(bucket="my-bucket", key_prefix="research", output_type="polars", file_type="parquet", overwrite=False)
```

---

### 3) Read from S3

Two ways to read from S3 :

- `download()` = **S3 → local files**
- `load()` = **S3 → Python objects** (JSON → dict, tabular → DataFrame)

```python
aws.s3.download("reports/report.pdf", to="downloads/")

cfg = aws.s3.load("configs/pipeline.json")        # -> dict
df  = aws.s3.load("raw/prices.csv")               # -> pandas/polars (by config)
dfs = aws.s3.load(["raw/a.parquet", "raw/b.parquet"]) # -> List[pandas/polars] (by config)
```

> Batch native : `load()` or `download()` can be used for a single or a list of keys

---

### 4) Write to S3

- `upload()` supports:
  - local file → copied as-is
  - `dict` → JSON
  - pandas/polars DataFrame → CSV/Parquet/Excel (based on key extension or default `file_type`)

```python
aws.s3.upload("local/report.pdf", "reports/report.pdf")
aws.s3.upload({"run_id": 1}, "configs/run")              # -> configs/run.json
aws.s3.upload(df, "processed/table")                     # -> processed/table.parquet (default config)
aws.s3.upload([df, df], ["processed/a.parquet", "processed/b.parquet"])
```

> Batch native : `upload()` can be used for a single or a list of keys

---

### 5) Utilities

```python
aws.s3.exists("raw/prices.parquet")
aws.s3.list(prefix="raw/", with_meta=True)
aws.s3.delete(["tmp/a.parquet", "tmp/b.parquet"])
```

---

## API reference

### `AWS`

- `AWS(profile=None, region=None, logger=None, verbose=False, ...)`
- `aws.s3` (S3 wrapper)
- `aws.identity(print_info=False)`
- `aws.info(msg, *args)` (respects `verbose`)
- `aws.reset_session` (delete AWS session from cache)

### `S3`

- `config(bucket=None, key_prefix=None, output_type=None, file_type=None, overwrite=None, encoding=None, ...)`
- `list(prefix="", bucket=None, limit=None, recursive=True, with_meta=True)`
- `exists(key, bucket=None)`
- `load(key|[keys], bucket=None, output_type=None)`
- `download(key|[keys], to=None, bucket=None)`
- `upload(src|[src], key|[key], bucket=None, overwrite=None, ...)`
- `delete(key|[keys], bucket=None)`

> All function can overload the `bucket` parameter (and more params as `overwrite`, `output_type`...)

---

## Logging

- `verbose=False` → **no package logs**
- `verbose=True` → a few `info` messages (minimal, no spam)
- Pass your own logger to unify output with your app (e.g., Rich handler)

```python
import logging
from rich.logging import RichHandler
from better_aws.aws_wrapper import AWS

logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)
logger.handlers = [RichHandler(rich_tracebacks=True)]
logger.propagate = False

# Custom logger passed to the API
aws = AWS(profile="s3admin", region="eu-west-3", logger=logger, verbose=True)

# No log at all
aws = AWS(profile="s3admin", region="eu-west-3", verbose=False)

# Minimal "print-like" logs
aws = AWS(profile="s3admin", region="eu-west-3", verbose=True)
```

---

## Roadmap

- EC2 wrapper (start/stop/list instances)

---

## License

MIT License

Copyright (c) 2026 better-aws Contributors

See LICENSE file for details.
