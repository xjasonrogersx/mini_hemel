#!/usr/bin/env python3
import mimetypes
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

CONFIG_PATH = Path(__file__).with_name("config.json")
PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-e615b9910ad849b2a11f2ca22ba7869b.r2.dev",
).rstrip("/")


def load_r2_config() -> dict[str, str]:
    try:
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            r2_config = json.load(config_file)["r2"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        fail(f"Invalid R2 configuration in {CONFIG_PATH}: {exc}")

    return {
        "account_id": os.getenv("R2_ACCOUNT_ID", r2_config["account_id"]),
        "access_key": os.getenv("R2_ACCESS_KEY_ID", r2_config["access_key"]),
        "secret_key": os.getenv("R2_SECRET_ACCESS_KEY", r2_config["secret_key"]),
        "bucket": os.getenv("R2_BUCKET", r2_config["bucket"]),
    }


def fail(msg: str, code: int = 1):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    if len(sys.argv) != 2:
        fail("Usage: python push.py <file>", 2)

    file_path = Path(sys.argv[1]).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        fail(f"File not found: {file_path}")
    r2_config = load_r2_config()

    object_key = f"images/{file_path.name}"
    content_type, _ = mimetypes.guess_type(file_path.name)
    content_type = content_type or "application/octet-stream"

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_config['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_config["access_key"],
        aws_secret_access_key=r2_config["secret_key"],
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )

    with file_path.open("rb") as f:
        s3.upload_fileobj(
            f,
            r2_config["bucket"],
            object_key,
            ExtraArgs={"ContentType": content_type},
        )

    print(f"{PUBLIC_BASE_URL}/{object_key}")


if __name__ == "__main__":
    main()
