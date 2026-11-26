"""Where the song data comes from.

Plan A: download songs.parquet from AWS S3 (when the AWS settings are present).
Plan B: use the copy saved in this repo (data/songs.parquet).
Plan C: use the small 10-song sample (data/songs.csv).

The app works the same either way. S3 is just the main source,
and the repo copy is the automatic fallback.
"""

import logging
import os
import tempfile
from typing import Optional, Tuple

import pandas as pd

from ai_explainer import _get_secret

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOCAL_PARQUET = os.path.join(DATA_DIR, "songs.parquet")
LOCAL_SAMPLE = os.path.join(DATA_DIR, "songs.csv")

# Where the S3 download is saved on the server (a temporary folder)
S3_DOWNLOAD_PATH = os.path.join(tempfile.gettempdir(), "music_engine_songs.parquet")


def s3_settings() -> Optional[dict]:
    """Reads the AWS settings. Returns None if any required one is missing."""
    bucket = _get_secret("S3_BUCKET")
    key_id = _get_secret("AWS_ACCESS_KEY_ID")
    secret = _get_secret("AWS_SECRET_ACCESS_KEY")
    if not (bucket and key_id and secret):
        return None
    return {
        "bucket": bucket,
        "object_key": _get_secret("S3_OBJECT_KEY") or "songs.parquet",
        "region": _get_secret("AWS_REGION") or "us-east-1",
        "key_id": key_id,
        "secret": secret,
    }


def download_from_s3(settings: dict, dest: str, client=None) -> None:
    """Downloads the song file from S3 and checks it is a real song table.
    Raises an error if anything goes wrong, so the caller can fall back."""
    if client is None:
        import boto3

        client = boto3.client(
            "s3",
            region_name=settings["region"],
            aws_access_key_id=settings["key_id"],
            aws_secret_access_key=settings["secret"],
        )

    # Download to a temporary name first, so a half-finished download is never used
    partial = dest + ".part"
    client.download_file(settings["bucket"], settings["object_key"], partial)

    # Make sure the file is a valid song table before trusting it
    columns = pd.read_parquet(partial, columns=["title", "genre"]).columns
    if not {"title", "genre"}.issubset(columns):
        raise ValueError("Downloaded file is missing song columns")

    os.replace(partial, dest)


def get_songs_path(client=None) -> Tuple[str, str]:
    """Returns (path to the song file, where it came from)."""
    settings = s3_settings()
    if settings:
        try:
            download_from_s3(settings, S3_DOWNLOAD_PATH, client=client)
            logger.info(f"Songs loaded from s3://{settings['bucket']}/{settings['object_key']}")
            return S3_DOWNLOAD_PATH, "AWS S3"
        except Exception as e:
            logger.warning(f"S3 download failed, using the repo copy instead: {e}")

    if os.path.exists(LOCAL_PARQUET):
        return LOCAL_PARQUET, "repo copy"
    return LOCAL_SAMPLE, "sample file"
