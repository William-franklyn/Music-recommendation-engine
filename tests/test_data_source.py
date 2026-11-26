import os

import pandas as pd

import data_source

AWS_SETTINGS = {
    "S3_BUCKET": "test-bucket",
    "AWS_ACCESS_KEY_ID": "test-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
}


class FakeS3:
    """Pretends to be S3, so the tests never touch the internet or a real account."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def download_file(self, bucket, key, dest):
        self.calls.append((bucket, key))
        if self.fail:
            raise ConnectionError("S3 is unreachable")
        pd.DataFrame({"title": ["Song"], "genre": ["pop"]}).to_parquet(dest)


def _use_aws(monkeypatch, tmp_path):
    for name, value in AWS_SETTINGS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(data_source, "S3_DOWNLOAD_PATH", str(tmp_path / "songs.parquet"))


def test_uses_repo_copy_when_aws_is_not_set(monkeypatch):
    for name in AWS_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(data_source, "_get_secret", lambda name: os.getenv(name))
    path, source = data_source.get_songs_path()
    assert source in ("repo copy", "sample file")
    assert os.path.exists(path)


def test_downloads_from_s3_when_aws_is_set(monkeypatch, tmp_path):
    _use_aws(monkeypatch, tmp_path)
    fake = FakeS3()
    path, source = data_source.get_songs_path(client=fake)
    assert source == "AWS S3"
    assert fake.calls == [("test-bucket", "songs.parquet")]
    assert pd.read_parquet(path)["title"].tolist() == ["Song"]


def test_falls_back_when_s3_fails(monkeypatch, tmp_path):
    _use_aws(monkeypatch, tmp_path)
    path, source = data_source.get_songs_path(client=FakeS3(fail=True))
    assert source != "AWS S3"
    assert os.path.exists(path)
