from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from helio_data_methods import data


def _checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        data.fetch_dataset("not-a-dataset")


def test_plasma_sheet_dataset_resolves_verified_compact_files(tmp_path: Path) -> None:
    resolved = data.fetch_dataset("plasma-sheet-prime-tm03", cache_dir=tmp_path)
    assert set(resolved) == {
        "plasma_sheet_prime_tm03_results.npz",
        "manifest.json",
    }
    for filename, path in resolved.items():
        expected = data.DATASETS["plasma-sheet-prime-tm03"][filename][1]
        assert data._sha256(path) == expected


def test_fetch_uses_verified_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified teaching fixture"
    dataset_root = tmp_path / "fixture"
    dataset_root.mkdir()
    (dataset_root / "sample.dat").write_bytes(payload)
    monkeypatch.setitem(
        data.DATASETS,
        "fixture",
        {"sample.dat": ("data/fixture/sample.dat", _checksum(payload))},
    )
    monkeypatch.setenv("HELIO_DATA_DIR", str(dataset_root))

    resolved = data.fetch_dataset("fixture", cache_dir=tmp_path / "cache")

    assert resolved == {"sample.dat": dataset_root / "sample.dat"}


def test_fetch_downloads_then_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"downloaded teaching fixture"
    monkeypatch.setitem(
        data.DATASETS,
        "fixture",
        {"sample.dat": ("data/fixture/sample.dat", _checksum(payload))},
    )
    monkeypatch.setattr(data, "_repository_root", lambda: None)
    calls: list[str] = []

    def fake_download(url: str, target: Path) -> None:
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    monkeypatch.setattr(data, "_download", fake_download)
    first = data.fetch_dataset("fixture", cache_dir=tmp_path)
    second = data.fetch_dataset("fixture", cache_dir=tmp_path)

    assert first == second
    assert len(calls) == 1
    assert "data/fixture/sample.dat" in calls[0]


def test_checksum_failure_removes_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        data.DATASETS,
        "fixture",
        {"sample.dat": ("data/fixture/sample.dat", _checksum(b"expected"))},
    )
    monkeypatch.setattr(data, "_repository_root", lambda: None)

    def fake_download(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"wrong")

    monkeypatch.setattr(data, "_download", fake_download)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.fetch_dataset("fixture", cache_dir=tmp_path)
    assert not (tmp_path / "datasets" / "fixture" / "sample.dat").exists()


def test_download_error_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sample.dat"

    def unavailable(*args: object, **kwargs: object) -> object:
        raise TimeoutError("offline")

    monkeypatch.setattr(data, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="HELIO_DATA_DIR"):
        data._download("https://example.invalid/sample.dat", target)
