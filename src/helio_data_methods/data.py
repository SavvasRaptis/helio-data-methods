"""Versioned data access for the teaching notebooks.

Teaching datasets live under ``data/``. Local checkouts use those files
directly; standalone and Colab notebooks download only the files declared for
their dataset and verify every payload before use.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


REPOSITORY: Final = "SavvasRaptis/helio-data-methods"
DEFAULT_REF: Final = "main"

DATASETS: Final[dict[str, dict[str, tuple[str, str]]]] = {
    "dst-omni-2010-2015": {
        "omni2_2010-2015.dat": (
            "data/dst-omni-2010-2015/omni2_2010-2015.dat",
            "18a4ce192bdcc481bdef699a6e11f7f0441b4e933def8dd0c9cd25fc766bcecf",
        ),
    },
    "sep-curated": {
        "x_train.pkl": (
            "data/sep-curated/x_train.pkl",
            "e809bf00498633f509a223d61f9b0006e6ed1803f6de22118bcf654f2ce8ba3b",
        ),
        "x_test.pkl": (
            "data/sep-curated/x_test.pkl",
            "1d0c5f84713d4fde34d567cdb62e9081c4d723f6fef9abd543137376350d5955",
        ),
        "y_train.pkl": (
            "data/sep-curated/y_train.pkl",
            "d7aa048f6b081a9fb1fc00dde19872c0f67ae5b4c8620daa5984b679f9f9dbdc",
        ),
        "y_test.pkl": (
            "data/sep-curated/y_test.pkl",
            "d44c5af108bab2b19f5f8082548282edd8aee89d57e15469516de1ca3f400ee5",
        ),
    },
    "coronal-loops": {
        "X2D.npy": (
            "data/coronal-loops/X2D.npy",
            "48fdc54c387642ed1dba563b3a27e5aa666327689d0b8db67355aa3015c0d658",
        ),
        "Y2D.npy": (
            "data/coronal-loops/Y2D.npy",
            "f5500bd93a542facd44bb0710abc2880bb0ea8ff7e769f64eb8332898c1bd35b",
        ),
        "LNGTH_L2D.npy": (
            "data/coronal-loops/LNGTH_L2D.npy",
            "38511b43978701e420cfd262fb98dfa2c816dbe8bb7f2ecd2759124073beb988",
        ),
        "DST2D_FP.npy": (
            "data/coronal-loops/DST2D_FP.npy",
            "ff6e05c3679aa876beb390a2fe4531992e2752360b66fe0bcfbb8057bd9c6a0e",
        ),
        "angle_top.npy": (
            "data/coronal-loops/angle_top.npy",
            "4fb89537bb58b44f75ff222dbb53ea8596b1f2900657bbbf3d0eff2fc45796b2",
        ),
        "Z3D.npy": (
            "data/coronal-loops/Z3D.npy",
            "9c04fd96c1c158607259cfbc9ac8e758cfaaf50f627622ee3e9761323b6b8bac",
        ),
        "coronal loop.jpg": (
            "data/coronal-loops/coronal loop.jpg",
            "c2829f86ed9a71c4ab31f0993701af7a65e9ba469c89c98334ab39c824dfe48c",
        ),
    },
    "plasma-sheet-prime-tm03": {
        "plasma_sheet_prime_tm03_results.npz": (
            "data/plasma-sheet-prime-tm03/plasma_sheet_prime_tm03_results.npz",
            "ed9392dcdc305f0275459efb52e3fd15e80fd0be9b76b7a1816b5177cf2ba604",
        ),
        "manifest.json": (
            "data/plasma-sheet-prime-tm03/manifest.json",
            "f845dd71a644bcf08c591df39431f08b6c6a5f33577e782e2e8f29030bbecadc",
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected: str) -> bool:
    return path.is_file() and _sha256(path) == expected


def _repository_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "data").is_dir() else None


def _local_override(dataset_id: str, filename: str) -> Path | None:
    value = os.getenv("HELIO_DATA_DIR")
    if not value:
        return None
    root = Path(value).expanduser()
    for candidate in (root / dataset_id / filename, root / filename):
        if candidate.is_file():
            return candidate
    return None


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, temporary)
        Path(temporary_name).replace(target)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Check network access or set HELIO_DATA_DIR "
            "to a directory containing the required files."
        ) from exc


def fetch_dataset(
    dataset_id: str,
    cache_dir: str | Path | None = None,
    repo_ref: str | None = None,
) -> dict[str, Path]:
    """Resolve and verify every file belonging to ``dataset_id``.

    Parameters
    ----------
    dataset_id:
        One of the keys in :data:`DATASETS`.
    cache_dir:
        Optional cache root. Defaults to ``HELIO_DATA_CACHE`` or
        ``~/.cache/helio-data-methods``.
    repo_ref:
        Git ref used for raw GitHub downloads. Defaults to
        ``HELIO_DATA_REF`` or ``main``.
    """

    if dataset_id not in DATASETS:
        allowed = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {dataset_id!r}; choose one of: {allowed}")

    cache_root = Path(
        cache_dir
        or os.getenv("HELIO_DATA_CACHE", Path.home() / ".cache" / "helio-data-methods")
    ).expanduser()
    ref = repo_ref or os.getenv("HELIO_DATA_REF", DEFAULT_REF)
    repository_root = _repository_root()
    resolved: dict[str, Path] = {}

    for filename, (relative_source, checksum) in DATASETS[dataset_id].items():
        candidates: list[Path] = []
        override = _local_override(dataset_id, filename)
        if override is not None:
            candidates.append(override)
        if repository_root is not None:
            candidates.append(repository_root / relative_source)
        cached = cache_root / "datasets" / dataset_id / filename
        candidates.append(cached)

        match = next((path for path in candidates if _verified(path, checksum)), None)
        if match is None:
            encoded_path = quote(relative_source, safe="/")
            encoded_ref = quote(ref, safe="")
            url = (
                f"https://raw.githubusercontent.com/{REPOSITORY}/"
                f"{encoded_ref}/{encoded_path}"
            )
            _download(url, cached)
            if not _verified(cached, checksum):
                cached.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {dataset_id}/{filename}; the downloaded "
                    "file was removed. Verify HELIO_DATA_REF and the dataset manifest."
                )
            match = cached
        resolved[filename] = match

    return resolved


__all__ = ["DATASETS", "fetch_dataset"]
