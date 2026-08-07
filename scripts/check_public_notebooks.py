"""Verify that published notebooks resolve from the configured GitHub repository."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import nbformat

from smoke_test_notebooks import ROOT, published_notebooks


REPOSITORY = "SavvasRaptis/helio-data-methods"


def main() -> None:
    ref = os.getenv("HELIO_DATA_REF", "main")
    token = os.getenv("GITHUB_TOKEN")
    failures: list[str] = []
    for path in published_notebooks(None):
        relative = path.relative_to(ROOT).as_posix()
        url = (
            f"https://api.github.com/repos/{REPOSITORY}/contents/"
            f"{quote(relative, safe='/')}?ref={quote(ref, safe='')}"
        )
        try:
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "helio-data-methods-ci",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    failures.append(f"{relative}: HTTP {response.status}")
        except Exception as exc:
            failures.append(f"{relative}: {exc}")

        notebook = nbformat.read(path, as_version=4)
        runtime = notebook.metadata["helio_data_methods"]["runtime"]
        if "colab" not in runtime:
            failures.append(f"{relative}: missing Colab runtime declaration")

    if failures:
        raise SystemExit("GitHub notebook verification failed:\n- " + "\n- ".join(failures))
    print(f"verified {len(published_notebooks(None))} GitHub notebook URLs at {ref}")


if __name__ == "__main__":
    main()
