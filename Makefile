.PHONY: sync sync-notebooks check test book smoke-notebooks smoke-module serve clean

sync:
	uv sync --frozen

sync-notebooks:
	uv python install 3.11
	UV_PROJECT_ENVIRONMENT=.venv-notebooks uv sync --managed-python --frozen --group notebooks

check:
	uv run python scripts/check_book_structure.py

test:
	uv run pytest

book: check test
	uv run jupyter-book clean .
	uv run jupyter-book build . --all -W

smoke-notebooks:
	UV_PROJECT_ENVIRONMENT=.venv-notebooks uv run --managed-python --frozen --group notebooks python scripts/smoke_test_notebooks.py

smoke-module:
	@test -n "$(MODULE)" || (echo "Use: make smoke-module MODULE=module-id" && exit 2)
	UV_PROJECT_ENVIRONMENT=.venv-notebooks uv run --managed-python --frozen --group notebooks python scripts/smoke_test_notebooks.py --module "$(MODULE)"

serve:
	uv run python -m http.server 8000 -d _build/html

clean:
	uv run jupyter-book clean .
