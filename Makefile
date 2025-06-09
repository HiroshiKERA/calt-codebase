.PHONY: lint
lint:
	uv sync --dev
	uv run ruff check
