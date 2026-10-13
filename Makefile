
install:
	uv sync
run:
	uv run python -m src

download:
	uv run python -c "from llm_sdk import Small_LLM_Model; Small_LLM_Model()"

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	rm -rf src/__pycache__ .mypy_cache llm_sdk/llm_sdk/__pycache__

.PHONY: install run download debug lint lint-strict clean
