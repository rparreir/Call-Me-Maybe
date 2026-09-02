
install:
	uv sync
run:
	uv run python -m src

download:
	uv run python -c "from llm_sdk import Small_LLM_Model; Small_LLM_Model()"

debug:
	uv run python -m pdb -m src

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 . 
	mypy . --strict

clean:
	rm -rf __pycache__ .mypy_cache .venv

.PHONY: install run debug lint lint-strict clean
