# Ambidex — dev entrypoints (plain commands work too, see README)

PY ?= python

setup:
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install -U pip
	.venv/Scripts/python -m pip install -e ".[dev]"

fetch-pan15:
	$(PY) eval/fetchers/fetch_pan15.py --langs en

smoke:
	$(PY) -m harness.cli smoke

verify-pan15-en:
	$(PY) -m harness.cli verify --pan15 data/raw/pan15/english --lang en --name pan15-en

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
