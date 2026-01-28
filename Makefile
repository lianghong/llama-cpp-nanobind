.PHONY: all build test wheel wheel-portable clean install dev

PYTHON ?= python3.14
VENV := .venv

all: build

$(VENV):
	uv venv --python $(PYTHON) $(VENV)

dev: $(VENV)
	. $(VENV)/bin/activate && uv pip install -e .[test]

build: $(VENV)
	. $(VENV)/bin/activate && uv pip install -e .

test: dev
	. $(VENV)/bin/activate && uv run pytest -q

wheel: $(VENV)
	. $(VENV)/bin/activate && uv pip install build && python -m build --wheel

wheel-portable: $(VENV)
	. $(VENV)/bin/activate && uv pip install build && CMAKE_ARGS="-DLLAMA_PORTABLE=ON" python -m build --wheel

clean:
	rm -rf build-debug build dist *.egg-info
	rm -rf src/llama_cpp/*.so src/llama_cpp/__pycache__
	rm -rf .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install: wheel
	uv pip install dist/*.whl
