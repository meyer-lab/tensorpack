.PHONY: clean test ty vulture

all: test

test: .venv
	uv run pytest -s -v -x

.venv:
	uv sync

ty: .venv
	uv run ty check

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=tensorpack --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml junit.xml .coverage

vulture: .venv
	uv run vulture
