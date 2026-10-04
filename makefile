.PHONY: clean test ty vulture

all: test

test:
	uv run pytest -s -v -x

ty:
	uv run ty check

coverage.xml:
	uv run pytest --junitxml=junit.xml --cov=tensorpack --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml

vulture:
	uv run vulture
