.PHONY: clean test

all: test

test:
	uv run pytest -s -v -x

coverage.xml:
	uv run pytest --junitxml=junit.xml --cov=tensorpack --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml
