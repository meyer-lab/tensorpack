.PHONY: clean test

all: test

test:
	poetry run pytest -s -v -x

coverage.xml:
	poetry run pytest --junitxml=junit.xml --cov=tensorpac --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml