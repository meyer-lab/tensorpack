.PHONY: clean test

all: test

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest -s -v -x

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

clean:
	rm -rf venv coverage.xml