VENV = .venv
PYTHON ?= $(VENV)/bin/python

install_dev:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -r requirements.txt

compile:
	$(PYTHON) -m piptools compile --resolver=backtracking --no-emit-index-url -o requirements.txt pyproject.toml

sync:
	$(PYTHON) -m piptools sync requirements.txt

up: compile sync