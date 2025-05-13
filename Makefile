#!make
SHELL := /usr/bin/env bash

build:
	docker compose build 

## Environment setup
up: ## Bring up your Docker environment
	docker compose up -d postgres
	docker compose run checkdb
	docker compose up -d jupyter

install-python-pkgs: ## Editable install of package for development
	docker compose exec jupyter bash -c "cd /home/jovyan && pip install -e ."

lint-src:
	ruff check --select I --fix src/
	ruff format --verbose src/

start-local-jupyter:
	cp jupyter_lab_config.py ${CONDA_PREFIX}/etc/jupyter/jupyter_lab_config.py
	nohup jupyter lab > /dev/null 2>&1 &
	sleep 10s
	jupyter lab list

sync-local-env:
	micromamba env update -n deafrica-water-quality-env -f environment.yaml 

