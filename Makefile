#!make
SHELL := /usr/bin/env bash

include  .env

ENV_FILE =  $(abspath .env)
export ENV_FILE

build:
	docker compose build 

## Environment setup
up: ## Bring up your Docker environment
	docker compose up -d db
	docker compose up -d jupyter

down:
	docker compose down --remove-orphans
	
init: ## Prepare the database, initialise the database schema.
	docker compose exec -T jupyter datacube -v system init

products:
	docker compose exec -T jupyter dc-sync-products products.csv --update-if-exists

index: ## Index the test data.
	cat index_test_tiles.sh | docker compose exec -T jupyter bash

install-pkg: ## Editable install of package for development
	docker compose exec jupyter bash -c "cd /home/jovyan && pip install -e ."

run-tests:
	pytest tests/

test-env: build up init products index install-waterbodies

lint-src:
	ruff check --select I --fix src/ tests/             
	ruff format --verbose src/ tests/
	
start-local-jupyter:
	cp jupyter_lab_config.py ${CONDA_PREFIX}/etc/jupyter/jupyter_lab_config.py
	nohup jupyter lab > /dev/null 2>&1 &
	sleep 10s
	jupyter lab list

sync-local-env:
	micromamba env update -n deafrica-water-quality-env -f environment.yaml 