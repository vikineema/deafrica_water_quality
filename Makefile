#!make
SHELL := /usr/bin/env bash

include  .env

ENV_FILE =  $(abspath .env)
export ENV_FILE

setup: build up init

build:
	docker compose -f compose_dev.yaml build

up: ## Bring up your Docker environment
	docker compose -f compose_dev.yaml up -d db
	docker compose -f compose_dev.yaml up -d jupyter

down:
	docker compose -f compose_dev.yaml down --remove-orphans

init: ## Prepare the database, initialise the database schema.
	docker compose -f compose_dev.yaml exec -T jupyter datacube -v system init

add-products: ## Add products to the datacube database:
	# Add products to be added to the csv file products/products.csv
	docker compose -f compose_dev.yaml exec -T jupyter dc-sync-products products/products.csv --update-if-exists

index-datasets: ## Index datasets from a given path
	docker compose exec -T jupyter \
		s3-to-dc s3://deafrica-water-quality-dev/mapping/wq_annual/1-0-0/x200/y034/*/*.stac-item.json \
		--no-sign-request --allow-unsafe --stac wq_annual

install-pkg: ## Editable install of package for development
	docker compose -f compose_dev.yaml exec jupyter bash -c "cd /home/jovyan && pip install -e ."

lint-src:
	ruff check --select I --fix src/ tests/             
	ruff format --verbose src/ tests/
	
start-local-jupyter:
	cp jupyter_lab_config.py ${CONDA_PREFIX}/etc/jupyter/jupyter_lab_config.py
	nohup jupyter lab > /dev/null 2>&1 &
	sleep 10s
	jupyter lab list

jupyter-shell: ## Open shell in jupyter service
	docker compose -f compose_dev.yaml exec jupyter /bin/bash

## Explorer
setup-explorer: ## Setup the datacube explorer
	# Initialise and create product summaries
	docker compose -f compose_dev.yaml up -d explorer
	docker compose -f compose_dev.yaml exec -T explorer cubedash-gen --init --all
	# Services available on http://localhost:8080/products

explorer-refresh-products:
	docker compose -f compose_dev.yaml exec -T explorer cubedash-gen --init --all

explorer-shell: ## Open shell in explorer service
	docker compose -f compose_dev.yaml exec explorer /bin/bash
