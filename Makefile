#!make
SHELL := /usr/bin/env bash

lint-src:
	ruff check --select I --fix src/
	ruff format --verbose src/

sync-local-env:
	micromamba env update -n deafrica-water-quality-env -f environment.yaml 