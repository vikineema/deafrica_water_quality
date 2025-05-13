#!make
SHELL := /usr/bin/env bash

sync-local-env:
	micromamba env update -n deafrica-water-quality-env -f environment.yaml 