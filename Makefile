.PHONY: help install dev demo lint test check build clean docker

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:             ## Install unalloc into the current environment
	pip install .

dev:                 ## Editable install with test and lint tooling
	pip install -e ".[dev]"

demo:                ## Run all three commands against bundled fixtures
	unalloc report --fixtures --dimension team
	unalloc labels --fixtures --dimension team --limit 5
	unalloc reconcile --fixtures --invoice openai=4031.36

lint:                ## Ruff
	ruff check .

test:                ## Pytest
	pytest

check: lint test     ## Everything CI runs

build:               ## Build wheel and sdist into dist/
	pip install -q build && python -m build

docker:              ## Build the container image
	docker build -t unalloc:0.1.0 .

clean:
	rm -rf dist build .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
