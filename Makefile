.PHONY: help install dev research require-research demo lint test check check-wheel build clean \
	docker case-studies case-studies-quick figures paper notebook ui devcontainer \
	devcontainer-check release-check pages

VERSION := $(shell python3 -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

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

check-wheel: build   ## Install the built wheel in a throwaway venv and run the demo
	@rm -rf /tmp/unalloc-verify && python3 -m venv /tmp/unalloc-verify
	@/tmp/unalloc-verify/bin/pip install -q dist/unalloc-*.whl
	@/tmp/unalloc-verify/bin/unalloc report --fixtures -D team
	@rm -rf /tmp/unalloc-verify

build:               ## Build wheel and sdist into dist/
	pip install -q build && python -m build

docker:              ## Build the container image
	docker build -t unalloc:$(VERSION) .

release-check: build ## Validate sdist/wheel metadata as PyPI will render it
	pip install -q twine && twine check --strict dist/*

pages: require-research ## Rebuild the GitHub Pages site in docs/ (landing, blog, paper, explorer)
	$(PY) -m docs_build

# --- case studies & paper ------------------------------------------------------

# The case studies, figures, paper and site need the research stack (torch,
# matplotlib, typst, markdown). Use the current interpreter when it has them,
# otherwise the project venv that `make research` builds. Override with
# `make paper PY=path/to/python`.
VENV := .venv-research
PY ?= $(shell if python3 -c 'import matplotlib, typst, markdown' 2>/dev/null; then echo python3; \
	elif [ -x $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo MISSING; fi)

research:            ## Build ./.venv-research with the case-study stack (CPU torch, notebook, typst)
	@command -v uv >/dev/null && uv venv $(VENV) || python3 -m venv $(VENV)
	@# unsafe-best-match: the CPU torch index carries a few unrelated packages at old
	@# versions, and uv's default (first index wins) then fails to resolve them.
	@command -v uv >/dev/null \
		&& uv pip install --python $(VENV)/bin/python --index-strategy unsafe-best-match \
			--extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,research]" \
		|| $(VENV)/bin/pip install \
			--extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,research]"
	@echo "research stack ready in $(VENV)"

require-research:
	@if [ "$(PY)" = "MISSING" ]; then \
		echo "This target needs the case-study stack (torch, matplotlib, typst, markdown)."; \
		echo "Run 'make research' once to build ./$(VENV), or pass your own:"; \
		echo "    make $(MAKECMDGOALS) PY=/path/to/python"; \
		exit 1; fi

STUDIES = kv_cache torch_kv distributed hybrid_e2e use_cases

case-studies: require-research ## Run every case study at full size (several minutes)
	@for s in $(STUDIES); do echo "== $$s"; $(PY) -m case_studies.$$s || exit 1; done

case-studies-quick: require-research ## Smoke-run every case study into a scratch directory
	@for s in $(STUDIES); do echo "== $$s"; $(PY) -m case_studies.$$s --quick --out .cache/quick/$$s || exit 1; done

figures: require-research ## Render paper figures (print) and site figures (dark)
	$(PY) -m paper.make_figures

paper: figures       ## Build paper/unalloc-case-studies.pdf
	$(PY) -m paper.build

notebook: require-research ## Execute the companion notebook in place
	$(PY) -m case_studies.notebook_exec

ui: require-research ## Serve the attribution explorer on http://localhost:8765
	$(PY) -m case_studies.ui

devcontainer:        ## Start the isolated dev stack (workspace + Postgres)
	docker compose -f .devcontainer/docker-compose.yml up -d --build

devcontainer-check: devcontainer ## Lint, test and smoke the case studies inside the container
	docker compose -f .devcontainer/docker-compose.yml exec -T dev sh -c "make check && make case-studies-quick"

clean:
	rm -rf dist build .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info .cache
