#########
# BUILD #
#########
.PHONY: develop-py develop-rs develop
develop-py:
	uv pip install -e .[develop]

develop-rs:
	make -C rust develop

develop: develop-rs develop-py  ## setup project for development

.PHONY: requirements-py requirements-rs requirements
requirements-py:  ## install prerequisite python build requirements
	python -m pip install --upgrade pip toml
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["build-system"]["requires"]))'`
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print(" ".join(c["project"]["optional-dependencies"]["develop"]))'`

requirements-rs:  ## install prerequisite rust build requirements
	make -C rust requirements

requirements: requirements-rs requirements-py  ## setup project for development

.PHONY: build-py build-rs build
build-py:
	python -m build -w -n

build-rs:
	make -C rust build

build: build-rs build-py  ## build the project

.PHONY: install
install:  ## install python library
	uv pip install .

#########
# LINTS #
#########
.PHONY: lint-py lint-rs lint-docs lint lints
lint-py:  ## run python linter with ruff
	python -m ruff check superstore
	python -m ruff format --check superstore

lint-rs:  ## run rust linter
	make -C rust lint

lint-docs:  ## lint docs with mdformat and codespell
	python -m mdformat --check README.md
	python -m codespell_lib README.md

lint: lint-rs lint-py lint-docs  ## run project linters

# alias
lints: lint

.PHONY: fix-py fix-rs fix-docs fix format
fix-py:  ## fix python formatting with ruff
	python -m ruff check --fix superstore
	python -m ruff format superstore

fix-rs:  ## fix rust formatting
	make -C rust fix

fix-docs:  ## autoformat docs with mdformat and codespell
	python -m mdformat README.md
	python -m codespell_lib --write README.md

fix: fix-rs fix-py fix-docs  ## run project autoformatters

# alias
format: fix

################
# Other Checks #
################
.PHONY: check-manifest stubtest checks check

check-manifest:  ## check python sdist manifest with check-manifest
	check-manifest -v

stubtest:  ## validate type stubs against runtime module
	python -m mypy.stubtest superstore.superstore --allowlist superstore/stubtest_allowlist.txt --ignore-unused-allowlist

checks: check-manifest stubtest

# alias
check: checks

#########
# TESTS #
#########
.PHONY: test-py tests-py coverage-py
test-py:  ## run python tests
	python -m pytest -v superstore/tests

# alias
tests-py: test-py

coverage-py:  ## run python tests and collect test coverage
	python -m pytest -v superstore/tests --cov=superstore --cov-report term-missing --cov-report xml

.PHONY: test-rs tests-rs coverage-rs
test-rs:  ## run rust tests
	make -C rust test

# alias
tests-rs: test-rs

coverage-rs:  ## run rust tests and collect test coverage
	make -C rust coverage

.PHONY: test coverage tests
test: test-py test-rs  ## run all tests
coverage: coverage-py coverage-rs  ## run all tests and collect test coverage

# alias
tests: test

###########
# VERSION #
###########
.PHONY: show-version patch minor major

show-version:  ## show current library version
	@bump-my-version show current_version

patch:  ## bump a patch version
	@bump-my-version bump patch

minor:  ## bump a minor version
	@bump-my-version bump minor

major:  ## bump a major version
	@bump-my-version bump major

########
# DIST #
########
.PHONY: dist-py-wheel dist-py-sdist dist-rs dist-check dist publish

dist-py-wheel:  ## build python wheel
	python -m cibuildwheel --output-dir dist

dist-py-sdist:  ## build python sdist
	python -m build --sdist -o dist

dist-rs:  ## build rust dists
	make -C rust dist

dist-check:  ## run python dist checker with twine
	python -m twine check dist/*

dist: clean build dist-rs dist-py-wheel dist-py-sdist dist-check  ## build all dists

publish: dist  ## publish python assets

##############
# BENCHMARKS #
##############
.PHONY: bench bench-run bench-quick bench-compare bench-publish bench-preview bench-machine

bench-machine:  ## configure asv machine info (required for CI)
	python -m asv machine --yes

bench:  ## run all benchmarks (quick mode, using current install)
	python -m asv machine --yes
	python -m asv run --quick --environment existing

bench-run:  ## run all benchmarks (full, from commits)
	python -m asv machine --yes
	python -m asv run

bench-quick:  ## run benchmarks with minimal samples (current install)
	python -m asv machine --yes
	python -m asv run --quick --show-stderr --environment existing

bench-compare:  ## compare benchmarks between HEAD and main
	python -m asv machine --yes
	python -m asv continuous main HEAD --quick

bench-publish:  ## generate benchmark HTML report
	python -m asv publish

bench-preview:  ## preview benchmark results in browser
	python -m asv preview

#########
# CLEAN #
#########
.PHONY: deep-clean clean

deep-clean: ## clean everything from the repository
	git clean -fdx

clean: ## clean the repository
	rm -rf .coverage coverage cover htmlcov logs build dist *.egg-info .asv

############################################################################################

.PHONY: help

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
