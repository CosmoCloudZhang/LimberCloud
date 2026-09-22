.PHONY: check check-all lint manuscript notebooks shell test test-fast test-science

# The ignored .venv link is the checkout-local interpreter. Using bare python3
# on this host selects /usr/bin/python3 (3.6), which cannot import the
# selected environment. Override with `make PYTHON=...` only for a deliberate
# other prefix.
PYTHON ?= $(firstword $(wildcard .venv/bin/python3) python3)

# Login-safe: these suites import NumPy and SciPy only. They never import h5py,
# Numba, JAX or CCL, so they cannot initialise a site-linked HDF5 or MPI stack.
FAST_TESTS = \
	tests.test_angular_contract \
	tests.test_environment_contracts \
	tests.test_experiment_contracts \
	tests.test_launcher_smoke \
	tests.test_project_paths \
	tests.test_run_identity

# Allocated: these import h5py and the compiled backends. An MPI-linked h5py
# build makes them unsafe on a login node; run them under an allocation.
SCIENCE_TESTS = \
	tests.test_projection_consistency \
	tests.test_science_artifacts

test-fast:
	PYTHONPATH=src $(PYTHON) -m unittest -v $(FAST_TESTS)

test-science:
	PYTHONPATH=src $(PYTHON) -m unittest -v $(SCIENCE_TESTS)

test: test-fast test-science

lint:
	$(PYTHON) -m ruff check .

shell:
	find experiments scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

notebooks:
	$(PYTHON) scripts/validate_notebooks.py

manuscript:
	cd manuscript && latexmk -pdf main.tex

# Login-safe gate.
check: lint test-fast shell notebooks

# Full gate. Run it on a supported allocation.
check-all: lint test shell notebooks
