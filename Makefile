.PHONY: check manuscript notebooks shell test

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

shell:
	find experiments scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

notebooks:
	python3 scripts/validate_notebooks.py

manuscript:
	cd manuscript && latexmk -pdf main.tex

check: test shell notebooks
