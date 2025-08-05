# Build mass
# J. Fowler, NIST
# Updated May 2023

PYSCRIPTS = bin/ljh_merge bin/ljh_truncate
PYFILES = $(shell find . -name "*.py") $(PYSCRIPTS)
FORMFILES := $(shell find mass -name "*_form_ui.py")

.PHONY: all build clean test pep8 autopep8 lint ruff

all: test

clean:
	rm -rf build || sudo rm -rf build

test:
	pytest

PEPFILES := $(PYFILES)
PEPFILES := $(filter-out $(FORMFILES), $(PEPFILES))  # Remove the UI.py forms

pep8: pep8-report.txt
pep8-report.txt: $(PEPFILES) Makefile
	pycodestyle --exclude=build,nonstandard . > $@ || true

autopep8: $(PEPFILES) Makefile
	autopep8 --verbose --in-place --recursive .

lint: lint-report.txt
lint-report.txt: $(PYFILES) Makefile
	ruff check > $@

ruff:
	ruff check
