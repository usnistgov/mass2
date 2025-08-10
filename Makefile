# Build mass
# J. Fowler, NIST
# Updated Aug 2025. It's getting to be pretty useless to have this!

PYSCRIPTS = bin/ljh_merge bin/ljh_truncate
PYFILES = $(shell find . -name "*.py") $(PYSCRIPTS)

.PHONY: all build clean test ruff

all: test

clean:
	rm -rf build || sudo rm -rf build

test:
	pytest

ruff:
	ruff check
