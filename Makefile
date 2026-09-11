# Developer-box entry points. Nothing here compiles a package: builds happen
# in CI only (docs/ARCHITECTURE.md, "Builds happen in CI").

PKG ?=
JOBS ?= 3
PKGFLAG := $(if $(PKG),--package $(PKG),)

.PHONY: clean-verify test regen-check

# The GPU half of the clean-environment test, against the LIVE channel and
# index: a fresh pixi env / venv per published cell holding only what the
# artifact declares, the package's verify op on this box's GPU. Run after
# every rebuild wave, before announcing it. `make clean-verify PKG=cumm`
# for one package.
clean-verify:
	python3 tools/clean_verify.py $(PKGFLAG) --jobs $(JOBS)

# Every negative-controlled test the regen-check workflow runs.
test:
	@rc=0; for t in scripts/test_*.py tools/test_*.py; do \
	  echo "== $$t"; python3 $$t || rc=1; done; exit $$rc

regen-check:
	python3 scripts/generate_recipes.py --check
	python3 scripts/lint_guarantee.py
	$(MAKE) test
