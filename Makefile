.PHONY: install test audit analysis verify figures report app data all clean

VENV_PYTHON = $(wildcard .venv/bin/python)
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),$(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python))

install:
	$(PYTHON) -m pip install -r requirements.lock.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) scripts/audit_sources.py

analysis:
	$(PYTHON) scripts/run_analysis.py

verify:
	$(PYTHON) scripts/verify_results.py

figures:
	$(PYTHON) scripts/build_figures.py

report: figures
	$(PYTHON) scripts/build_report.py

app:
	$(PYTHON) -m streamlit run app.py

data:
	$(PYTHON) scripts/acquire_data.py
	$(PYTHON) scripts/acquire_macro.py

all: audit analysis verify report test

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
