# Tiered reproduction for the PVLDB Reproducibility Committee.
#
# The claim->command->artifact table in REPRODUCIBILITY.md is complete but not ordered by cost:
# some rows take seconds, one takes 2.5 hours, and two need a paid API key. A committee member
# should not have to read twenty rows to find out which is which. These targets are that ordering.
#
#   make verify          seconds, no data, no network   -- does the shipped artifact hold together?
#   make reproduce-fast  ~40 min, Mooncake traces only  -- the central claims, from scratch
#   make reproduce-full  ~6 h,    + the LRB sweep       -- everything CPU-only
#   make reproduce-llm   ~3 h,    + OPENAI_API_KEY, ~$5 -- the two claims that cannot be free
#
# `verify` is the one to run first and the one that fails loudest. It needs nothing but Python.

PY ?= python3
SHELL := /bin/bash

.PHONY: help verify reproduce-fast reproduce-full reproduce-llm figures paper clean data-check

help:
	@echo "make verify          — 231 tests + every paper number against its artifact (no data needed)"
	@echo "make data-check      — are the traces present and the right length?"
	@echo "make reproduce-fast  — regenerate the central claims from the Mooncake traces (~40 min)"
	@echo "make reproduce-full  — the above plus the LRB hyperparameter sweep (~6 h)"
	@echo "make reproduce-llm   — the two LoCoMo claims; needs OPENAI_API_KEY, costs about \$$5"
	@echo "make figures paper   — redraw figures, rebuild the PDF"
	@echo ""
	@echo "Start with 'make verify'. It exits non-zero on any drift and needs no downloads."

# --- tier 0: does the shipped artifact hold together? --------------------------------------
verify:
	$(PY) -m pytest -q
	$(PY) scripts/check_paper_numbers.py
	@echo
	@echo "To also fetch the artifact URL with no credentials (what a reviewer sees):"
	@echo "  STOA_CHECK_URL=1 $(PY) scripts/check_paper_numbers.py"

# --- data ---------------------------------------------------------------------------------
data-check:
	@for f in data/mooncake_conversation_trace.jsonl data/mooncake_toolagent_trace.jsonl; do \
	  if [ -f $$f ]; then printf "%8d lines  %s\n" "$$(wc -l < $$f)" "$$f"; \
	  else echo "MISSING $$f — see data/README.md"; fi; done
	@echo "Expect 12031 (conversation) and 23608 (toolagent). Several results run both traces to exhaustion, so a"
	@echo "truncated download changes the numbers silently."

# --- tier 1: the central claims, CPU only, no API key -------------------------------------
reproduce-fast: data-check
	$(PY) scripts/run_reachable_ceiling.py
	$(PY) scripts/run_arrival_admission.py
	$(PY) scripts/run_belief_controls.py
	$(PY) scripts/run_capacity_ladder.py
	$(PY) scripts/run_calibration_sensitivity.py
	$(PY) scripts/check_paper_numbers.py

# --- tier 2: everything that does not cost money -------------------------------------------
reproduce-full: reproduce-fast
	$(PY) scripts/run_reactive_real.py --lrb
	$(PY) scripts/run_lrb_sweep.py
	$(PY) scripts/run_split_sensitivity.py
	$(PY) scripts/run_locomo_power.py
	$(PY) scripts/check_paper_numbers.py

# --- tier 3: the two claims that need a paid third-party API --------------------------------
# Stated plainly rather than buried: these cannot be reproduced for free, and a committee
# member who discovers that halfway through a 2.5-hour run reports it as irreproducible.
reproduce-llm:
	@if [ -z "$$OPENAI_API_KEY" ]; then \
	  echo "OPENAI_API_KEY is unset."; \
	  echo "These two results call a hosted model at temperature=0 and cost about \$$5 total."; \
	  echo "They cannot be bit-reproduced even with a key: the served weights are not pinned."; \
	  echo "The shipped artifacts record every prompt and response id."; exit 1; fi
	$(PY) scripts/run_locomo_powered.py --judge
	$(PY) scripts/run_representation_axis.py --judge

figures:
	$(PY) scripts/make_figures.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode main && bibtex main \
	  && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main
	$(PY) scripts/check_paper_numbers.py

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out
