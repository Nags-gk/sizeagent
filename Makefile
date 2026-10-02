.PHONY: setup test lint bench robust stats report
setup:
	./scripts/setup_pdk.sh && pip install -e .[dev]
test:
	pytest -q --cov=sizeagent
lint:
	ruff check .
bench:
	python scripts/run_benchmark.py --seeds 20 --budget 300
robust:
	python scripts/robust_study.py
stats:
	python scripts/stats.py
report:
	python scripts/make_report.py
