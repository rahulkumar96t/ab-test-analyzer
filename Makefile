.PHONY: install lint test data demo validate peeking reports clean

install:
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

# Synthetic experiment: +8 % conversion, +3 % sessions, page load 0.5 % slower (all relative).
data:
	uv run abtest simulate --users 120000 --seed 7 --conversion-lift 0.08 \
		--sessions-lift 0.03 --page-load-lift 0.005 --out data/experiment.csv

demo: data
	uv run abtest analyze data/experiment.csv --spec examples/checkout.toml \
		--report reports/example_report.md

validate:
	uv run abtest validate --sims 2000 --seed 2026 --out reports/validation.json

peeking:
	uv run abtest peeking --sims 10000 --seed 2026 --out reports/peeking.json \
		--plot reports/peeking.png

reports: demo validate peeking

clean:
	rm -rf data .pytest_cache .ruff_cache
