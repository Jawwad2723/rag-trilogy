.PHONY: install up down test lint security clean ingest eval bench

install:
	poetry install

up:
	docker compose up -d

up-local:
	docker compose -f docker-compose.local.yml up -d

down:
	docker compose down

test:
	poetry run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

lint:
	poetry run mypy --strict src/ --ignore-missing-imports

security:
	poetry run bandit -r src/ -ll

ingest:
	poetry run python scripts/ingest_docs.py --dir data/

eval:
	poetry run python scripts/run_eval.py --output eval_results.json

bench:
	poetry run python scripts/benchmark_models.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
