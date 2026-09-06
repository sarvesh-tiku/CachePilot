.PHONY: dev down install backend frontend test lint fmt bench

dev:
	docker compose up --build

down:
	docker compose down

install:
	pip install -e ".[dev]"
	cd frontend && npm ci

backend:
	PYTHONPATH=backend uvicorn cachepilot.main:app --reload

frontend:
	cd frontend && npm run dev

test:
	pytest

lint:
	ruff check backend benchmark
	ruff format --check backend benchmark
	mypy backend/cachepilot benchmark

fmt:
	ruff check --fix backend benchmark
	ruff format backend benchmark

bench:
	PYTHON=$${PYTHON:-python} ./scripts/run_benchmark.sh $(WORKLOAD) $(REQUESTS) $(SEED)
