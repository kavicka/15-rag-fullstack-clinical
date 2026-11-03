.PHONY: sync data index eval-retrieval eval-generation plot api web test lint

sync:
	uv sync
	cd frontend && npm ci

data:             ## download the frozen list of trials and build the eval set
	uv run python -m rag.ingest --from-ids
	uv run python eval/build_eval.py

index:            ## chunk + embed both chunking strategies
	uv run python -m rag.index

eval-retrieval:
	uv run python eval/run_eval.py --retrieval

eval-generation:  ## needs Ollama running with $(LLM_MODEL) pulled
	uv run python eval/run_eval.py --generation

plot:
	uv run python plot.py

api:
	uv run uvicorn api.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
