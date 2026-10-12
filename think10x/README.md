# Think10x: Uncertainty-Aware Hallucination Detection Engine

Pipeline: question -> LLM answer -> atomic claims -> FAISS retrieval -> NLI (+ optional LLM judge) -> uncertainty -> claim and answer risk.

## Run
```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # optional: add ANTHROPIC_API_KEY for real answers/claims/judge
uvicorn app.main:app --reload  # http://localhost:8000/docs

cd frontend && npm install && npm run dev   # new terminal
```
Without an API key, paste an answer in the UI; claims use a heuristic splitter and the judge is skipped.

## API
`POST /verify {question, answer?, mode: baseline|rag}`, `POST /compare {question}`, `GET /health`

## Your own knowledge
Drop `.txt` files into `backend/data/kb/` and restart.

## Evaluate
`cd backend && python -m eval.evaluate` prints accuracy, P/R/F1, ROC-AUC, ECE. The bundled 6-row dataset is a smoke test only.
Build 100+ labeled rows and report only measured numbers.

## Design notes
- Similarity never decides truth: NLI and the judge do; similarity is only a mismatch penalty.
- Weights/thresholds in `app/scoring.py` are starting values; tune on a held-out split.
- High uncertainty triggers wider retrieval, then flags `needs_human_review`.
