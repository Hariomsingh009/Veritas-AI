from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .pipeline import verify

app = FastAPI(title="Think10x Hallucination Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    question: str
    answer: str | None = None
    mode: str = "baseline"
    use_judge: bool = True

@app.post("/verify")
def _verify(r: Req): return verify(r.question, r.answer, r.mode, r.use_judge)

@app.post("/compare")
def compare(r: Req):
    return {"baseline": verify(r.question, mode="baseline", use_judge=r.use_judge),
            "rag": verify(r.question, mode="rag", use_judge=r.use_judge)}

@app.get("/health")
def health(): return {"ok": True}
