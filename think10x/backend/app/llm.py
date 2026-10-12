"""LLM wrapper (answer generation, claim extraction, judge). Falls back to heuristics without an API key."""
import os, re, json
from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
_client = None
if os.getenv("ANTHROPIC_API_KEY"):
    import anthropic
    _client = anthropic.Anthropic()

def _ask(prompt: str, max_tokens=800) -> str:
    r = _client.messages.create(model=MODEL, max_tokens=max_tokens,
                                messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

def generate_answer(question: str, context: str | None = None) -> str:
    """context=None -> baseline GPT-style; context given -> RAG answer."""
    if not _client:
        return "No API key set. Provide an answer to check, or set ANTHROPIC_API_KEY."
    p = question if context is None else (
        f"Answer using ONLY this context. If insufficient, say so.\nContext:\n{context}\n\nQuestion: {question}")
    return _ask(p + "\nAnswer concisely in 1-3 sentences.")

def extract_claims(answer: str) -> list[str]:
    if _client:
        try:
            out = _ask("Split the text into atomic, self-contained factual claims. "
                       "Return ONLY a JSON list of strings.\n\n" + answer)
            return json.loads(re.search(r"\[.*\]", out, re.S).group(0))
        except Exception:
            pass
    sents = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = []
    for s in sents:
        parts = re.split(r",? and (?=[A-Z]?[a-z]+ (?:is|was|are|were|has|have))", s)
        claims += [p.strip() for p in parts if len(p.split()) > 2]
    return claims

def judge(claim: str, evidence: str) -> dict:
    """LLM-as-judge: one verification signal, not ground truth."""
    if not _client:
        return {"label": None, "confidence": 0.0}
    try:
        out = _ask(f'Evidence:\n{evidence}\n\nClaim: {claim}\nReturn ONLY JSON: '
                   '{"label":"supported|contradicted|uncertain","confidence":0-1}', 100)
        return json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    except Exception:
        return {"label": None, "confidence": 0.0}
