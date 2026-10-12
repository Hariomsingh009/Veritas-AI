from .llm import generate_answer, extract_claims, judge
from .rag import KnowledgeBase
from .verifier import NLIVerifier
from .scoring import best_evidence_nli, score_claim, needs_review

kb, nli = KnowledgeBase(), NLIVerifier()

def verify(question: str, answer: str | None = None, mode="baseline", use_judge=True):
    """mode 'baseline' = plain LLM; 'rag' = LLM answers from retrieved context.
    Both are verified against the same trusted KB so the comparison is fair."""
    if answer is None:
        ctx = "\n".join(h["text"] for h in kb.search(question, 3)) if mode == "rag" else None
        answer = generate_answer(question, ctx)
    out = []
    for c in extract_claims(answer):
        hits = kb.search(c, 3)
        p, top = best_evidence_nli(c, hits, nli)
        sim = top["similarity"] if top else 0.0
        j = judge(c, top["text"]) if (use_judge and top) else {}
        label, risk, unc, sig = score_claim(p, sim, j)
        if top and needs_review(unc):  # adaptive re-verification: widen retrieval
            p2, top2 = best_evidence_nli(c, kb.search(c, 6), nli)
            if top2:
                top, sim = top2, top2["similarity"]
                label, risk, unc, sig = score_claim(p2, sim, j)
        out.append(dict(claim=c, label=label, risk=risk, uncertainty=unc, signals=sig,
                        judge=j.get("label"), similarity=round(sim, 3),
                        evidence=top, needs_human_review=needs_review(unc)))
    h = sum(x["risk"] for x in out) / len(out) if out else 0.0
    return dict(question=question, answer=answer, mode=mode, claims=out,
                hallucination_score=round(h, 3), reliability_score=round(1 - h, 3))
