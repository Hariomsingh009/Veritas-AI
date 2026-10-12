"""Combine signals into claim label, uncertainty and risk. Weights are starting points:
tune them on your labeled set with eval/evaluate.py."""
import math
W = dict(contradiction=0.55, unsupported=0.25, uncertainty=0.12, mismatch=0.08)
T_SUPPORT, T_CONTRA, T_UNCERTAIN = 0.60, 0.50, 0.55

def entropy(p: dict) -> float:
    v = [x for x in p.values() if x > 0]
    return -sum(x * math.log(x) for x in v) / math.log(max(len(p), 2))

def best_evidence_nli(claim, hits, nli):
    best, best_hit = None, None
    for h in hits:
        p = nli.probs(h["text"], claim)
        s = max(p.get("entailment", 0), p.get("contradiction", 0))
        if best is None or s > best[0]: best, best_hit = (s, p), h
    return (best[1], best_hit) if best else ({}, None)

def score_claim(nli_p: dict, sim: float, judge: dict):
    ent, con = nli_p.get("entailment", 0), nli_p.get("contradiction", 0)
    unc = entropy(nli_p) if nli_p else 1.0
    jl, jc = judge.get("label"), float(judge.get("confidence", 0) or 0)
    if jl == "contradicted": con = (con + jc) / 2
    elif jl == "supported": ent = (ent + jc) / 2
    elif jl == "uncertain": unc = max(unc, jc)
    unsupported = max(0.0, 1 - ent - con)
    risk = (W["contradiction"] * con + W["unsupported"] * unsupported
            + W["uncertainty"] * unc + W["mismatch"] * (1 - max(sim, 0)))
    if not nli_p or sim < 0.25: label = "uncertain"
    elif con >= T_CONTRA and con > ent: label = "hallucination"
    elif ent >= T_SUPPORT: label = "supported"
    else: label = "uncertain"
    return label, round(min(risk, 1.0), 3), round(unc, 3), {"entailment": round(ent, 3), "contradiction": round(con, 3)}

def needs_review(unc: float) -> bool:
    return unc > T_UNCERTAIN
