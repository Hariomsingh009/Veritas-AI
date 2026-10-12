"""Run from backend/: python -m eval.evaluate. Extend dataset.json to 100+ rows for real results."""
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from app.pipeline import verify

def ece(scores, y, bins=5):
    e, n = 0.0, len(y)
    for b in range(bins):
        idx = [i for i, s in enumerate(scores) if b / bins <= s < (b + 1) / bins or (b == bins - 1 and s == 1)]
        if idx: e += len(idx) / n * abs(sum(scores[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return e

data = json.load(open("eval/dataset.json"))
scores, y = [], []
for d in data:
    scores.append(verify(d["question"], d["answer"])["hallucination_score"]); y.append(d["label"])
for thr in (0.3, 0.4, 0.5):
    pred = [int(s >= thr) for s in scores]
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    print(f"thr={thr} acc={accuracy_score(y, pred):.3f} P={p:.3f} R={r:.3f} F1={f:.3f}")
if len(set(y)) > 1: print("ROC-AUC", round(roc_auc_score(y, scores), 3))
print("ECE", round(ece(scores, y), 3))
