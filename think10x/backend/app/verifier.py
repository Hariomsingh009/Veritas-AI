"""NLI verification: evidence (premise) vs claim (hypothesis)."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIVerifier:
    def __init__(self, name="cross-encoder/nli-deberta-v3-small"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.labels = {i: l.lower() for i, l in self.m.config.id2label.items()}  # read from config

    @torch.no_grad()
    def probs(self, evidence: str, claim: str) -> dict:
        x = self.tok(evidence, claim, return_tensors="pt", truncation=True, max_length=384)
        p = torch.softmax(self.m(**x).logits[0], -1).tolist()
        return {self.labels[i]: p[i] for i in range(len(p))}
