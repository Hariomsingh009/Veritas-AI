"""Document ingestion -> chunking -> embeddings -> FAISS vector search."""
import glob, os, numpy as np, faiss
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self, kb_dir="data/kb", model="sentence-transformers/all-MiniLM-L6-v2",
                 chunk_words=60, overlap=15):
        self.emb = SentenceTransformer(model)
        self.chunks, self.sources = [], []
        for f in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
            w = open(f, encoding="utf-8").read().split()
            for i in range(0, max(len(w), 1), chunk_words - overlap):
                c = " ".join(w[i:i + chunk_words])
                if c: self.chunks.append(c); self.sources.append(os.path.basename(f))
        self.index = None
        if self.chunks:
            v = self.emb.encode(self.chunks, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(v.shape[1]); self.index.add(np.asarray(v, "float32"))

    def search(self, query: str, k=3):
        if not self.index: return []
        q = self.emb.encode([query], normalize_embeddings=True)
        s, i = self.index.search(np.asarray(q, "float32"), min(k, len(self.chunks)))
        return [{"text": self.chunks[j], "source": self.sources[j], "similarity": float(sc)}
                for sc, j in zip(s[0], i[0])]
