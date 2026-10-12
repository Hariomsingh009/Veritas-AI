import { useState } from "react";
const API = import.meta.env.VITE_API || "http://localhost:8000";
const ICON = { supported: "✓ Supported", hallucination: "✗ Contradicted", uncertain: "? Uncertain" };

function Result({ title, r }) {
  return (
    <section>
      <h2>{title}</h2>
      <p className="answer">{r.answer}</p>
      <div className="scores">
        <div><b>{(r.reliability_score * 100).toFixed(0)}%</b> reliability</div>
        <div><b>{(r.hallucination_score * 100).toFixed(0)}%</b> hallucination risk</div>
      </div>
      {r.claims.map((c, i) => (
        <article key={i} className={"claim " + c.label}>
          <header><span>{ICON[c.label]}</span><span>risk {c.risk}, uncertainty {c.uncertainty}</span></header>
          <p>{c.claim}</p>
          {c.evidence && <blockquote>{c.evidence.text}<cite>{c.evidence.source}, similarity {c.similarity}</cite></blockquote>}
          {c.needs_human_review && <em>Low confidence: needs human review.</em>}
        </article>
      ))}
    </section>
  );
}

export default function App() {
  const [q, setQ] = useState(""), [a, setA] = useState("");
  const [data, setData] = useState(null), [busy, setBusy] = useState(false), [err, setErr] = useState("");
  async function run(path) {
    setBusy(true); setErr(""); setData(null);
    try {
      const res = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, answer: a || null }) });
      if (!res.ok) throw new Error("Server returned " + res.status);
      setData(await res.json());
    } catch (e) { setErr("Could not reach the backend at " + API + ". Start it with uvicorn. (" + e.message + ")"); }
    setBusy(false);
  }
  return (
    <main>
      <h1>Is this answer true?</h1>
      <p className="sub">Ask a question. Each claim in the answer is checked against your knowledge base.</p>
      <input value={q} onChange={e => setQ(e.target.value)} placeholder="When was the Eiffel Tower completed?" />
      <textarea value={a} onChange={e => setA(e.target.value)} rows={2}
        placeholder="Optional: paste an answer to check instead of generating one" />
      <div className="row">
        <button disabled={!q || busy} onClick={() => run("/verify")}>Verify answer</button>
        <button className="ghost" disabled={!q || busy || !!a} onClick={() => run("/compare")}>Compare plain LLM vs RAG</button>
      </div>
      {busy && <p>Checking claims…</p>}{err && <p className="err">{err}</p>}
      {data && (data.baseline
        ? <div className="cols"><Result title="Plain LLM" r={data.baseline} /><Result title="LLM + RAG" r={data.rag} /></div>
        : <Result title="Verification" r={data} />)}
    </main>
  );
}
