import { useEffect, useMemo, useRef, useState } from 'react'

const EXAMPLES = {
  chinook: [
    'How many customers are from Germany?',
    'Which genre earned the most revenue?',
    'Top 5 artists by number of tracks',
    'Which employee supports the most customers?',
  ],
  tpch_sf001: [
    'How many orders were placed in 1995?',
    'Which nation has the most suppliers?',
    'Total revenue per region',
  ],
}

// ---- tiny inline SVG icon set (Lucide paths)
const Icon = ({ d, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d.split('|').map((p, i) => <path key={i} d={p} />)}
  </svg>
)
const I = {
  send: 'M22 2 11 13|M22 2 15 22 11 13 2 9 22 2',
  sun: 'M12 3v1|M12 20v1|M3 12h1|M20 12h1|M5.6 5.6l.7.7|M17.7 17.7l.7.7|M5.6 18.4l.7-.7|M17.7 6.3l.7-.7|M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
  moon: 'M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z',
  trash: 'M3 6h18|M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6|M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2',
  db: 'M12 8c4.97 0 9-1.34 9-3s-4.03-3-9-3-9 1.34-9 3 4.03 3 9 3Z|M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5|M3 12c0 1.66 4 3 9 3s9-1.34 9-3',
  copy: 'M8 8h12v12H8z|M4 16V4h12',
  check: 'M20 6 9 17l-5-5',
  alert: 'M12 9v4|M12 17h.01|M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z',
  retry: 'M3 12a9 9 0 1 0 3-6.7L3 8|M3 3v5h5',
}

// ---- SQL highlighting without a dependency
const KW = new Set(('select from where group by order having limit offset join inner left right outer full cross on as and or not in is null like between ' +
  'case when then else end distinct union all intersect except with exists asc desc count sum avg min max cast coalesce round strftime lower upper over partition').split(' '))
function SqlCode({ sql }) {
  const parts = sql.split(/('(?:[^']|'')*'|"[^"]*"|--[^\n]*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]+\b)/g)
  return (
    <pre className="code"><code>{parts.map((p, i) => {
      if (!p) return null
      if (p.startsWith("'")) return <span key={i} className="t-str">{p}</span>
      if (p.startsWith('--')) return <span key={i} className="t-com">{p}</span>
      if (/^\d/.test(p)) return <span key={i} className="t-num">{p}</span>
      if (KW.has(p.toLowerCase())) return <span key={i} className="t-kw">{p}</span>
      return p
    })}</code></pre>
  )
}

function CopyButton({ text }) {
  const [done, setDone] = useState(false)
  return (
    <button className="icon-btn small" aria-label="Copy SQL" title="Copy"
      onClick={() => navigator.clipboard.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1200) })}>
      <Icon d={done ? I.check : I.copy} size={16} />
    </button>
  )
}

function ResultTable({ columns, rows, total }) {
  const [sort, setSort] = useState({ col: -1, dir: 1 })
  const sorted = useMemo(() => {
    if (sort.col < 0) return rows
    const c = sort.col
    return [...rows].sort((a, b) => {
      const x = a[c], y = b[c]
      if (x === y) return 0
      if (x === null) return 1
      if (y === null) return -1
      return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * sort.dir
    })
  }, [rows, sort])
  if (!columns.length) return <p className="muted">No rows returned.</p>
  return (
    <>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Query result">
        <table>
          <thead><tr>{columns.map((c, i) => (
            <th key={i} aria-sort={sort.col === i ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none'}>
              <button onClick={() => setSort(s => ({ col: i, dir: s.col === i ? -s.dir : 1 }))}>
                {c}<span className="sort-ind" aria-hidden="true">{sort.col === i ? (sort.dir > 0 ? '▲' : '▼') : '↕'}</span>
              </button>
            </th>))}</tr></thead>
          <tbody>{sorted.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === 'number' ? 'num' : ''}>{v === null ? <span className="muted">NULL</span> : String(v)}</td>)}</tr>
          ))}</tbody>
        </table>
      </div>
      <p className="muted small">{total} row{total === 1 ? '' : 's'}{total > rows.length ? ` (showing first ${rows.length})` : ''} · click a header to sort</p>
    </>
  )
}

const Panel = ({ title, badge, open, children }) => (
  <details className="panel" open={open}>
    <summary><span>{title}</span>{badge != null && <span className="pill">{badge}</span>}</summary>
    <div className="panel-body">{children}</div>
  </details>
)

function fmtInput(v) {
  if (v == null) return ''
  if (typeof v === 'string') return v
  return Object.entries(v).map(([k, x]) => `${k}=${typeof x === 'string' ? x : JSON.stringify(x)}`).join(', ')
}

function Answer({ msg, labels, onRetry }) {
  const { r, cfg } = msg
  const m = r.metrics
  const tag = `${labels.path[cfg.path] || cfg.path} · ${labels.model[cfg.model] || cfg.model} · ${cfg.db}`
  return (
    <article className="bubble bot" aria-label="Answer">
      <div className="cfg-tag">{tag}</div>
      <p className="answer-text">{r.answer}</p>
      <div className="badges">
        {r.grounding && (r.grounding.grounded
          ? <span className="badge ok"><Icon d={I.check} size={14} /> grounded</span>
          : <span className="badge warn"><Icon d={I.alert} size={14} /> ungrounded: {r.grounding.flagged.join(', ')}</span>)}
        {r.error && <span className="badge err"><Icon d={I.alert} size={14} /> error</span>}
        {m.retries > 0 && <span className="badge">{m.retries} repair{m.retries > 1 ? 's' : ''}</span>}
        <span className="badge">{m.total_s.toFixed(1)} s</span>
        <span className="badge">{m.prompt_tokens} in / {m.completion_tokens} out</span>
      </div>
      {r.sql && (
        <Panel title="SQL" open>
          <div className="code-head"><span className="muted small">{m.retries ? 'final (after repair)' : 'executed'}</span><CopyButton text={r.sql} /></div>
          <SqlCode sql={r.sql} />
          {r.generated_sql && r.generated_sql !== r.sql && (<>
            <p className="muted small">Originally generated (failed, repaired {m.retries}×):</p>
            <SqlCode sql={r.generated_sql} />
          </>)}
        </Panel>
      )}
      {r.sql && !r.error && <Panel title="Result" badge={r.row_count}><ResultTable columns={r.columns} rows={r.rows} total={r.row_count} /></Panel>}
      <Panel title="Schema used" badge={r.schema.tables.length}>
        {r.schema.tables.length ? (<>
          <div className="chips">{r.schema.tables.map(t => <span key={t} className="chip">{t}</span>)}</div>
          <SqlCode sql={r.schema.ddl} />
        </>) : <p className="muted">No tables recorded for this run.</p>}
      </Panel>
      {(r.steps.length > 0 || r.trace.length > 0) && (
        <Panel title="Agent steps" badge={r.steps.length || r.trace.length}>
          {r.steps.length ? (
            <ol className="steps">{r.steps.map((s, i) => (
              <li key={i}>
                <details>
                  <summary><code>{s.tool}</code><span className="muted"> ({fmtInput(s.input)})</span></summary>
                  <pre className="code small-code">{s.output ?? '(no output)'}</pre>
                </details>
              </li>))}</ol>
          ) : <p className="muted">Trace: {r.trace.join(' → ')}</p>}
        </Panel>
      )}
      <Panel title="Metrics">
        <dl className="metrics">
          <div><dt>Total</dt><dd>{m.total_s.toFixed(2)} s</dd></div>
          <div><dt>Generation</dt><dd>{(m.generation_ms / 1000).toFixed(2)} s</dd></div>
          {m.setup_ms > 0 && <div><dt>Path setup</dt><dd>{(m.setup_ms / 1000).toFixed(2)} s</dd></div>}
          <div><dt>Tool calls</dt><dd>{m.tool_calls}</dd></div>
          <div><dt>Repairs</dt><dd>{m.retries}</dd></div>
          <div><dt>Prompt tokens</dt><dd>{m.prompt_tokens}</dd></div>
          <div><dt>Completion tokens</dt><dd>{m.completion_tokens}</dd></div>
          <div><dt>Throughput</dt><dd>{m.tok_s} tok/s</dd></div>
        </dl>
      </Panel>
      {r.error && <button className="btn ghost" onClick={onRetry}><Icon d={I.retry} size={16} /> Try again</button>}
    </article>
  )
}

function Pending({ cfg, labels }) {
  const [t, setT] = useState(0)
  useEffect(() => { const s = Date.now(); const id = setInterval(() => setT((Date.now() - s) / 1000), 200); return () => clearInterval(id) }, [])
  return (
    <div className="bubble bot pending" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      Running {labels.path[cfg.path]} on {labels.model[cfg.model]}… <span className="muted">{t.toFixed(0)} s</span>
    </div>
  )
}

export default function App() {
  const [opts, setOpts] = useState(null)
  const [optsErr, setOptsErr] = useState(null)
  const [cfg, setCfg] = useState({ path: 'B_stuff', model: 'tui:gemma4:31b', db: 'chinook' })
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'))
  const inputRef = useRef(null)
  const endRef = useRef(null)

  useEffect(() => {
    fetch('/api/options').then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(setOpts).catch(e => setOptsErr(String(e)))
  }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('theme', theme) } catch { /* private mode */ }
  }, [theme])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [msgs, busy])

  const labels = useMemo(() => ({
    path: Object.fromEntries((opts?.paths || []).map(p => [p.id, p.label])),
    model: Object.fromEntries((opts?.models || []).map(p => [p.id, p.label])),
  }), [opts])

  async function ask(question, c = cfg) {
    question = question.trim()
    if (!question || busy) return
    setBusy(true)
    setInput('')
    setMsgs(m => [...m, { role: 'user', text: question, cfg: c }])
    try {
      const res = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, ...c }) })
      const body = await res.json().catch(() => ({ detail: res.statusText }))
      if (!res.ok) throw new Error(body.detail || res.statusText)
      setMsgs(m => [...m, { role: 'bot', r: body, cfg: c, question }])
    } catch (e) {
      setMsgs(m => [...m, { role: 'error', text: e.message, cfg: c, question }])
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  // keyboard shortcuts: "/" focus, Alt+1..5 pick configuration, Alt+T theme
  useEffect(() => {
    const onKey = e => {
      const typing = ['TEXTAREA', 'INPUT', 'SELECT'].includes(document.activeElement?.tagName)
      if (e.key === '/' && !typing) { e.preventDefault(); inputRef.current?.focus() }
      else if (e.altKey && /^[1-5]$/.test(e.key) && opts) { e.preventDefault(); setCfg(c => ({ ...c, path: opts.paths[+e.key - 1].id })) }
      else if (e.altKey && e.key.toLowerCase() === 't') { e.preventDefault(); setTheme(t => t === 'dark' ? 'light' : 'dark') }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [opts])

  const examples = EXAMPLES[cfg.db] || []

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><Icon d={I.db} size={22} /><h1>Structured RAG</h1></div>
        <div className="controls">
          <label className="field"><span>Configuration</span>
            <select value={cfg.path} onChange={e => setCfg({ ...cfg, path: e.target.value })} disabled={!opts}>
              {opts?.paths.map((p, i) => <option key={p.id} value={p.id}>{p.label} (Alt+{i + 1})</option>)}
            </select>
          </label>
          <label className="field"><span>Model</span>
            <select value={cfg.model} onChange={e => setCfg({ ...cfg, model: e.target.value })} disabled={!opts}>
              {opts?.models.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
          <label className="field"><span>Database</span>
            <select value={cfg.db} onChange={e => setCfg({ ...cfg, db: e.target.value })} disabled={!opts}>
              {opts?.databases.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
        </div>
        <div className="actions">
          <button className="icon-btn" onClick={() => setMsgs([])} disabled={!msgs.length || busy} aria-label="Clear chat" title="Clear chat">
            <Icon d={I.trash} />
          </button>
          <button className="icon-btn" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title="Toggle theme (Alt+T)">
            <Icon d={theme === 'dark' ? I.sun : I.moon} />
          </button>
        </div>
      </header>

      <main className="thread" aria-live="polite">
        {optsErr && <div className="bubble error" role="alert"><Icon d={I.alert} /> Cannot reach the API: {optsErr}. Is <code>python -m webui.server</code> running?</div>}
        {!msgs.length && !optsErr && (
          <section className="empty">
            <h2>Ask a question about the database</h2>
            <p className="muted">The answer shows its evidence: the schema the configuration retrieved, the agent's tool calls, the SQL, and the rows it ran on.</p>
            {examples.length > 0 && <div className="examples">{examples.map(q => (
              <button key={q} className="example" onClick={() => ask(q)} disabled={busy || !opts}>{q}</button>))}</div>}
          </section>
        )}
        {msgs.map((m, i) => m.role === 'user'
          ? <div key={i} className="bubble user">{m.text}</div>
          : m.role === 'bot'
            ? <Answer key={i} msg={m} labels={labels} onRetry={() => ask(m.question, m.cfg)} />
            : <div key={i} className="bubble error" role="alert">
                <p><Icon d={I.alert} /> {m.text}</p>
                <button className="btn ghost" onClick={() => ask(m.question, m.cfg)} disabled={busy}><Icon d={I.retry} size={16} /> Try again</button>
              </div>)}
        {busy && <Pending cfg={msgs[msgs.length - 1]?.cfg || cfg} labels={labels} />}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={e => { e.preventDefault(); ask(input) }}>
        <label htmlFor="q" className="sr-only">Question</label>
        <textarea id="q" ref={inputRef} rows={1} value={input} autoFocus
          placeholder="Ask about the database, e.g. 'Which genre earned the most revenue?'"
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(input) } }} />
        <button className="btn primary" type="submit" disabled={busy || !input.trim() || !opts} aria-label="Send question">
          <Icon d={I.send} /> <span className="hide-sm">Ask</span>
        </button>
        <p className="hint muted small hide-sm">Enter send · Shift+Enter newline · / focus · Alt+1–5 configuration · Alt+T theme</p>
      </form>
    </div>
  )
}
