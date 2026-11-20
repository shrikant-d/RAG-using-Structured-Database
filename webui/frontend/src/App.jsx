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

// what each configuration puts in front of the model (shown under its name)
const PATH_NOTES = {
  A_vector: 'Embeds the question and retrieves the most similar tables from a vector index.',
  B_stuff: 'Puts the whole schema in the prompt. The baseline.',
  B_select: 'The model picks the tables it needs from a list, then sees only those.',
  C_agentic: 'An agent explores the schema with tools before writing SQL.',
  D_graph: 'The agent from C, plus a join graph that answers how tables connect.',
}

// ---- inline SVG icons (Lucide paths, one stroke weight)
const Icon = ({ d, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d.split('|').map((p, i) => <path key={i} d={p} />)}
  </svg>
)
const I = {
  send: 'M5 12h14|M12 5l7 7-7 7',
  sun: 'M12 3v1|M12 20v1|M3 12h1|M20 12h1|M5.6 5.6l.7.7|M17.7 17.7l.7.7|M5.6 18.4l.7-.7|M17.7 6.3l.7-.7|M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
  moon: 'M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z',
  trash: 'M3 6h18|M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6|M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2',
  db: 'M12 8c4.97 0 9-1.34 9-3s-4.03-3-9-3-9 1.34-9 3 4.03 3 9 3Z|M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5|M3 12c0 1.66 4 3 9 3s9-1.34 9-3',
  copy: 'M8 8h12v12H8z|M4 16V4h12',
  check: 'M20 6 9 17l-5-5',
  alert: 'M12 9v4|M12 17h.01|M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z',
  retry: 'M3 12a9 9 0 1 0 3-6.7L3 8|M3 3v5h5',
  sliders: 'M4 21v-7|M4 10V3|M12 21v-9|M12 8V3|M20 21v-5|M20 12V3|M1 14h6|M9 8h6|M17 16h6',
  up: 'm18 15-6-6-6 6',
  down: 'm6 9 6 6 6-6',
  updown: 'm7 15 5 5 5-5|m7 9 5-5 5 5',
}

// ---- SQL highlighting without a dependency
const KW = new Set(('select from where group by order having limit offset join inner left right outer full cross on as and or not in is null like between ' +
  'case when then else end distinct union all intersect except with exists asc desc count sum avg min max cast coalesce round strftime lower upper over partition ' +
  'create table primary key foreign references integer text real numeric varchar char date decimal').split(' '))
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
    <button className="btn quiet sm" aria-label="Copy SQL"
      onClick={() => navigator.clipboard.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1200) })}>
      <Icon d={done ? I.check : I.copy} size={15} />{done ? 'Copied' : 'Copy'}
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
  if (!columns.length) return <p className="muted">The query ran and returned no rows.</p>
  return (
    <>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Query result">
        <table>
          <thead><tr>{columns.map((c, i) => (
            <th key={i} aria-sort={sort.col === i ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none'}>
              <button onClick={() => setSort(s => ({ col: i, dir: s.col === i ? -s.dir : 1 }))}>
                {c}<span className="sort-ind"><Icon size={13} d={sort.col === i ? (sort.dir > 0 ? I.up : I.down) : I.updown} /></span>
              </button>
            </th>))}</tr></thead>
          <tbody>{sorted.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === 'number' ? 'num' : ''}>{v === null ? <span className="muted">NULL</span> : String(v)}</td>)}</tr>
          ))}</tbody>
        </table>
      </div>
      <p className="muted small">{total} row{total === 1 ? '' : 's'}{total > rows.length ? `, first ${rows.length} shown` : ''}. Click a column to sort.</p>
    </>
  )
}

function fmtInput(v) {
  if (v == null) return ''
  if (typeof v === 'string') return v
  return Object.entries(v).map(([k, x]) => `${k}=${typeof x === 'string' ? x : JSON.stringify(x)}`).join(', ')
}

// accessible tabs: arrow keys move between tabs, only the active panel is in the DOM
function Tabs({ tabs, initial, label }) {
  const [active, setActive] = useState(initial)
  const refs = useRef([])
  const idx = tabs.findIndex(t => t.id === active)
  const onKey = e => {
    const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key]
    if (!d) return
    const n = (idx + d + tabs.length) % tabs.length
    setActive(tabs[n].id)
    refs.current[n]?.focus()
  }
  const uid = useMemo(() => Math.random().toString(36).slice(2, 8), [])
  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} className="tablist" onKeyDown={onKey}>
        {tabs.map((t, i) => (
          <button key={t.id} ref={el => (refs.current[i] = el)} role="tab" id={`${uid}-${t.id}`}
            aria-selected={t.id === active} aria-controls={`${uid}-${t.id}-p`} tabIndex={t.id === active ? 0 : -1}
            onClick={() => setActive(t.id)}>
            {t.title}{t.count != null && <span className="count">{t.count}</span>}
          </button>))}
      </div>
      <div role="tabpanel" id={`${uid}-${active}-p`} aria-labelledby={`${uid}-${active}`} className="tabpanel" tabIndex={0}>
        {tabs[idx]?.body}
      </div>
    </div>
  )
}

function Answer({ msg, labels, onRetry }) {
  const { r, cfg } = msg
  const m = r.metrics
  const tabs = []
  if (r.sql && !r.error) tabs.push({ id: 'result', title: 'Result', count: r.row_count,
    body: <ResultTable columns={r.columns} rows={r.rows} total={r.row_count} /> })
  if (r.sql) tabs.push({ id: 'sql', title: 'SQL', body: (<>
    <div className="code-head"><span className="muted small">{m.retries ? `Final query, after ${m.retries} repair${m.retries > 1 ? 's' : ''}` : 'Executed query'}</span><CopyButton text={r.sql} /></div>
    <SqlCode sql={r.sql} />
    {r.generated_sql && r.generated_sql !== r.sql && (<>
      <p className="muted small">First attempt, which failed:</p>
      <SqlCode sql={r.generated_sql} />
    </>)}
  </>) })
  tabs.push({ id: 'schema', title: 'Schema', count: r.schema.tables.length, body: r.schema.tables.length ? (<>
    <p className="muted small">Tables this configuration put in front of the model.</p>
    <div className="chips">{r.schema.tables.map(t => <span key={t} className="chip">{t}</span>)}</div>
    <SqlCode sql={r.schema.ddl} />
  </>) : <p className="muted">No tables were recorded for this run.</p> })
  if (r.steps.length || r.trace.length) tabs.push({ id: 'steps', title: 'Agent steps', count: r.steps.length || r.trace.length,
    body: r.steps.length ? (
      <ol className="steps">{r.steps.map((s, i) => (
        <li key={i}><details>
          <summary><code>{s.tool}</code> <span className="muted">{fmtInput(s.input)}</span></summary>
          <pre className="code small-code">{s.output ?? '(no output)'}</pre>
        </details></li>))}</ol>
    ) : <p className="muted">Trace: {r.trace.join(' → ')}</p> })
  tabs.push({ id: 'metrics', title: 'Metrics', body: (
    <dl className="metrics">
      <div><dt>Total</dt><dd>{m.total_s.toFixed(2)} s</dd></div>
      <div><dt>Generation</dt><dd>{(m.generation_ms / 1000).toFixed(2)} s</dd></div>
      {m.setup_ms > 0 && <div><dt>Path setup</dt><dd>{(m.setup_ms / 1000).toFixed(2)} s</dd></div>}
      <div><dt>Tool calls</dt><dd>{m.tool_calls}</dd></div>
      <div><dt>Repairs</dt><dd>{m.retries}</dd></div>
      <div><dt>Prompt tokens</dt><dd>{m.prompt_tokens.toLocaleString()}</dd></div>
      <div><dt>Completion tokens</dt><dd>{m.completion_tokens.toLocaleString()}</dd></div>
      <div><dt>Throughput</dt><dd>{m.tok_s} tok/s</dd></div>
    </dl>) })

  return (
    <section className="turn-answer" aria-label="Answer">
      <p className="answer-text">{r.answer}</p>
      <div className="facts">
        {r.grounding && (r.grounding.grounded
          ? <span className="status ok"><Icon d={I.check} size={14} />Grounded in the result</span>
          : <span className="status warn"><Icon d={I.alert} size={14} />Not found in the result: {r.grounding.flagged.join(', ')}</span>)}
        {r.error && <span className="status err"><Icon d={I.alert} size={14} />{r.error}</span>}
        <span className="fact">{m.total_s.toFixed(1)} s</span>
        <span className="fact">{(m.prompt_tokens + m.completion_tokens).toLocaleString()} tokens</span>
        {m.retries > 0 && <span className="fact">{m.retries} repair{m.retries > 1 ? 's' : ''}</span>}
      </div>
      <Tabs tabs={tabs} initial={tabs[0].id} label="Evidence for this answer" />
      {r.error && <button className="btn outline" onClick={onRetry}><Icon d={I.retry} size={16} />Ask again</button>}
    </section>
  )
}

function Pending({ cfg, labels }) {
  const [t, setT] = useState(0)
  useEffect(() => { const s = Date.now(); const id = setInterval(() => setT((Date.now() - s) / 1000), 200); return () => clearInterval(id) }, [])
  return (
    <section className="turn-answer pending" role="status" aria-live="polite">
      <p className="muted small">Running {labels.path[cfg.path] || cfg.path} on {cfg.db} · {t.toFixed(0)} s</p>
      <div className="skeleton" aria-hidden="true"><span /><span /><span /></div>
    </section>
  )
}

export default function App() {
  const [opts, setOpts] = useState(null)
  const [optsErr, setOptsErr] = useState(null)
  const [cfg, setCfg] = useState({ path: 'B_stuff', model: 'tui:gemma4:31b', db: 'chinook' })
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [showCfg, setShowCfg] = useState(false)
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

  // group the flat message list into question/answer turns
  const turns = useMemo(() => {
    const out = []
    for (const m of msgs) {
      if (m.role === 'user') out.push({ q: m, a: null })
      else if (out.length) out[out.length - 1].a = m
    }
    return out
  }, [msgs])

  async function ask(question, c = cfg) {
    question = question.trim()
    if (!question || busy) return
    setBusy(true)
    setInput('')
    setShowCfg(false)
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
  const dbs = opts?.databases || []

  return (
    <div className="app">
      <aside className={`side${showCfg ? ' open' : ''}`} aria-label="Configuration">
        <div className="brand">
          <span className="mark"><Icon d={I.db} size={18} /></span>
          <div><h1>Structured RAG</h1><p className="muted small">Questions over relational databases</p></div>
        </div>

        <fieldset className="group" disabled={!opts}>
          <legend>Schema retrieval</legend>
          <div className="paths">
            {opts?.paths.map((p, i) => (
              <label key={p.id} className="path">
                <input type="radio" name="path" value={p.id} checked={cfg.path === p.id}
                  onChange={() => setCfg({ ...cfg, path: p.id })} />
                <span className="path-body">
                  <span className="path-name">{p.label}<kbd>Alt {i + 1}</kbd></span>
                  <span className="path-note">{PATH_NOTES[p.id]}</span>
                </span>
              </label>))}
          </div>
        </fieldset>

        <label className="group">
          <span className="legend">Database</span>
          <select value={cfg.db} onChange={e => setCfg({ ...cfg, db: e.target.value })} disabled={!opts}>
            <optgroup label="Included">{dbs.filter(d => !d.startsWith('spider/')).map(d => <option key={d} value={d}>{d}</option>)}</optgroup>
            {dbs.some(d => d.startsWith('spider/')) &&
              <optgroup label="Spider">{dbs.filter(d => d.startsWith('spider/')).map(d => <option key={d} value={d}>{d.slice(7)}</option>)}</optgroup>}
          </select>
        </label>

        {opts?.models.length === 1 ? (
          <div className="group">
            <span className="legend">Model</span>
            <p className="model-fixed">{opts.models[0].label}</p>
          </div>
        ) : (
          <label className="group">
            <span className="legend">Model</span>
            <select value={cfg.model} onChange={e => setCfg({ ...cfg, model: e.target.value })} disabled={!opts}>
              {opts?.models.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>)}

        <p className="keys muted small">
          <kbd>/</kbd> focus question · <kbd>Alt T</kbd> theme
        </p>
      </aside>

      <div className="main">
        <header className="topbar">
          <button className="btn quiet cfg-toggle" onClick={() => setShowCfg(s => !s)} aria-expanded={showCfg}>
            <Icon d={I.sliders} size={16} />{labels.path[cfg.path]?.split(' · ')[0] || 'Settings'} · {cfg.db}
          </button>
          <p className="crumb muted small">{labels.path[cfg.path] || ''}<span className="sep">/</span>{cfg.db}</p>
          <div className="actions">
            <button className="icon-btn" onClick={() => setMsgs([])} disabled={!msgs.length || busy} aria-label="Clear the conversation" title="Clear">
              <Icon d={I.trash} />
            </button>
            <button className="icon-btn" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title="Theme (Alt+T)">
              <Icon d={theme === 'dark' ? I.sun : I.moon} />
            </button>
          </div>
        </header>

        <main className="thread">
          {optsErr && <div className="notice err" role="alert"><Icon d={I.alert} />The API is not reachable ({optsErr}). Start it with <code>python -m webui.server</code>.</div>}
          {!msgs.length && !optsErr && (
            <section className="empty">
              <h2>Ask {cfg.db.startsWith('spider/') ? cfg.db.slice(7) : cfg.db} a question</h2>
              <p className="muted">Each answer comes with its evidence: the rows it was read from, the SQL that produced them, the schema the configuration retrieved, and every tool call the agent made.</p>
              {examples.length > 0 && <ul className="examples">{examples.map(q => (
                <li key={q}><button className="example" onClick={() => ask(q)} disabled={busy || !opts}>
                  <span>{q}</span><Icon d={I.send} size={15} />
                </button></li>))}</ul>}
            </section>
          )}
          {turns.map((t, i) => (
            <article key={i} className="turn">
              <div className="turn-q">
                <p className="bubble-user">{t.q.text}</p>
                <p className="muted small">{labels.path[t.q.cfg.path] || t.q.cfg.path} · {t.q.cfg.db}</p>
              </div>
              {t.a?.role === 'bot' && <Answer msg={t.a} labels={labels} onRetry={() => ask(t.a.question, t.a.cfg)} />}
              {t.a?.role === 'error' && (
                <div className="notice err" role="alert">
                  <Icon d={I.alert} /><span>{t.a.text}</span>
                  <button className="btn outline sm" onClick={() => ask(t.a.question, t.a.cfg)} disabled={busy}><Icon d={I.retry} size={15} />Ask again</button>
                </div>)}
              {!t.a && busy && i === turns.length - 1 && <Pending cfg={t.q.cfg} labels={labels} />}
            </article>
          ))}
          <div ref={endRef} />
        </main>

        <form className="composer" onSubmit={e => { e.preventDefault(); ask(input) }}>
          <div className="composer-box">
            <label htmlFor="q" className="sr-only">Question</label>
            <textarea id="q" ref={inputRef} rows={1} value={input} autoFocus
              placeholder="Ask about the data, e.g. Which genre earned the most revenue?"
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(input) } }} />
            <button className="btn primary" type="submit" disabled={busy || !input.trim() || !opts} aria-label="Ask">
              <span className="hide-sm">Ask</span><Icon d={I.send} size={16} />
            </button>
          </div>
          <p className="hint muted small hide-sm">Enter to ask · Shift+Enter for a new line</p>
        </form>
      </div>
    </div>
  )
}
