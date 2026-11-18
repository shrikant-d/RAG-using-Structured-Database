# RAG using Structured Database

Ask questions in plain English about a relational database. The system retrieves the relevant
part of the schema, has an LLM write SQL, runs that SQL read-only, and answers from the result
rows, with a check that every value in the answer appears in those rows.

This repository is the runnable companion to the master's thesis *Extending RAG with Structured
Databases* (TU Ilmenau). It contains the web interface, its local HTTP API, the full retrieval
and generation pipeline, and the two databases used in the thesis: **Chinook** and **TPC-H**
(scale factor 0.01). The model is **Gemma 4 31B on the TU Ilmenau DBIS server, with reasoning
switched off**.

![Chinook question answered by configuration D](docs/screenshot.png)

## The five configurations

They differ only in how the schema reaches the model. You can switch between them in the UI.

| UI label | What the model sees before writing SQL |
|---|---|
| **A · Vector retrieval** | The *k* = 5 tables whose descriptions are most similar to the question (ChromaDB, all-MiniLM-L6-v2) |
| **B · Full schema stuffing** ($B_1$ in the thesis) | The DDL of every table |
| **B · LLM self-selection** ($B_2$) | The table names; the model picks the tables, then gets their DDL |
| **C · Agentic tool exploration** | Nothing up front; a ReAct agent calls `list_tables`, `describe_table`, `sample_rows` |
| **D · Agentic + join graph** | As C, plus `find_join_path` and `get_related_tables` over the foreign-key graph |

All five then share the same steps: up to two SQL repair attempts, the answer generator and the grounding check.

---

## Quick start

You need **Python 3.11 or newer** (tested on 3.13), **Git**, and an **API key for the TU Ilmenau
DBIS LLM server**. You do not need Node.js, because the built web page is included.

### 1. Get the code

```bash
git clone https://github.com/shrikant-d/RAG-using-Structured-Database.git
cd RAG-using-Structured-Database
```

### 2. Create a virtual environment and install the dependencies

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your API key

Copy the template:

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Then open `.env` and paste your key after `TUI_LLM_API_KEY=`. Create the key in the server's
Open WebUI: **Settings → Account → API keys**. Leave `TUI_REASONING_EFFORT=none` as it is,
which keeps Gemma's reasoning off.

`.env` is listed in `.gitignore`. The key stays in the Python process and is never sent to
the browser.

### 4. Start the app

```bash
python -m webui.server
```

### 5. Open the page

Go to **http://127.0.0.1:8600**.

1. Pick a **Configuration**, keep **Gemma 4 31B** as the model, and pick a **Database**.
2. Type a question and press **Enter**.
3. Open the panels under the answer to check it:
   - **SQL**: the executed query, plus the original query if a repair changed it.
   - **Result**: the rows, sortable by column.
   - **Schema used**: the tables the configuration put in front of the model.
   - **Agent steps** (C and D only): every tool call with its input and output.
   - **Metrics**: latency, tool calls, repairs, tokens.

Keyboard shortcuts: `/` focuses the question box, `Alt+1`…`Alt+5` switch the configuration,
and `Alt+T` toggles light/dark.

### Questions to try

| Database | Question |
|---|---|
| chinook | How many tracks belong to the Jazz genre? |
| chinook | Which artist earned the most revenue from track sales? |
| chinook | Which media type is used by the most tracks that appear on the Music playlist? |
| tpch_sf001 | How many customers are there in each market segment? |
| tpch_sf001 | Which nation has the most suppliers? |

The full question sets used in the thesis, with gold SQL, are in
[`questions/chinook.json`](questions/chinook.json) (25 questions) and
[`questions/tpch.json`](questions/tpch.json) (the 22 TPC-H queries).

---

## Check the installation without the LLM

These checks need neither the API key nor the network. They cover:
- both databases load;
- write statements are refused;
- the database files are opened read-only;
- the API's input checks work.

```bash
pip install pytest httpx
python -m pytest tests
```

## Notes on reproducing thesis numbers

- **This is the interactive interface, not the benchmark harness.** It uses the same pipeline
  functions with temperature 0, but it allows two repair attempts. Answers and SQL should
  match the thesis' reasoning-off runs. Latency depends on the load on the shared server.
- **The first question under configuration A takes longer.** ChromaDB downloads its embedding
  model (about 80 MB) once and builds a local index in `.chroma/`.
- **Each question runs alone.** The server answers one question at a time, so a second tab
  waits for the first.

## Repository layout

```
core/          SQL generation, execution with the read-only guard, repair loop,
               answer generation, grounding check, LLM client (core/llm.py)
paths/         the five schema-retrieval configurations (registry.py builds them)
db/            database helpers; build_tpch.py regenerates the TPC-H database
data/          chinook.sqlite, tpch_sf001.sqlite
webui/         server.py (FastAPI) and frontend/ (React + Vite; dist/ is the built page)
questions/     thesis question sets with gold SQL
tests/         offline smoke tests
```

## Optional: change the frontend

You need Node.js 20 or newer for this.

```bash
cd webui/frontend
npm ci
npm run dev       # http://localhost:5173, forwards /api to the Python server on :8600
npm run build     # writes webui/frontend/dist, which python -m webui.server serves
```

## Optional: regenerate the TPC-H database

```bash
pip install duckdb
python db/build_tpch.py --sf 0.01
```

This uses DuckDB's `tpch` extension, which implements the official `dbgen` algorithm, and adds
the primary and foreign keys from the TPC-H specification. Configuration D builds its join
graph from those foreign keys.

## Data sources

- **Chinook**: sample music-store database by Luis Rocha,
  [lerocha/chinook-database](https://github.com/lerocha/chinook-database) (MIT licence).
- **TPC-H**: the schema and data generator are from the
  [TPC-H benchmark](https://www.tpc.org/tpch/). The data was generated at SF 0.01 with DuckDB's
  `tpch` extension. Results from this repository are not TPC-H benchmark results.

## Security

- The server binds to `127.0.0.1` only and refuses requests whose `Host` header is not
  `127.0.0.1` or `localhost`. This guards against DNS rebinding.
- Generated SQL passes two independent guards:
  - a SELECT-only check that rejects multiple statements and data-modifying CTEs;
  - SQLite connections opened in read-only mode.
- The API key is read from `.env` on the server side. It never appears in a response, and
  `.env` is gitignored.
- Questions are limited to 2,000 characters, and the API documentation routes are disabled.
- `pip-audit` reports advisories against `chromadb`. All of them concern Chroma's
  HTTP server mode. This project uses Chroma embedded in the process (`PersistentClient`)
  and opens no Chroma port, so none of them applies. `npm audit` reports no
  vulnerabilities.

Do not expose this server to a network: it is a local, single-user research tool.

## Licence

Code: [MIT](LICENSE). The databases in `data/` keep the terms of their sources (see *Data sources*).
