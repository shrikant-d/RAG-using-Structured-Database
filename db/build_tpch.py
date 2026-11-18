"""One-off generator for the TPC-H evaluation database.

Uses DuckDB's bundled `tpch` extension, which implements the official dbgen
algorithm (fixed value grammars, deterministic cardinality formulas per the
TPC-H spec, revision 3.0.0 -- verified directly against tpc.org before this
was written, see design_notes.md §31) rather than the reference C dbgen tool,
which requires a manual EULA-gated download and a local C build. The schema
and data this produces are spec-compliant; only the *scale factor* is a choice
made for this thesis (small, for a fast local eval loop -- see below), not a
deviation from the standard.

Usage:
    python db/build_tpch.py --sf 0.01
"""
import argparse
import decimal
import os
import sqlite3

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DuckDB's tpch extension populates the standard columns but the spec itself
# says the PK/FK annotations in its table layout are "for information only"
# (Clause 1.4.1) -- so `CREATE TABLE ... AS SELECT` carries no constraint
# metadata at all. That's silently fatal for this project specifically:
# Path D's build_graph_known (paths/path_d_graph.py) builds its join graph
# from inspector.get_foreign_keys() alone, so a TPC-H DB with no declared FKs
# would give Path D nodes with zero edges -- find_join_path always empty --
# defeating the one thing Path D exists to test. DDL below declares the real
# TPC-H relationships (spec Figure 2, the schema ER diagram) explicitly.
SCHEMA_DDL = """
CREATE TABLE region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name TEXT NOT NULL,
    r_comment TEXT
);
CREATE TABLE nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name TEXT NOT NULL,
    n_regionkey INTEGER NOT NULL REFERENCES region(r_regionkey),
    n_comment TEXT
);
CREATE TABLE part (
    p_partkey INTEGER PRIMARY KEY,
    p_name TEXT NOT NULL,
    p_mfgr TEXT NOT NULL,
    p_brand TEXT NOT NULL,
    p_type TEXT NOT NULL,
    p_size INTEGER NOT NULL,
    p_container TEXT NOT NULL,
    p_retailprice REAL NOT NULL,
    p_comment TEXT
);
CREATE TABLE supplier (
    s_suppkey INTEGER PRIMARY KEY,
    s_name TEXT NOT NULL,
    s_address TEXT NOT NULL,
    s_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey),
    s_phone TEXT NOT NULL,
    s_acctbal REAL NOT NULL,
    s_comment TEXT
);
CREATE TABLE customer (
    c_custkey INTEGER PRIMARY KEY,
    c_name TEXT NOT NULL,
    c_address TEXT NOT NULL,
    c_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey),
    c_phone TEXT NOT NULL,
    c_acctbal REAL NOT NULL,
    c_mktsegment TEXT NOT NULL,
    c_comment TEXT
);
CREATE TABLE partsupp (
    ps_partkey INTEGER NOT NULL REFERENCES part(p_partkey),
    ps_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    ps_availqty INTEGER NOT NULL,
    ps_supplycost REAL NOT NULL,
    ps_comment TEXT,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);
CREATE TABLE orders (
    o_orderkey INTEGER PRIMARY KEY,
    o_custkey INTEGER NOT NULL REFERENCES customer(c_custkey),
    o_orderstatus TEXT NOT NULL,
    o_totalprice REAL NOT NULL,
    o_orderdate TEXT NOT NULL,
    o_orderpriority TEXT NOT NULL,
    o_clerk TEXT NOT NULL,
    o_shippriority INTEGER NOT NULL,
    o_comment TEXT
);
CREATE TABLE lineitem (
    l_orderkey INTEGER NOT NULL REFERENCES orders(o_orderkey),
    l_partkey INTEGER NOT NULL,
    l_suppkey INTEGER NOT NULL,
    l_linenumber INTEGER NOT NULL,
    l_quantity REAL NOT NULL,
    l_extendedprice REAL NOT NULL,
    l_discount REAL NOT NULL,
    l_tax REAL NOT NULL,
    l_returnflag TEXT NOT NULL,
    l_linestatus TEXT NOT NULL,
    l_shipdate TEXT NOT NULL,
    l_commitdate TEXT NOT NULL,
    l_receiptdate TEXT NOT NULL,
    l_shipinstruct TEXT NOT NULL,
    l_shipmode TEXT NOT NULL,
    l_comment TEXT,
    PRIMARY KEY (l_orderkey, l_linenumber),
    FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)
);
"""

# Load order matters: a table can only be created/inserted after every table
# its FOREIGN KEY clauses point at already exists.
TABLES = ["region", "nation", "part", "supplier", "customer", "partsupp", "orders", "lineitem"]


def build(sf: float, out_path: str) -> None:
    if os.path.exists(out_path):
        os.remove(out_path)

    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")

    sq = sqlite3.connect(out_path)
    sq.executescript(SCHEMA_DDL)
    for t in TABLES:
        cols = [r[1] for r in sq.execute(f"PRAGMA table_info({t})").fetchall()]
        rows = con.execute(f"SELECT {', '.join(cols)} FROM {t}").fetchall()
        # duckdb hands back DECIMAL columns as decimal.Decimal, which sqlite3
        # cannot bind directly
        rows = [tuple(float(v) if isinstance(v, decimal.Decimal) else v for v in row)
                for row in rows]
        placeholders = ", ".join("?" * len(cols))
        sq.executemany(f"INSERT INTO {t} VALUES ({placeholders})", rows)
    sq.commit()
    sq.close()
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.01,
                    help="TPC-H scale factor. 0.01 gives ~60k lineitem rows -- enough "
                         "for meaningful multi-table joins while keeping local qwen2.5:7b "
                         "eval sweeps fast; this thesis uses TPC-H as a fixed-schema "
                         "known-schema stress test, not a performance benchmark, so a "
                         "larger SF buys nothing the RQs need.")
    args = ap.parse_args()
    tag = str(args.sf).replace(".", "")
    out_path = os.path.join(ROOT, "data", f"tpch_sf{tag}.sqlite")
    build(args.sf, out_path)
    print(f"wrote {out_path}")
