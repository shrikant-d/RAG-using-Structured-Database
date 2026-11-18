"""Schema rendering shared by the retrieval paths and the repair loop."""


def render_ddl(inspector, table_names: list[str]) -> str:
    blocks = []
    for t in table_names:
        cols = inspector.get_columns(t)
        col_lines = [f"  {c['name']} {c['type']}" for c in cols]
        fk_lines = [
            f"  FOREIGN KEY ({fk['constrained_columns'][0]}) REFERENCES "
            f"{fk['referred_table']}({fk['referred_columns'][0]})"
            for fk in inspector.get_foreign_keys(t)
        ]
        blocks.append(f"CREATE TABLE {t} (\n" + ",\n".join(col_lines + fk_lines) + "\n);")
    return "\n\n".join(blocks)
