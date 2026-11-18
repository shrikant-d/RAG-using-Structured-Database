from sqlalchemy import create_engine, inspect


def get_engine(db_url: str):
    """db_url examples: 'sqlite:///spider/database/chinook/chinook.sqlite',
    'postgresql://user:pass@localhost/tpch'."""
    return create_engine(db_url)


def get_inspector(engine):
    return inspect(engine)
