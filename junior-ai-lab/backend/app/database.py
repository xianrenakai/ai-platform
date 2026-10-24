import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "ai_lab.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Postgres (Neon) free tier drops idle connections; test before reuse.
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_recycle"] = 180
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 5

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def migrate_schema() -> None:
    """Small, idempotent migrations for existing classroom SQLite databases."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    statements = []
    if "display_name" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN display_name VARCHAR(64)")
    if "must_change_password" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT 0")
    if "is_active" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1")
    if "plain_password" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN plain_password VARCHAR(255)")
    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            connection.execute(text("UPDATE users SET display_name = username WHERE display_name IS NULL"))

    # Backfill plain_password for legacy seed users (NULL before this column existed).
    with engine.begin() as connection:
        connection.execute(text(
            "UPDATE users SET plain_password = 'admin123' WHERE username = 'admin' AND plain_password IS NULL"
        ))
        connection.execute(text(
            "UPDATE users SET plain_password = 'teacher123' WHERE role = 'teacher' AND plain_password IS NULL"
        ))
        connection.execute(text(
            "UPDATE users SET plain_password = '123456' WHERE role = 'student' AND plain_password IS NULL"
        ))
        connection.execute(text(
            "UPDATE users SET plain_password = '20261023' WHERE username = '潘唯一' AND plain_password IS NULL"
        ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
