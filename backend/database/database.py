from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import Settings


DATABASE_URL = Settings.from_environment().database_url


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


Base = declarative_base()


def configure_database(database_url):
    """Rebind the shared session factory, primarily for controlled test databases."""
    global DATABASE_URL, engine
    engine.dispose()
    DATABASE_URL = database_url
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    )
    SessionLocal.configure(bind=engine)


def initialize_database():
    """Create current tables and add safe columns to legacy local SQLite tables."""
    # Import registers all model classes with the shared metadata.
    from backend.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if not DATABASE_URL.startswith("sqlite"):
        return
    expected_columns = {
        "analyses": {
            "filename": "VARCHAR",
            "status": "VARCHAR DEFAULT 'completed'",
            "total_passages": "INTEGER DEFAULT 0",
            "searched_passages": "INTEGER DEFAULT 0",
            "source_count": "INTEGER DEFAULT 0",
            "retrieved_source_count": "INTEGER DEFAULT 0",
            "suspicious_match_count": "INTEGER DEFAULT 0",
            "overall_risk": "VARCHAR DEFAULT 'no_strong_evidence'",
            "error_message": "TEXT",
        },
        "match_results": {
            "page_number": "INTEGER",
            "source_title": "VARCHAR",
            "source_url": "VARCHAR",
            "combined_score": "FLOAT DEFAULT 0",
            "risk_level": "VARCHAR DEFAULT 'no_strong_evidence'",
        },
    }
    with engine.begin() as connection:
        for table, columns in expected_columns.items():
            existing = {row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in existing:
                    connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
