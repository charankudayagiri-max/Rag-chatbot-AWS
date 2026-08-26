"""
database.py — Relational Database Connection Layer

Establishes the SQLAlchemy engine and connection pooling. Provides a
FastAPI dependency to retrieve transactional sessions.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import DATABASE_URL, logger

# Create engine with connection pool configurations
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # checks connection health before query
        pool_size=10,        # max persistent connections
        max_overflow=20,     # additional connections allowed under load
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as exc:
    logger.exception("Failed to initialise database engine for URL: %s", DATABASE_URL)
    raise exc

Base = declarative_base()

def init_db() -> None:
    """Create all tables in the relational database if they do not exist."""
    logger.info("Initializing relational database schema...")
    try:
        import app.db_models  # ensure models are imported to register metadata
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.exception("Failed to initialize database tables:")
        raise exc

def get_db():
    """
    FastAPI dependency yielding a transactional DB session.

    Ensures that the connection is closed after the request lifecycle
    is completed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
