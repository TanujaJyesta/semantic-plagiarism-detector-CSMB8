from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey
)

from sqlalchemy.orm import relationship

from backend.database.database import Base


class Analysis(Base):

    __tablename__ = "analyses"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    source_document = Column(
        String,
        nullable=True
    )

    submitted_document = Column(
        String,
        nullable=True
    )

    filename = Column(String, nullable=True)

    status = Column(String, default="completed", nullable=False)

    total_passages = Column(Integer, default=0)

    searched_passages = Column(Integer, default=0)

    source_count = Column(Integer, default=0)

    retrieved_source_count = Column(Integer, default=0)

    suspicious_match_count = Column(Integer, default=0)

    overall_risk = Column(String, default="no_strong_evidence")

    error_message = Column(Text, nullable=True)

    source_sentence_count = Column(
        Integer,
        default=0
    )

    submitted_sentence_count = Column(
        Integer,
        default=0
    )

    matched_sentence_count = Column(
        Integer,
        default=0
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    matches = relationship(
        "MatchResult",
        back_populates="analysis",
        cascade="all, delete-orphan"
    )

    sources = relationship(
        "SourceRecord",
        back_populates="analysis",
        cascade="all, delete-orphan"
    )


class MatchResult(Base):

    __tablename__ = "match_results"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id"),
        nullable=False
    )

    source_sentence_id = Column(
        Integer,
        nullable=True
    )

    submitted_sentence_id = Column(
        Integer,
        nullable=True
    )

    source_page = Column(
        Integer,
        nullable=True
    )

    submitted_page = Column(
        Integer,
        nullable=True
    )

    source_text = Column(
        Text,
        nullable=False
    )

    submitted_text = Column(
        Text,
        nullable=False
    )

    tfidf_score = Column(
        Float,
        nullable=False
    )

    sbert_score = Column(
        Float,
        nullable=False
    )

    match_type = Column(
        String,
        nullable=False
    )

    explanation = Column(
        Text,
        nullable=False
    )

    page_number = Column(Integer, nullable=True)

    source_title = Column(String, nullable=True)

    source_url = Column(String, nullable=True)

    combined_score = Column(Float, default=0)

    risk_level = Column(String, default="no_strong_evidence")

    analysis = relationship(
        "Analysis",
        back_populates="matches"
    )


class SourceRecord(Base):

    __tablename__ = "source_records"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)

    title = Column(String, nullable=False)

    url = Column(String, nullable=False)

    retrieval_status = Column(String, nullable=False)

    error_message = Column(Text, nullable=True)

    search_score = Column(Float, default=0)

    analysis = relationship("Analysis", back_populates="sources")
