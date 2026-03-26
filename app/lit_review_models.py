"""
Database models for the Literature Review feature.
"""

from datetime import datetime
from typing import Optional
import json

from sqlmodel import Field, SQLModel, Session, select


class LitReviewSession(SQLModel, table=True):
    __tablename__ = "lit_review_session"

    id: Optional[int] = Field(default=None, primary_key=True)
    field_description: str  # Research field / scope of the review
    initial_query: str       # arxiv ID, DOI, URL, or title of seed paper

    # Resolved seed paper info
    initial_paper_title: Optional[str] = None
    initial_paper_ss_id: Optional[str] = None  # Semantic Scholar paper ID

    # Config
    max_papers: Optional[int] = None   # None = unlimited
    research_model_provider: str = "mistral"
    research_model_name: str = "open-mistral-nemo"
    writing_model_provider: str = "mistral"
    writing_model_name: str = "mistral-large-latest"
    citation_format: str = "APA"       # APA | IEEE | Nature

    # Runtime state
    status: str = "created"            # created | running | completed | stopped | error
    papers_processed: int = 0
    papers_relevant: int = 0
    papers_queued: int = 0
    error_message: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Output
    review_text: Optional[str] = None  # Final review in Markdown


class LitReviewPaper(SQLModel, table=True):
    __tablename__ = "lit_review_paper"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="lit_review_session.id", index=True)

    # Paper metadata
    title: str
    authors: Optional[str] = None    # JSON list of author names
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None      # Journal / conference

    # Identifiers
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None   # Local path to downloaded PDF

    # Relevance judgment
    relevance_score: Optional[float] = None   # 0.0 – 1.0
    relevance_reason: Optional[str] = None
    key_contribution: Optional[str] = None    # 1–2 sentence contribution note

    # Processing state
    is_seed: bool = False            # True = the initial paper for this session
    depth: int = 0                   # Citation depth (0 = seed, 1 = its refs, …)
    status: str = "queued"           # queued | processing | processed | failed | irrelevant | skipped

    added_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    def get_authors_list(self) -> list[str]:
        return json.loads(self.authors) if self.authors else []

    def short_citation(self) -> str:
        """Returns 'Author et al. (year)' style short citation."""
        authors = self.get_authors_list()
        if not authors:
            return f"({self.year or 'n.d.'})"
        first_last = authors[0].split()[-1] if authors else "Unknown"
        suffix = " et al." if len(authors) > 2 else (f" & {authors[1].split()[-1]}" if len(authors) == 2 else "")
        return f"{first_last}{suffix} ({self.year or 'n.d.'})"


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_session_by_id(session: Session, session_id: int) -> Optional[LitReviewSession]:
    return session.get(LitReviewSession, session_id)


def get_papers_for_session(session: Session, session_id: int) -> list[LitReviewPaper]:
    return session.exec(
        select(LitReviewPaper)
        .where(LitReviewPaper.session_id == session_id)
        .order_by(LitReviewPaper.added_at)
    ).all()


def get_queued_papers(session: Session, session_id: int) -> list[LitReviewPaper]:
    return session.exec(
        select(LitReviewPaper)
        .where(LitReviewPaper.session_id == session_id)
        .where(LitReviewPaper.status == "queued")
        .order_by(LitReviewPaper.depth, LitReviewPaper.added_at)
    ).all()


def get_relevant_papers(session: Session, session_id: int) -> list[LitReviewPaper]:
    return session.exec(
        select(LitReviewPaper)
        .where(LitReviewPaper.session_id == session_id)
        .where(LitReviewPaper.status == "processed")
        .order_by(LitReviewPaper.relevance_score.desc())
    ).all()


def paper_ss_id_exists(session: Session, session_id: int, ss_id: str) -> bool:
    return session.exec(
        select(LitReviewPaper)
        .where(LitReviewPaper.session_id == session_id)
        .where(LitReviewPaper.semantic_scholar_id == ss_id)
    ).first() is not None
