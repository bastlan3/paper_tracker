from datetime import datetime
from typing import Optional
import json

from sqlmodel import Field, SQLModel, Session, create_engine, select

from app.config import settings


class Paper(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    arxiv_id: str = Field(unique=True, index=True)
    title: str
    authors: str  # JSON list
    abstract: str
    categories: str  # comma-separated
    published: datetime
    url: str
    pdf_url: str

    # Our enrichments
    summary: Optional[str] = None
    keywords: Optional[str] = None  # JSON list
    similarity_score: Optional[float] = None
    embedding: Optional[str] = None  # JSON list of floats
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None

    # Metadata
    source: str = "arxiv"  # "arxiv" or "manual"
    is_notified: bool = False
    added_at: datetime = Field(default_factory=datetime.utcnow)

    def get_authors_list(self) -> list[str]:
        return json.loads(self.authors) if self.authors else []

    def get_keywords_list(self) -> list[str]:
        return json.loads(self.keywords) if self.keywords else []

    def get_embedding_list(self) -> list[float]:
        return json.loads(self.embedding) if self.embedding else []


engine = create_engine(settings.DATABASE_URL, echo=False)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


def get_all_papers(session: Session) -> list[Paper]:
    return session.exec(select(Paper).order_by(Paper.published.desc())).all()


def get_paper_by_arxiv_id(session: Session, arxiv_id: str) -> Optional[Paper]:
    return session.exec(select(Paper).where(Paper.arxiv_id == arxiv_id)).first()


def get_unnotified_papers(session: Session) -> list[Paper]:
    return session.exec(
        select(Paper)
        .where(Paper.is_notified == False)
        .where(Paper.summary != None)
        .order_by(Paper.published.desc())
    ).all()


def get_papers_with_embeddings(session: Session) -> list[Paper]:
    return session.exec(
        select(Paper).where(Paper.embedding != None)
    ).all()
