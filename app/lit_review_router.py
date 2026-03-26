"""
FastAPI router for the Literature Review feature.

Endpoints:
  POST /api/lit-review/sessions             Create a new review session
  GET  /api/lit-review/sessions             List all sessions
  GET  /api/lit-review/sessions/{id}        Get session details + stats
  POST /api/lit-review/sessions/{id}/start  Start (or resume) background processing
  POST /api/lit-review/sessions/{id}/stop   Stop background processing
  GET  /api/lit-review/sessions/{id}/papers Get all papers in a session
  POST /api/lit-review/sessions/{id}/generate  Generate the review text
  GET  /api/lit-review/sessions/{id}/review    Get the review text
  GET  /api/lit-review/models               List available models
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlmodel import Session as DBSession

from app.config import settings, DATA_DIR
from app.database import engine
from app.lit_review_models import (
    LitReviewSession,
    LitReviewPaper,
    get_session_by_id,
    get_papers_for_session,
    get_queued_papers,
    get_relevant_papers,
    paper_ss_id_exists,
)
from app.paper_fetcher import (
    SemanticScholarClient,
    resolve_initial_paper,
    download_pdf,
    extract_metadata,
)
from app.relevance_agent import check_relevance, get_api_key, AVAILABLE_MODELS
from app.lit_review_generator import generate_literature_review

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/lit-review", tags=["Literature Review"])

# Track running background tasks per session_id
_running_tasks: dict[int, bool] = {}

LIT_REVIEW_PDF_DIR = DATA_DIR / "lit_review_pdfs"


# ── Request / Response schemas ────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    field_description: str
    initial_query: str
    max_papers: Optional[int] = None
    research_model_provider: str = "mistral"
    research_model_name: str = "open-mistral-nemo"
    writing_model_provider: str = "mistral"
    writing_model_name: str = "mistral-large-latest"
    citation_format: str = "APA"


class GenerateReviewRequest(BaseModel):
    target_words: int = 4000


# ── Helper: serialise a session ───────────────────────────────────────────────

def _session_dict(s: LitReviewSession) -> dict:
    return {
        "id": s.id,
        "field_description": s.field_description,
        "initial_query": s.initial_query,
        "initial_paper_title": s.initial_paper_title,
        "max_papers": s.max_papers,
        "status": s.status,
        "research_model_provider": s.research_model_provider,
        "research_model_name": s.research_model_name,
        "writing_model_provider": s.writing_model_provider,
        "writing_model_name": s.writing_model_name,
        "citation_format": s.citation_format,
        "papers_processed": s.papers_processed,
        "papers_relevant": s.papers_relevant,
        "papers_queued": s.papers_queued,
        "error_message": s.error_message,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
        "has_review": bool(s.review_text),
        "is_running": _running_tasks.get(s.id, False),
    }


def _paper_dict(p: LitReviewPaper) -> dict:
    return {
        "id": p.id,
        "session_id": p.session_id,
        "title": p.title,
        "authors": p.get_authors_list(),
        "year": p.year,
        "venue": p.venue,
        "doi": p.doi,
        "arxiv_id": p.arxiv_id,
        "semantic_scholar_id": p.semantic_scholar_id,
        "url": p.url,
        "relevance_score": p.relevance_score,
        "relevance_reason": p.relevance_reason,
        "key_contribution": p.key_contribution,
        "is_seed": p.is_seed,
        "depth": p.depth,
        "status": p.status,
        "has_pdf": bool(p.pdf_path),
        "added_at": p.added_at.isoformat() if p.added_at else None,
    }


# ── Background processing task ────────────────────────────────────────────────

async def _process_session(session_id: int):
    """
    Main BFS loop:
      1. Pop a paper from the queue.
      2. Fetch its references via Semantic Scholar.
      3. For each reference, run relevance check.
      4. If relevant: save to DB, queue for processing, download PDF.
      5. Repeat until queue is empty or max_papers reached.
    """
    _running_tasks[session_id] = True
    ss_client = SemanticScholarClient(
        api_key=getattr(settings, "SEMANTIC_SCHOLAR_API_KEY", None) or None
    )

    try:
        with DBSession(engine) as db:
            lit_session = get_session_by_id(db, session_id)
            if not lit_session:
                logger.error(f"Session {session_id} not found")
                return

            # If this is the first run, resolve the initial paper
            if lit_session.status == "created":
                logger.info(f"[Session {session_id}] Resolving initial paper: {lit_session.initial_query}")
                seed_paper_ss = await resolve_initial_paper(lit_session.initial_query, ss_client)

                if not seed_paper_ss:
                    lit_session.status = "error"
                    lit_session.error_message = (
                        f"Could not find paper: '{lit_session.initial_query}' "
                        f"on Semantic Scholar. Try a more specific title or an arxiv ID."
                    )
                    db.add(lit_session)
                    db.commit()
                    return

                meta = extract_metadata(seed_paper_ss)
                lit_session.initial_paper_title = meta["title"]
                lit_session.initial_paper_ss_id = meta["semantic_scholar_id"]
                lit_session.status = "running"
                db.add(lit_session)

                # Add seed paper to the session
                seed = LitReviewPaper(
                    session_id=session_id,
                    title=meta["title"],
                    authors=json.dumps(meta["authors"]),
                    abstract=meta["abstract"],
                    year=meta["year"],
                    venue=meta["venue"],
                    doi=meta["doi"],
                    arxiv_id=meta["arxiv_id"],
                    semantic_scholar_id=meta["semantic_scholar_id"],
                    url=meta["url"],
                    pdf_url=meta["pdf_url"],
                    relevance_score=1.0,
                    relevance_reason="Initial seed paper.",
                    key_contribution="Seed paper for this literature review.",
                    is_seed=True,
                    depth=0,
                    status="queued",
                )
                db.add(seed)
                db.commit()
                db.refresh(seed)
                logger.info(f"[Session {session_id}] Seed paper added: {meta['title'][:60]}")

            lit_session.status = "running"
            db.add(lit_session)
            db.commit()

        # ── BFS loop ──────────────────────────────────────────────────────────
        while _running_tasks.get(session_id, False):
            with DBSession(engine) as db:
                lit_session = get_session_by_id(db, session_id)
                if not lit_session or lit_session.status not in ("running",):
                    break

                queued = get_queued_papers(db, session_id)
                if not queued:
                    # Nothing left to process — done!
                    lit_session.status = "completed"
                    lit_session.completed_at = datetime.utcnow()
                    db.add(lit_session)
                    db.commit()
                    logger.info(f"[Session {session_id}] Processing complete.")
                    break

                # Check limit
                if (
                    lit_session.max_papers is not None
                    and lit_session.papers_relevant >= lit_session.max_papers
                ):
                    lit_session.status = "completed"
                    lit_session.completed_at = datetime.utcnow()
                    db.add(lit_session)
                    db.commit()
                    logger.info(
                        f"[Session {session_id}] Reached max_papers={lit_session.max_papers}."
                    )
                    break

                paper = queued[0]
                paper.status = "processing"
                db.add(paper)
                db.commit()
                db.refresh(paper)

                paper_ss_id = paper.semantic_scholar_id
                paper_title = paper.title
                paper_depth = paper.depth
                field_desc = lit_session.field_description
                max_p = lit_session.max_papers
                relevant_count = lit_session.papers_relevant
                research_provider = lit_session.research_model_provider
                research_model = lit_session.research_model_name
                writing_provider = lit_session.writing_model_provider

            # Outside DB session: do network I/O
            logger.info(
                f"[Session {session_id}] Processing: '{paper_title[:55]}' (depth={paper_depth})"
            )

            # 1. Download PDF for this paper
            paper_info_for_dl = {
                "paperId": paper_ss_id,
                "openAccessPdf": {"url": None},
                "externalIds": {},
            }
            with DBSession(engine) as db:
                p = db.get(LitReviewPaper, paper.id)
                if p and p.pdf_url:
                    paper_info_for_dl["openAccessPdf"] = {"url": p.pdf_url}
                if p and p.arxiv_id:
                    paper_info_for_dl["externalIds"]["ArXiv"] = p.arxiv_id

            pdf_path = await download_pdf(paper_info_for_dl, LIT_REVIEW_PDF_DIR)

            # 2. Get references from Semantic Scholar
            refs: list[dict] = []
            if paper_ss_id:
                try:
                    refs = await ss_client.get_references(paper_ss_id, max_refs=150)
                    logger.info(
                        f"[Session {session_id}] Got {len(refs)} references for '{paper_title[:40]}'"
                    )
                except Exception as e:
                    logger.warning(f"[Session {session_id}] Reference fetch failed: {e}")

            # 3. Process each reference
            try:
                research_api_key = get_api_key(research_provider)
            except ValueError as e:
                with DBSession(engine) as db:
                    s = get_session_by_id(db, session_id)
                    if s:
                        s.status = "error"
                        s.error_message = str(e)
                        db.add(s)
                        db.commit()
                _running_tasks[session_id] = False
                return

            new_relevant = 0
            for ref in refs:
                if not _running_tasks.get(session_id, False):
                    break

                ref_ss_id = ref.get("paperId")
                if not ref_ss_id:
                    continue

                with DBSession(engine) as db:
                    lit_session = get_session_by_id(db, session_id)
                    # Re-check limits inside the loop
                    if (
                        lit_session
                        and lit_session.max_papers is not None
                        and lit_session.papers_relevant >= lit_session.max_papers
                    ):
                        break

                    if paper_ss_id_exists(db, session_id, ref_ss_id):
                        continue  # Already seen

                ref_meta = extract_metadata(ref)
                if not ref_meta["title"]:
                    continue

                # Check relevance
                is_rel, score, reason, contribution = check_relevance(
                    paper_info=ref_meta,
                    field_description=field_desc,
                    model_provider=research_provider,
                    model_name=research_model,
                    api_key=research_api_key,
                )

                with DBSession(engine) as db:
                    # Double-check (may have been added by another path)
                    if paper_ss_id_exists(db, session_id, ref_ss_id):
                        continue

                    new_paper = LitReviewPaper(
                        session_id=session_id,
                        title=ref_meta["title"],
                        authors=json.dumps(ref_meta["authors"]),
                        abstract=ref_meta["abstract"],
                        year=ref_meta["year"],
                        venue=ref_meta["venue"],
                        doi=ref_meta["doi"],
                        arxiv_id=ref_meta["arxiv_id"],
                        semantic_scholar_id=ref_ss_id,
                        url=ref_meta["url"],
                        pdf_url=ref_meta["pdf_url"],
                        relevance_score=score,
                        relevance_reason=reason,
                        key_contribution=contribution if is_rel else None,
                        depth=paper_depth + 1,
                        status="queued" if is_rel else "irrelevant",
                    )
                    db.add(new_paper)

                    s = get_session_by_id(db, session_id)
                    if s:
                        s.papers_queued = s.papers_queued + (1 if is_rel else 0)
                        if is_rel:
                            s.papers_relevant += 1
                            new_relevant += 1
                        db.add(s)
                    db.commit()

                if is_rel:
                    logger.info(
                        f"[Session {session_id}] + Relevant [{score:.2f}]: {ref_meta['title'][:50]}"
                    )

            # 4. Mark current paper as processed
            with DBSession(engine) as db:
                p = db.get(LitReviewPaper, paper.id)
                if p:
                    p.status = "processed"
                    p.pdf_path = pdf_path
                    p.processed_at = datetime.utcnow()
                    db.add(p)

                s = get_session_by_id(db, session_id)
                if s:
                    s.papers_processed += 1
                    s.papers_queued = max(0, s.papers_queued - 1)
                    db.add(s)
                db.commit()

            logger.info(
                f"[Session {session_id}] Processed '{paper_title[:40]}' "
                f"— {new_relevant} new relevant refs found."
            )

    except Exception as e:
        logger.error(f"[Session {session_id}] Fatal error: {e}", exc_info=True)
        with DBSession(engine) as db:
            s = get_session_by_id(db, session_id)
            if s:
                s.status = "error"
                s.error_message = str(e)
                db.add(s)
                db.commit()
    finally:
        _running_tasks.pop(session_id, None)
        await ss_client.close()
        logger.info(f"[Session {session_id}] Background task ended.")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.post("/sessions")
async def create_session(req: CreateSessionRequest):
    """Create a new literature review session."""
    if req.citation_format not in ("APA", "IEEE", "Nature"):
        raise HTTPException(400, "citation_format must be APA, IEEE, or Nature")

    with DBSession(engine) as db:
        s = LitReviewSession(
            field_description=req.field_description.strip(),
            initial_query=req.initial_query.strip(),
            max_papers=req.max_papers,
            research_model_provider=req.research_model_provider,
            research_model_name=req.research_model_name,
            writing_model_provider=req.writing_model_provider,
            writing_model_name=req.writing_model_name,
            citation_format=req.citation_format,
            status="created",
        )
        db.add(s)
        db.commit()
        db.refresh(s)
        return {"status": "created", "session": _session_dict(s)}


@router.get("/sessions")
async def list_sessions():
    """List all literature review sessions."""
    with DBSession(engine) as db:
        from sqlmodel import select
        sessions = db.exec(
            select(LitReviewSession).order_by(LitReviewSession.created_at.desc())
        ).all()
        return {"sessions": [_session_dict(s) for s in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: int):
    """Get details for a single session."""
    with DBSession(engine) as db:
        s = get_session_by_id(db, session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id} not found")
        return _session_dict(s)


@router.post("/sessions/{session_id}/start")
async def start_session(session_id: int, background_tasks: BackgroundTasks):
    """Start (or resume) background processing for a session."""
    with DBSession(engine) as db:
        s = get_session_by_id(db, session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id} not found")
        if s.status == "completed":
            raise HTTPException(400, "Session already completed. Create a new session.")
        if _running_tasks.get(session_id):
            return {"status": "already_running", "session": _session_dict(s)}

    background_tasks.add_task(_process_session, session_id)
    return {"status": "started", "session_id": session_id}


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: int):
    """Signal the background task to stop after the current paper."""
    _running_tasks[session_id] = False

    with DBSession(engine) as db:
        s = get_session_by_id(db, session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id} not found")
        if s.status == "running":
            s.status = "stopped"
            db.add(s)
            db.commit()
        return {"status": "stopping", "session": _session_dict(s)}


@router.get("/sessions/{session_id}/papers")
async def list_papers(
    session_id: int,
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    """List papers in a session, optionally filtered by status."""
    with DBSession(engine) as db:
        s = get_session_by_id(db, session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id} not found")

        papers = get_papers_for_session(db, session_id)
        if status:
            papers = [p for p in papers if p.status == status]

        total = len(papers)
        papers = papers[offset: offset + limit]
        return {
            "total": total,
            "papers": [_paper_dict(p) for p in papers],
        }


@router.post("/sessions/{session_id}/generate")
async def generate_review(
    session_id: int,
    req: GenerateReviewRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger literature review generation.
    Runs in background; poll GET /sessions/{id} for status.
    """
    with DBSession(engine) as db:
        s = get_session_by_id(db, session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id} not found")
        if s.status not in ("completed", "stopped"):
            raise HTTPException(
                400,
                "Session must be completed or stopped before generating the review. "
                f"Current status: {s.status}",
            )
        if _running_tasks.get(session_id):
            raise HTTPException(400, "A task is already running for this session.")

    background_tasks.add_task(_run_generate, session_id, req.target_words)
    return {"status": "generating", "session_id": session_id}


async def _run_generate(session_id: int, target_words: int):
    _running_tasks[session_id] = True
    try:
        with DBSession(engine) as db:
            s = get_session_by_id(db, session_id)
            if not s:
                return

            papers = get_relevant_papers(db, session_id)
            # Also include the seed paper
            from sqlmodel import select
            seed_papers = db.exec(
                select(LitReviewPaper)
                .where(LitReviewPaper.session_id == session_id)
                .where(LitReviewPaper.is_seed == True)
            ).all()
            all_paper_ids = {p.id for p in papers}
            for sp in seed_papers:
                if sp.id not in all_paper_ids:
                    papers = [sp] + list(papers)

            if not papers:
                s.status = "error"
                s.error_message = "No relevant papers found to generate a review."
                db.add(s)
                db.commit()
                return

            s.status = "generating"
            db.add(s)
            db.commit()

            logger.info(
                f"[Session {session_id}] Generating review for {len(papers)} papers…"
            )

        # Do the generation outside the DB session (long-running)
        with DBSession(engine) as db:
            s = get_session_by_id(db, session_id)
            papers = get_relevant_papers(db, session_id)
            from sqlmodel import select as sql_select
            seed_papers = db.exec(
                sql_select(LitReviewPaper)
                .where(LitReviewPaper.session_id == session_id)
                .where(LitReviewPaper.is_seed == True)
            ).all()
            all_ids = {p.id for p in papers}
            full_papers = [sp for sp in seed_papers if sp.id not in all_ids] + list(papers)

            try:
                research_key = get_api_key(s.research_model_provider)
                writing_key = get_api_key(s.writing_model_provider)
            except ValueError as e:
                s.status = "error"
                s.error_message = str(e)
                db.add(s)
                db.commit()
                return

            review_text = generate_literature_review(
                session=s,
                papers=full_papers,
                research_provider=s.research_model_provider,
                research_model=s.research_model_name,
                research_api_key=research_key,
                writing_provider=s.writing_model_provider,
                writing_model=s.writing_model_name,
                writing_api_key=writing_key,
                target_words=target_words,
            )

            s.review_text = review_text
            s.status = "completed"
            s.completed_at = datetime.utcnow()
            db.add(s)
            db.commit()
            logger.info(f"[Session {session_id}] Review generated ({len(review_text)} chars).")

    except Exception as e:
        logger.error(f"[Session {session_id}] Review generation failed: {e}", exc_info=True)
        with DBSession(engine) as db:
            s = get_session_by_id(db, session_id)
            if s:
                s.status = "error"
                s.error_message = f"Review generation failed: {e}"
                db.add(s)
                db.commit()
    finally:
        _running_tasks.pop(session_id, None)


@router.get("/sessions/{session_id}/review")
async def get_review(session_id: int):
    """Return the generated review text (Markdown)."""
    with DBSession(engine) as db:
        s = get_session_by_id(db, session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id} not found")
        if not s.review_text:
            raise HTTPException(
                404,
                "Review not yet generated. "
                "POST /sessions/{id}/generate to start generation.",
            )
        return {
            "session_id": session_id,
            "citation_format": s.citation_format,
            "review": s.review_text,
        }


@router.get("/models")
async def list_models():
    """Return the catalogue of available research and writing models."""
    return AVAILABLE_MODELS
