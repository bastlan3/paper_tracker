"""
Literature review generator.

Two-pass approach:
  Pass 1 – Per-paper notes: distil each relevant paper into a structured note
            (~200 words) that captures method, contribution, and relevance.
  Pass 2 – Full review: feed all notes to a powerful LLM and generate a
            publication-grade literature review.

Journal-style guidelines are embedded in the system prompt.
"""

import json
import logging
from typing import Optional

from app.lit_review_models import LitReviewPaper, LitReviewSession

logger = logging.getLogger(__name__)


# ── Citation format details ───────────────────────────────────────────────────

CITATION_FORMAT_DETAILS: dict[str, str] = {
    "APA": """
APA 7th Edition:
- In-text citations: (Author, Year) or Author (Year) for narrative citations.
  Multiple authors: (Author1 & Author2, Year) for two; (Author1 et al., Year) for three+.
- Reference list (alphabetical by first author surname):
  Author, A. A., Author, B. B., & Author, C. C. (Year). Title of article.
  *Journal Name*, *volume*(issue), pages. https://doi.org/xxxxx
- Use hanging indent in reference list.
""".strip(),

    "IEEE": """
IEEE Reference Style:
- In-text citations: numbered in square brackets [1], [2], [1]–[3].
  Cited in order of first appearance.
- Reference list (numbered, in order of citation):
  [1] A. Author, B. Author, and C. Author, "Article title," *Journal Name*,
      vol. X, no. Y, pp. ZZ–ZZ, Mon. Year, doi: 10.xxxx/xxxxxx.
  [2] A. Author, *Book Title*, Xth ed. City, State, Country: Publisher, Year.
""".strip(),

    "Nature": """
Nature / Vancouver Reference Style:
- In-text citations: superscript numbers in order of first appearance (e.g. ¹·²).
  Same number reused for subsequent citations of the same work.
- Reference list (numbered, in order of citation):
  1. Author, A. B., Author, C. D. & Author, E. F. Article title.
     *Journal Abbrev.* **volume**, pages (year).
  2. For preprints: Author, A. B. et al. Preprint title. Preprint at
     https://arxiv.org/abs/XXXX.XXXXX (year).
""".strip(),
}

# ── Review writing guidelines ─────────────────────────────────────────────────

_REVIEW_SYSTEM = """You are an expert academic writer producing a publication-grade literature review.

=== JOURNAL GUIDELINES ===

Structure (follow this exactly):
  1. Abstract (150–250 words): scope, key themes, major findings, gaps.
  2. Introduction: motivate the review, define the field scope, state what the reader
     will gain, and outline the structure.
  3. Background: foundational concepts and terminology a reader needs before the review.
  4. Thematic Sections (3–6 sections, each with a descriptive heading): Each section
     covers a coherent sub-topic. Within each section synthesise across papers — compare
     approaches, note agreements and contradictions, identify trends.
  5. Discussion: cross-cutting insights, open controversies, limitations of current work.
  6. Future Directions: concrete open problems and promising research lines.
  7. Conclusion: 1–2 paragraphs summarising the state of the field.
  8. References: formatted according to the citation style specified below.

Writing standards (non-negotiable):
  - EVERY factual claim must be backed by one or more citations from the provided papers.
  - Synthesise ideas — do not just describe papers one by one. Compare, contrast, critique.
  - Use precise academic language appropriate for a graduate-level audience.
  - Highlight consensus, disagreements, and open questions.
  - Be critical: acknowledge limitations of existing work and methodological weaknesses.
  - Use sub-headings freely to organise each section.
  - Prefer active voice where natural; passive voice is acceptable for methods descriptions.

Citation format: {citation_format}

{citation_format_details}

=== CITATION PLACEHOLDERS ===
When citing a paper in the body text, use the placeholder format [REF:paper_id] where
paper_id is given in each paper note below. The system will replace these with proper
formatted citations and build the reference list automatically. Do NOT write references
inline; the reference list will be appended after generation.

Target length: {target_length} words (body only, excluding reference list).
"""

# ── Per-paper note prompt ─────────────────────────────────────────────────────

_NOTE_SYSTEM = """You are a research assistant helping to prepare structured notes for a
literature review. Distil each paper into a concise but information-rich note."""

_NOTE_USER = """
Review topic: {field_description}

Paper details:
  Title:    {title}
  Authors:  {authors}
  Year:     {year}
  Venue:    {venue}
  Abstract: {abstract}

Write a structured note (max 200 words) with these four labelled sections:
  CONTRIBUTION: What is the key contribution or finding?
  METHOD: What approach or methodology is used?
  RELEVANCE: Why is this paper important for the review topic?
  LIMITATIONS: What are the key limitations or open questions left by this paper?

Be specific and technical. Avoid generic phrases like "the authors propose".
"""

_REVIEW_USER = """
Research field / scope of the review:
\"\"\"
{field_description}
\"\"\"

Below are structured notes for {n_papers} papers to be included in the review.
Each note is prefixed with its citation ID in the form [paper_id].

{paper_notes}

Write the full literature review now. Remember to use [REF:paper_id] placeholders
for all citations (e.g. [REF:abc123de] or [REF:abc123de][REF:def456gh] for multiple).
"""


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _llm_call(
    system: str,
    user: str,
    provider: str,
    model_name: str,
    api_key: str,
    max_tokens: int = 4096,
) -> str:
    provider = provider.lower()

    if provider == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        resp = client.chat.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Per-paper note generation ─────────────────────────────────────────────────

def generate_paper_note(
    paper: LitReviewPaper,
    field_description: str,
    provider: str,
    model_name: str,
    api_key: str,
) -> str:
    """Generate a concise structured note for a single paper."""
    authors_list = paper.get_authors_list()
    authors_str = ", ".join(authors_list[:5])
    if len(authors_list) > 5:
        authors_str += " et al."

    user = _NOTE_USER.format(
        field_description=field_description,
        title=paper.title or "(no title)",
        authors=authors_str or "(unknown)",
        year=paper.year or "n.d.",
        venue=paper.venue or "unknown venue",
        abstract=paper.abstract or "(no abstract available)",
    )

    try:
        return _llm_call(
            system=_NOTE_SYSTEM,
            user=user,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            max_tokens=400,
        )
    except Exception as e:
        logger.warning(f"Note generation failed for '{paper.title[:50]}': {e}")
        # Fallback: use the key_contribution if available
        return paper.key_contribution or paper.abstract or "(no note available)"


# ── Citation formatting ───────────────────────────────────────────────────────

def _format_apa(paper: LitReviewPaper, index: int) -> str:
    authors = paper.get_authors_list()
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        parts = authors[0].split()
        author_str = f"{parts[-1]}, {'. '.join(p[0] for p in parts[:-1])}." if len(parts) > 1 else authors[0]
    else:
        formatted = []
        for a in authors:
            parts = a.split()
            if len(parts) > 1:
                formatted.append(f"{parts[-1]}, {'. '.join(p[0] for p in parts[:-1])}.")
            else:
                formatted.append(a)
        author_str = ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    year = paper.year or "n.d."
    title = paper.title or "Untitled"
    venue = f"*{paper.venue}*" if paper.venue else "Preprint"
    doi_part = f" https://doi.org/{paper.doi}" if paper.doi else (
        f" https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else ""
    )
    return f"{author_str} ({year}). {title}. {venue}.{doi_part}"


def _format_ieee(paper: LitReviewPaper, index: int) -> str:
    authors = paper.get_authors_list()
    if not authors:
        author_str = "Unknown Author"
    else:
        formatted = []
        for a in authors[:6]:
            parts = a.split()
            if len(parts) > 1:
                formatted.append(f"{''.join(p[0] + '.' for p in parts[:-1])} {parts[-1]}")
            else:
                formatted.append(a)
        author_str = ", ".join(formatted)
        if len(authors) > 6:
            author_str += " et al."

    year = paper.year or "n.d."
    title = paper.title or "Untitled"
    venue = paper.venue or "Preprint"
    doi_part = f", doi: {paper.doi}" if paper.doi else ""
    return f"[{index}] {author_str}, \"{title},\" *{venue}*, {year}{doi_part}."


def _format_nature(paper: LitReviewPaper, index: int) -> str:
    authors = paper.get_authors_list()
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) <= 6:
        formatted = []
        for a in authors:
            parts = a.split()
            if len(parts) > 1:
                formatted.append(f"{parts[-1]}, {''.join(p[0] + '.' for p in parts[:-1])}")
            else:
                formatted.append(a)
        author_str = ", ".join(formatted[:-1]) + " & " + formatted[-1] if len(formatted) > 1 else formatted[0]
    else:
        parts = authors[0].split()
        first = f"{parts[-1]}, {''.join(p[0] + '.' for p in parts[:-1])}" if len(parts) > 1 else authors[0]
        author_str = f"{first} et al."

    year = paper.year or "n.d."
    title = paper.title or "Untitled"
    venue = f"*{paper.venue}*" if paper.venue else "*Preprint*"
    loc = f" ({year})." if paper.venue else f" Preprint at https://arxiv.org/abs/{paper.arxiv_id} ({year})."
    return f"{index}. {author_str} {title}. {venue}{loc}"


_FORMATTERS = {
    "APA": _format_apa,
    "IEEE": _format_ieee,
    "Nature": _format_nature,
}


def build_reference_list(
    papers: list[LitReviewPaper], citation_format: str
) -> tuple[dict[str, int], str]:
    """
    Build a numbered reference list for the given papers.

    Returns:
        (id_to_number, formatted_reference_list_markdown)
    """
    fmt = _FORMATTERS.get(citation_format, _format_apa)
    id_to_num: dict[str, int] = {}
    lines: list[str] = []

    for i, paper in enumerate(papers, start=1):
        if paper.semantic_scholar_id:
            id_to_num[paper.semantic_scholar_id] = i
        lines.append(fmt(paper, i))

    ref_text = "\n\n".join(lines)
    return id_to_num, ref_text


def replace_placeholders(
    text: str,
    id_to_num: dict[str, int],
    citation_format: str,
    papers_by_ss_id: dict[str, LitReviewPaper],
) -> str:
    """
    Replace [REF:paper_id] placeholders in the review text with proper citations.
    """
    import re

    def replacer(m: re.Match) -> str:
        pid = m.group(1)
        num = id_to_num.get(pid)
        paper = papers_by_ss_id.get(pid)
        if num is None or paper is None:
            return f"[?]"

        if citation_format == "APA":
            return f"({paper.short_citation()})"
        elif citation_format == "IEEE":
            return f"[{num}]"
        else:  # Nature
            return f"^{num}"

    return re.sub(r"\[REF:([^\]]+)\]", replacer, text)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_literature_review(
    session: LitReviewSession,
    papers: list[LitReviewPaper],
    research_provider: str,
    research_model: str,
    research_api_key: str,
    writing_provider: str,
    writing_model: str,
    writing_api_key: str,
    target_words: int = 4000,
    progress_callback=None,
) -> str:
    """
    Generate a complete literature review for the given papers.

    Steps:
      1. Generate per-paper structured notes (using the research model).
      2. Call the writing model with all notes + guidelines.
      3. Replace citation placeholders with formatted references.
      4. Append the reference list.

    Returns the full review as a Markdown string.
    """
    if not papers:
        return "No relevant papers found to generate a literature review."

    citation_format = session.citation_format
    format_details = CITATION_FORMAT_DETAILS.get(citation_format, CITATION_FORMAT_DETAILS["APA"])

    # ── Pass 1: per-paper notes ───────────────────────────────────────────────
    logger.info(f"Generating notes for {len(papers)} papers…")
    notes_by_id: dict[str, str] = {}

    for i, paper in enumerate(papers):
        pid = paper.semantic_scholar_id or str(paper.id)
        if progress_callback:
            progress_callback(f"Generating note {i+1}/{len(papers)}: {paper.title[:50]}…")

        note = generate_paper_note(
            paper=paper,
            field_description=session.field_description,
            provider=research_provider,
            model_name=research_model,
            api_key=research_api_key,
        )
        notes_by_id[pid] = note
        logger.debug(f"Note generated for '{paper.title[:50]}'")

    # Assemble the combined notes block
    note_blocks: list[str] = []
    for paper in papers:
        pid = paper.semantic_scholar_id or str(paper.id)
        authors_list = paper.get_authors_list()
        author_str = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            author_str += " et al."
        header = f"[{pid}] {author_str} ({paper.year or 'n.d.'}) — {paper.title}"
        note_blocks.append(f"{header}\n{notes_by_id.get(pid, '(no note)')}")

    notes_text = "\n\n---\n\n".join(note_blocks)

    # ── Pass 2: full review ───────────────────────────────────────────────────
    logger.info(
        f"Generating review with {writing_provider}/{writing_model} "
        f"({len(papers)} papers, target {target_words} words)…"
    )
    if progress_callback:
        progress_callback(f"Writing literature review with {writing_model}…")

    system = _REVIEW_SYSTEM.format(
        citation_format=citation_format,
        citation_format_details=format_details,
        target_length=target_words,
    )
    user = _REVIEW_USER.format(
        field_description=session.field_description,
        n_papers=len(papers),
        paper_notes=notes_text,
    )

    # Writing models may produce long output; use a generous token budget.
    max_tokens = min(16000, max(4096, target_words * 2))

    review_body = _llm_call(
        system=system,
        user=user,
        provider=writing_provider,
        model_name=writing_model,
        api_key=writing_api_key,
        max_tokens=max_tokens,
    )

    # ── Pass 3: resolve citations + append reference list ─────────────────────
    id_to_num, ref_list_text = build_reference_list(papers, citation_format)
    papers_by_ss_id = {
        p.semantic_scholar_id: p for p in papers if p.semantic_scholar_id
    }

    review_body = replace_placeholders(
        review_body, id_to_num, citation_format, papers_by_ss_id
    )

    divider = "\n\n---\n\n"
    if citation_format == "IEEE":
        ref_header = "## References\n\n"
    else:
        ref_header = "## References\n\n"

    full_review = review_body + divider + ref_header + ref_list_text

    return full_review
