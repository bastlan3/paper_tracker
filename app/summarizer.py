"""
LLM-based paper summarization and keyword extraction.
Default: Mistral (free Experiment plan, Apache 2.0 models).
Also supports Anthropic and OpenAI as fallbacks.
"""

import json
import logging
from sqlmodel import Session

from app.config import settings
from app.database import Paper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant specialized in energy-based models (EBMs)
and related machine learning research. You help a graduate student in cognitive science
and machine learning stay on top of new papers."""

SUMMARIZE_PROMPT = """Analyze this paper and provide:
1. A 2-3 sentence summary that captures the key contribution and method.
   Skip obvious filler — focus on what's actually new. Write for someone who already
   knows EBMs, score matching, contrastive divergence, etc.
2. 3-6 keyword tags for quick triage. Use specific terms (not generic ones like "machine learning").
   Good tags: "composable-energy", "langevin-sampling", "image-generation", "planning", "continual-learning"

Paper title: {title}
Authors: {authors}
Abstract: {abstract}

Respond in JSON format only, no markdown wrapping:
{{"summary": "...", "keywords": ["tag1", "tag2", ...]}}"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _summarize_mistral(title: str, authors: str, abstract: str) -> dict:
    from mistralai import Mistral

    client = Mistral(api_key=settings.MISTRAL_API_KEY)

    response = client.chat.complete(
        model="mistral-small-latest",  # Free, fast, good enough for summaries
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARIZE_PROMPT.format(
                    title=title, authors=authors, abstract=abstract
                ),
            },
        ],
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def _summarize_anthropic(title: str, authors: str, abstract: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": SUMMARIZE_PROMPT.format(
                title=title, authors=authors, abstract=abstract
            ),
        }],
    )

    return _extract_json(response.content[0].text)


def _summarize_openai(title: str, authors: str, abstract: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARIZE_PROMPT.format(
                    title=title, authors=authors, abstract=abstract
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


PROVIDERS = {
    "mistral": _summarize_mistral,
    "anthropic": _summarize_anthropic,
    "openai": _summarize_openai,
}


def summarize_paper(title: str, authors: str, abstract: str) -> dict:
    """Generate summary + keywords for a paper. Returns {"summary": ..., "keywords": [...]}."""
    provider = settings.LLM_PROVIDER.lower()

    summarize_fn = PROVIDERS.get(provider)
    if not summarize_fn:
        raise ValueError(f"Unknown LLM provider: {provider}. Use: {list(PROVIDERS.keys())}")

    try:
        return summarize_fn(title, authors, abstract)
    except Exception as e:
        logger.error(f"Summarization failed for '{title[:60]}': {e}")
        # Fallback: use first 2 sentences of abstract as summary
        sentences = abstract.split(". ")
        fallback_summary = ". ".join(sentences[:2]) + "."
        return {"summary": fallback_summary, "keywords": []}


def enrich_papers(session: Session, papers: list[Paper]) -> list[Paper]:
    """Add summaries and keywords to papers that don't have them yet."""
    enriched = []

    for paper in papers:
        if paper.summary:
            continue

        logger.info(f"Summarizing: {paper.title[:60]}...")
        result = summarize_paper(
            paper.title,
            ", ".join(paper.get_authors_list()[:5]),
            paper.abstract,
        )

        paper.summary = result.get("summary", "")
        paper.keywords = json.dumps(result.get("keywords", []))
        session.add(paper)
        enriched.append(paper)

    if enriched:
        session.commit()

    logger.info(f"Enriched {len(enriched)} papers with summaries")
    return enriched
