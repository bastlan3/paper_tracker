"""
Relevance agent: uses an LLM to decide whether a cited paper belongs
in the target literature review.

Supports Mistral (free), Anthropic, and OpenAI.
"""

import json
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a research librarian helping to build a systematic literature review.
Your task is to evaluate whether individual papers should be included in a review on a
specific research topic. Be thorough but strict: only recommend papers that genuinely
contribute to understanding the topic."""

_USER = """
Research topic / field for the literature review:
\"\"\"
{field_description}
\"\"\"

Evaluate the following paper for inclusion:

Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Abstract: {abstract}

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "is_relevant": true,
  "relevance_score": 0.85,
  "reason": "One or two sentences explaining why this paper is or is not relevant.",
  "key_contribution": "One sentence capturing the paper's main contribution (fill only when relevant)."
}}

Scoring guide:
  0.9–1.0  Core paper — directly addresses the topic
  0.7–0.89 Highly relevant — closely related methods or results
  0.5–0.69 Moderately relevant — tangentially useful background
  0.0–0.49 Not relevant — set is_relevant to false
"""


# ── Provider implementations ──────────────────────────────────────────────────

def _call_mistral(
    field_description: str,
    title: str,
    authors: str,
    year: str,
    venue: str,
    abstract: str,
    model_name: str,
    api_key: str,
) -> dict:
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    field_description=field_description,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    abstract=abstract,
                ),
            },
        ],
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _call_anthropic(
    field_description: str,
    title: str,
    authors: str,
    year: str,
    venue: str,
    abstract: str,
    model_name: str,
    api_key: str,
) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model_name,
        max_tokens=300,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": _USER.format(
                field_description=field_description,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                abstract=abstract,
            ),
        }],
    )
    text = resp.content[0].text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _call_openai(
    field_description: str,
    title: str,
    authors: str,
    year: str,
    venue: str,
    abstract: str,
    model_name: str,
    api_key: str,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        max_tokens=300,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    field_description=field_description,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    abstract=abstract,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


_PROVIDERS = {
    "mistral": _call_mistral,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}


# ── Public interface ──────────────────────────────────────────────────────────

def check_relevance(
    paper_info: dict,
    field_description: str,
    model_provider: str,
    model_name: str,
    api_key: str,
) -> Tuple[bool, float, str, str]:
    """
    Ask the LLM whether a paper is relevant to the given field.

    Returns:
        (is_relevant, relevance_score, reason, key_contribution)
    """
    title = (paper_info.get("title") or "").strip()
    abstract = (paper_info.get("abstract") or "").strip()

    if not title:
        return False, 0.0, "No title available.", ""

    authors_list = paper_info.get("authors") or []
    if isinstance(authors_list, list):
        authors_str = ", ".join(authors_list[:5])
        if len(authors_list) > 5:
            authors_str += " et al."
    else:
        authors_str = str(authors_list)

    year = str(paper_info.get("year") or "n.d.")
    venue = paper_info.get("venue") or "unknown venue"

    fn = _PROVIDERS.get(model_provider.lower())
    if not fn:
        raise ValueError(
            f"Unknown model provider '{model_provider}'. "
            f"Choose from: {list(_PROVIDERS)}"
        )

    try:
        result = fn(
            field_description=field_description,
            title=title,
            authors=authors_str,
            year=year,
            venue=venue,
            abstract=abstract or "(no abstract available)",
            model_name=model_name,
            api_key=api_key,
        )
        is_relevant = bool(result.get("is_relevant", False))
        score = float(result.get("relevance_score", 0.0))
        reason = str(result.get("reason", ""))
        key_contribution = str(result.get("key_contribution", ""))
        return is_relevant, score, reason, key_contribution

    except Exception as e:
        logger.error(
            f"Relevance check failed for '{title[:60]}' "
            f"[{model_provider}/{model_name}]: {e}"
        )
        # Conservative fallback: uncertain — skip
        return False, 0.0, f"LLM error: {e}", ""


def get_api_key(provider: str) -> str:
    """Retrieve the API key for the given provider from settings."""
    from app.config import settings

    mapping = {
        "mistral": settings.MISTRAL_API_KEY,
        "anthropic": settings.ANTHROPIC_API_KEY,
        "openai": settings.OPENAI_API_KEY,
    }
    key = mapping.get(provider.lower(), "")
    if not key:
        raise ValueError(
            f"No API key configured for provider '{provider}'. "
            f"Set {provider.upper()}_API_KEY in your .env file."
        )
    return key


# ── Available model catalogue ─────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "research": {
        "free": [
            {"provider": "mistral", "name": "open-mistral-nemo",     "label": "Mistral Nemo (free)"},
            {"provider": "mistral", "name": "mistral-small-latest",  "label": "Mistral Small (free)"},
        ],
        "paid": [
            {"provider": "anthropic", "name": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
            {"provider": "anthropic", "name": "claude-sonnet-4-6",          "label": "Claude Sonnet 4.6"},
            {"provider": "openai",    "name": "gpt-4o-mini",                "label": "GPT-4o mini"},
            {"provider": "openai",    "name": "gpt-4o",                     "label": "GPT-4o"},
        ],
    },
    "writing": {
        "free": [
            {"provider": "mistral", "name": "mistral-large-latest",  "label": "Mistral Large (free tier)"},
        ],
        "paid": [
            {"provider": "anthropic", "name": "claude-sonnet-4-6",   "label": "Claude Sonnet 4.6"},
            {"provider": "anthropic", "name": "claude-opus-4-6",     "label": "Claude Opus 4.6"},
            {"provider": "openai",    "name": "gpt-4o",              "label": "GPT-4o"},
        ],
    },
}
