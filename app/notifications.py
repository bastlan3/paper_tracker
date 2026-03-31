"""
Notification system: Telegram bot + email digest.
Each paper gets: summary, keywords, arxiv link, PDF link.
Optionally sends an audio summary via Mistral Voxtral TTS.
"""

import asyncio
import base64
import logging
import tempfile
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import aiosmtplib
from jinja2 import Template
from sqlmodel import Session

from app.config import settings
from app.database import Paper, get_unnotified_papers

logger = logging.getLogger(__name__)

TELEGRAM_MAX_CHUNK = 4000
TELEGRAM_RATE_LIMIT = 0.5  # seconds between messages
MAX_AUTHORS_DISPLAY = 3


def _format_authors(authors: list[str]) -> str:
    s = ", ".join(authors[:MAX_AUTHORS_DISPLAY])
    if len(authors) > MAX_AUTHORS_DISPLAY:
        s += f" +{len(authors) - MAX_AUTHORS_DISPLAY} more"
    return s


def _paper_summary(paper: Paper) -> str:
    return paper.summary or paper.abstract[:300] or "No summary available"


# ── Telegram ──────────────────────────────────────────────────────────────────

TELEGRAM_MSG_TEMPLATE = """📄 {title}
by {authors}

{summary}

🏷 {keywords}
📊 Similarity: {score:.0%}

🔗 Paper: {url}
📥 PDF: {pdf_url}
"""


def _build_tts_script(paper: Paper) -> str:
    """Build a concise spoken-word script for a single paper."""
    authors = paper.get_authors_list()
    first_author = authors[0].split()[-1] if authors else "Unknown"
    et_al = " and colleagues" if len(authors) > 1 else ""
    summary = paper.summary or paper.abstract[:300] or "No summary available."
    return (
        f"{paper.title}, by {first_author}{et_al}. "
        f"{summary}"
    )


def generate_paper_audio(script: str, voice_id: str | None = None) -> bytes:
    """
    Synthesise speech for a paper script using Mistral Voxtral TTS.
    Returns raw MP3 bytes, or raises on failure.
    """
    from mistralai import Mistral

    client = Mistral(api_key=settings.MISTRAL_API_KEY)
    response = client.audio.speech.complete(
        model="voxtral-mini-tts-2603",
        input=script,
        voice_id=voice_id or settings.VOXTRAL_VOICE_ID,
        response_format="mp3",
    )
    return base64.b64decode(response.audio_data)


async def send_telegram_digest(papers: list[Paper]):
    """Send daily digest via Telegram using a single Bot instance."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping")
        return

    if not papers:
        logger.info("No new papers for Telegram digest")
        return

    from telegram import Bot

    async with Bot(token=settings.TELEGRAM_BOT_TOKEN) as bot:
        header = f"📬 EBM Paper Digest — {datetime.utcnow().strftime('%Y-%m-%d')}\n{len(papers)} new paper(s)"
        await bot.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=header, disable_web_page_preview=True)

        for paper in papers:
            msg = format_telegram_paper(paper)
            for chunk in [msg[i:i + TELEGRAM_MAX_CHUNK] for i in range(0, len(msg), TELEGRAM_MAX_CHUNK)]:
                await bot.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=chunk, disable_web_page_preview=True)

            if settings.TELEGRAM_AUDIO and settings.MISTRAL_API_KEY:
                try:
                    script = _build_tts_script(paper)
                    audio_bytes = await asyncio.get_event_loop().run_in_executor(
                        None, generate_paper_audio, script, settings.VOXTRAL_VOICE_ID
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    await bot.send_voice(
                        chat_id=settings.TELEGRAM_CHAT_ID,
                        voice=Path(tmp_path).open("rb"),
                    )
                    Path(tmp_path).unlink(missing_ok=True)
                    logger.info(f"Audio summary sent for: {paper.title[:50]}")
                except Exception as e:
                    logger.warning(f"Voxtral TTS failed for '{paper.title[:50]}': {e}")

            await asyncio.sleep(TELEGRAM_RATE_LIMIT)


def format_telegram_paper(paper: Paper) -> str:
    keywords = paper.get_keywords_list()
    kw_str = " ".join(f"#{kw.replace('-', '_').replace(' ', '_')}" for kw in keywords) if keywords else ""

    return TELEGRAM_MSG_TEMPLATE.format(
        title=paper.title,
        authors=_format_authors(paper.get_authors_list()),
        summary=_paper_summary(paper),
        keywords=kw_str,
        score=paper.similarity_score or 0,
        url=paper.url,
        pdf_url=paper.pdf_url,
    )


# ── Email ─────────────────────────────────────────────────────────────────────

EMAIL_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 700px; margin: 0 auto; padding: 20px; color: #1a1a1a; }
  .header { border-bottom: 2px solid #2563eb; padding-bottom: 12px; margin-bottom: 24px; }
  .header h1 { font-size: 20px; color: #2563eb; margin: 0; }
  .header p { color: #666; margin: 4px 0 0 0; font-size: 14px; }
  .paper { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
           padding: 16px; margin-bottom: 16px; }
  .paper h2 { font-size: 16px; margin: 0 0 6px 0; color: #1e293b; }
  .paper .authors { font-size: 13px; color: #64748b; margin-bottom: 8px; }
  .paper .summary { font-size: 14px; line-height: 1.5; margin-bottom: 10px; }
  .paper .keywords { margin-bottom: 10px; }
  .paper .keywords span { display: inline-block; background: #dbeafe; color: #1e40af;
                           font-size: 12px; padding: 2px 8px; border-radius: 12px;
                           margin: 2px 4px 2px 0; }
  .paper .meta { font-size: 12px; color: #94a3b8; }
  .paper .links { margin-top: 10px; }
  .paper .links a { color: #2563eb; text-decoration: none; font-size: 13px;
                     margin-right: 16px; }
  .paper .links a:hover { text-decoration: underline; }
  .score-bar { display: inline-block; width: 60px; height: 6px; background: #e2e8f0;
               border-radius: 3px; vertical-align: middle; margin-left: 6px; }
  .score-fill { height: 100%; background: #2563eb; border-radius: 3px; }
  .footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid #e2e8f0;
            font-size: 12px; color: #94a3b8; }
</style>
</head>
<body>
  <div class="header">
    <h1>EBM Paper Digest</h1>
    <p>{{ date }} — {{ papers | length }} new paper{{ 's' if papers | length != 1 else '' }}</p>
  </div>

  {% for paper in papers %}
  <div class="paper">
    <h2>{{ paper.title }}</h2>
    <div class="authors">{{ paper.author_str }}</div>
    <div class="summary">{{ paper.summary }}</div>
    <div class="keywords">
      {% for kw in paper.keywords %}
      <span>{{ kw }}</span>
      {% endfor %}
    </div>
    <div class="meta">
      Similarity: {{ "%.0f" | format(paper.similarity_score * 100) }}%
      <span class="score-bar">
        <span class="score-fill" style="width: {{ (paper.similarity_score * 100) | int }}%"></span>
      </span>
    </div>
    <div class="links">
      <a href="{{ paper.url }}">📄 Paper</a>
      <a href="{{ paper.pdf_url }}">📥 PDF</a>
    </div>
  </div>
  {% endfor %}

  <div class="footer">
    EBM Paper Tracker — automated digest
  </div>
</body>
</html>
""")


async def send_email_digest(papers: list[Paper]):
    """Send daily digest via email."""
    if not settings.SMTP_USER or not settings.EMAIL_TO:
        logger.warning("Email not configured, skipping")
        return

    if not papers:
        logger.info("No new papers for email digest")
        return

    # Prepare template data
    paper_data = []
    for p in papers:
        paper_data.append({
            "title": p.title,
            "author_str": _format_authors(p.get_authors_list()),
            "summary": _paper_summary(p),
            "keywords": p.get_keywords_list(),
            "similarity_score": p.similarity_score or 0,
            "url": p.url,
            "pdf_url": p.pdf_url,
        })

    html = EMAIL_TEMPLATE.render(
        papers=paper_data,
        date=datetime.utcnow().strftime("%B %d, %Y"),
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"EBM Paper Digest — {len(papers)} new paper{'s' if len(papers) != 1 else ''}"
    msg["From"] = settings.SMTP_USER
    msg["To"] = settings.EMAIL_TO
    msg.attach(MIMEText(html, "html"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            start_tls=True,
        )
        logger.info(f"Email digest sent to {settings.EMAIL_TO}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise


# ── Combined digest ──────────────────────────────────────────────────────────

async def send_daily_digest(session: Session, papers: list[Paper] | None = None):
    """
    Send notifications for papers.
    If papers is provided, notify those specific papers.
    If papers is None (scheduled job), notify all unnotified papers in DB.
    """
    if papers is None:
        papers = get_unnotified_papers(session)

    # Filter to only unnotified ones (in case caller passes already-notified papers)
    papers = [p for p in papers if not p.is_notified]

    if not papers:
        logger.info("No papers to notify, skipping digest")
        return

    logger.info(f"Sending digest for {len(papers)} papers")

    try:
        await send_telegram_digest(papers)
        for paper in papers:
            paper.is_notified = True
            session.add(paper)
        session.commit()
        logger.info("Digest sent and papers marked as notified")
    except Exception as e:
        logger.error(f"Telegram digest failed: {e}")