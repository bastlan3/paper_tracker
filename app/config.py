import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings:
    # LLM — default to Mistral (free Experiment plan)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mistral")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Email
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    EMAIL_TO: str = os.getenv("EMAIL_TO", "")

    # App
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    DAILY_DIGEST_HOUR: int = int(os.getenv("DAILY_DIGEST_HOUR", "9"))
    WEEKLY_UMAP_DAY: str = os.getenv("WEEKLY_UMAP_DAY", "monday")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'ebm_tracker.db'}")

    # Arxiv search config
    ARXIV_CATEGORIES: list[str] = ["cs.LG", "cs.AI", "stat.ML"]
    ARXIV_KEYWORDS: list[str] = [
        "energy-based model",
        "energy based model",
        "EBM",
        "energy function",
        "composable energy",
        "contrastive divergence",
        "Langevin dynamics",
        "score matching",
        "implicit generation",
        "energy landscape",
    ]

    # Seed papers (arxiv IDs) — Yilun Du style EBM work
    SEED_PAPER_IDS: list[str] = [
        "1903.08689",  # Implicit Generation and Modeling with Energy-Based Models
        "2004.06030",  # Compositional Visual Generation with Energy Based Models
        "2012.01316",  # Energy-Based Models for Continual Learning
        "2101.03288",  # Improved Contrastive Divergence Training of Energy-Based Models
        "2206.11763",  # Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models
        "2209.09874",  # Learning Iterative Reasoning through Energy Minimization
        "2305.05252",  # Learning Universal Policies via Text-Guided Video Generation
        "1609.03126",  # A Tutorial on Energy-Based Learning
        "2107.00517",  # GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
        "2010.02502",  # Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One
    ]


settings = Settings()
