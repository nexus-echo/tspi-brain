"""Application settings, loaded from environment / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # App
    tspi_env: str = "local"
    tspi_log_level: str = "INFO"

    # Data
    database_url: str | None = None
    vector_backend: str = "pgvector"

    # Embeddings
    embedding_backend: str = "hash"      # hash (no deps) | api (OpenAI-compatible) | ollama (local)
    embedding_dim: int = 384             # MUST match the chosen model's output dimension
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_api_send_dimensions: bool = True  # set False for providers that don't accept the dimensions param (e.g. OpenRouter)
    ollama_embedding_model: str = "bge-m3"   # local Ollama embed model (multilingual)

    # LLM
    llm_primary: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    # --- Lab/imaging extraction (LOCAL-ONLY by governance decision) ---
    # Reads uploaded reports (images, scanned PDFs) with a local Ollama VISION model, and
    # structures text-PDF content with a local Ollama TEXT model. Nothing leaves the server.
    extraction_backend: str = "ollama"          # ollama | heuristic (regex-only, no model)
    vision_model: str = "qwen2.5vl"             # Ollama vision model (alt: llama3.2-vision, minicpm-v)
    extraction_text_model: str = ""             # blank -> reuse ollama_model for text structuring
    extraction_timeout_s: float = 120.0
    extraction_max_pages: int = 8               # cap pages rasterized from a scanned PDF
    pdf_text_min_chars: int = 40                # below this, treat a PDF page as scanned -> vision

    # Safety / governance
    require_doctor_validation: bool = True
    enforce_deidentification: bool = True

    # --- Phase B: auth / RBAC / audit ---
    # OFF by default so the local pilot + MiHealth keep working unchanged. Turn on per environment.
    auth_enabled: bool = False
    # Comma-separated service bearer tokens the trusted callers (MCP / MiHealth) present.
    # The caller then forwards the end-user identity via X-TSPI-* headers.
    service_tokens: str = ""

    # --- Production pilot ---
    # When true, PROVISIONAL/candidate clinical datasets are usable for ranking, but every report is
    # watermarked PROVISIONAL and must not auto-reach a patient. Physician approval still required.
    pilot_mode: bool = False
    # Default language for generated reports (per-request override allowed): "en" | "th".
    report_language_default: str = "en"

    # --- PII / PHI (P4) ---
    # Fernet key for encrypting stored patient identity (PHI) at rest. REQUIRED in any environment
    # that receives PII. Unset -> dev fallback stores plaintext (flagged); never do that in prod.
    encryption_key: str | None = None


settings = Settings()
