"""Application settings, read from environment variables (prefix ``FETALAI_``) or a ``.env`` file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FETALAI_", env_file=".env", extra="ignore")

    app_name: str = "FetalAI API"
    version: str = "1.0.0"
    checkpoint_path: Path = REPO_ROOT / "outputs" / "hc18_unet_best.pth"
    device: str | None = Field(default=None, description="'cuda', 'cpu' or None for automatic")
    max_upload_mb: float = 10.0
    min_image_side: int = 64
    max_image_pixels: int = 20_000_000
    allowed_content_types: tuple[str, ...] = ("image/png", "image/jpeg")
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
