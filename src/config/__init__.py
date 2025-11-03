"""
Configuration constants for the project.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

config_dir = Path(__file__).parent
src_dir = config_dir.parent
project_dir = src_dir.parent
pth_env = project_dir / ".env"
# The path to the public env file (for cloud environments)
pth_env_default = config_dir / ".env.public"


class Settings(BaseSettings):
    """Application settings with environment variables support."""

    model_config = SettingsConfigDict(
        env_file=pth_env,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# === Load environment variables === #

settings = Settings()

# Set env variable (for development)
if pth_env.exists():
    load_dotenv(pth_env)
else:
    load_dotenv(pth_env_default)
