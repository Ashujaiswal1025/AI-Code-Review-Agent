from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Code Review Agent"

    # Optional
    GITHUB_TOKEN: str = ""

    # Storage
    CHROMA_DB_DIR: str = "./chroma_db"
    REPO_TEMP_DIR: str = "./repos"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()