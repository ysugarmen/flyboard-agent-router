from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = Field(default="flyboard_agent_router")
    APP_ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")

    # Kb
    KB_PATH: str = Field(default="kb.json")
    KB_TOP_K_DEFAULT: int = Field(default=5)

    # Agent Router
    AGENT_MAX_SECONDS: int = Field(default=60)
    AGENT_TRACE_LOGS: bool = Field(default=False)
    MAX_TOOLS_ITERATIONS: int = Field(default=6)

    # External keys
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4.1-mini")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
