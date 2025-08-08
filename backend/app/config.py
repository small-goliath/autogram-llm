from functools import lru_cache
from typing import Annotated, Any, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    AnyUrl,
    BeforeValidator,
    computed_field
)

def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".config/.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    FRONTEND_HOST: str = "http://localhost:3000"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    @computed_field
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    PROJECT_NAME: str = "Autogram"
    LOGGER_CONFIG_PATH: str
    LLM_HOST: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "llama3.3"
    LLM_TEMPERATURE: float = 0.9
    LLM_PROVIDER: Literal["google", "ollama"] = "ollama"
    VECTOR_STORE_PATH: str = "annoy/annoy_index"
    TARGET_INSTAGRAM_USERNAME: str
    MY_INSTAGRAM_USERNAME: str

@lru_cache
def get_settings() -> Settings:
    return Settings()
