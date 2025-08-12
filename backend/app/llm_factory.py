import logging
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from .config import get_settings, Settings

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    설정에 따른 LLM 및 임베딩 모델 인스턴스를 생성하는 팩토리 클래스
    """
    def __init__(self, settings: Settings):
        provider = settings.LLM_PROVIDER.lower()
        logger.info(f"LLM Provider를 초기화합니다: {provider}")
        if provider == "google":
            self._initialize_google(settings)
        elif provider == "ollama":
            self._initialize_ollama(settings)
        elif provider == "openai":
            self._initialize_openai(settings)
        else:
            raise ValueError(f"알 수 없는 LLM Provider: {settings.LLM_PROVIDER}")

    def _initialize_google(self, settings: Settings):
        self.embeddings: Embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        )
        self.llm: BaseChatModel = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            convert_system_message_to_human=True
        )

    def _initialize_ollama(self, settings: Settings):
        self.embeddings: Embeddings = OllamaEmbeddings(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_HOST
        )
        self.llm: BaseChatModel = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

    def _initialize_openai(self, settings: Settings):
        self.embeddings: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.llm: BaseChatModel = ChatOpenAI(
            model_name=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

def get_llm_factory() -> LLMFactory:
    settings = get_settings()
    return LLMFactory(settings)
