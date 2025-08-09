import logging
import os
from typing import List, Optional

from instaloader import Profile, InstaloaderException
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Annoy
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from operator import itemgetter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from app.component.instagram_component import InstagramComponent
from app.config import Settings
from app.llm_factory import LLMFactory, get_llm_factory

logger = logging.getLogger(__name__)

COMMENT_PROMPT_TEMPLATE = """
당신은 소셜 미디어 댓글 작성 전문가입니다.
아래는 특정 인스타그램 게시물에 달린 실제 댓글들입니다. 이 댓글들의 스타일, 어조, 단어 사용법을 깊이 분석하되, 결과물은 반드시 다음 `지시사항`을 따라야 합니다.

[지시사항]
- 주어진 '댓글을 달 내용'에 대해 전혀 다른 새로운 댓글을 사람이 직점 작성한 것처럼 생성하세요.
- `참고할 댓글들`과 비슷한 어조, 말투, 단어의 한국어로 50자 이내로 완전히 전혀 다른 {amount}개의 댓글을 작성해주세요.
- 감탄사는 필수로 사용하지 않아도 됩니다.
- 만약 감탄사를 사용한다면 `와!`와 같은 인위적인 감탄사보다는 `오!?`, `헐` `!?!?` 등과 같은 실제 사람이 사용하는 감탄사를 사용하세요.
- 다음 JSON 스키마에 따라 결과물을 생성해주세요.
{format_instructions}

[참고할 댓글들]
{context}

[댓글을 달 내용]
{input}

[새로운 댓글 (JSON)]
"""

RE_COMMENT_PROMPT_TEMPLATE = """
당신은 소셜 미디어 댓글 작성 전문가입니다.
아래는 특정 인스타그램 게시물에 달린 실제 댓글들입니다. 이 댓글들의 스타일, 어조, 단어 사용법을 깊이 분석하되, 결과물은 반드시 다음 `지시사항`을 따라야 합니다.

[지시사항]
- 주어진 '댓글을 달 내용'에 대해서 대댓글을 사람이 직점 작성한 것처럼 생성하세요.
- `참고할 댓글들`과 비슷한 어조, 말투, 단어의 한국어로 50자 이내로 완전히 전혀 다른 {amount}개의 댓글을 작성해주세요.
- 감탄사는 필수로 사용하지 않아도 됩니다.
- 만약 감탄사를 사용한다면 `와!`와 같은 인위적인 감탄사보다는 `오!?`, `헐` `!?!?` 등과 같은 실제 사람이 사용하는 감탄사를 사용하세요.
- 다음 JSON 스키마에 따라 결과물을 생성해주세요.
{format_instructions}

[참고할 댓글들]
{context}

[댓글을 달 내용]
{input}

[새로운 댓글 (JSON)]
"""


class CommentList(BaseModel):
    comments: List[str] = Field(description="생성된 댓글 목록")


class RAGService:
    def __init__(
        self,
        llm_factory: LLMFactory,
        instagram_component: Optional[InstagramComponent] = None,
    ):
        self.llm_factory = llm_factory
        self.vector_store: Optional[Annoy] = None
        self.comment_retrieval_chain = None
        self.re_comment_retrieval_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.instagram = instagram_component
        self.parser = JsonOutputParser(pydantic_object=CommentList)
        self.comment_prompt = ChatPromptTemplate.from_template(
            COMMENT_PROMPT_TEMPLATE,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.re_comment_prompt = ChatPromptTemplate.from_template(
            RE_COMMENT_PROMPT_TEMPLATE,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def initialize(self, settings: Settings):
        """
        LangChain RAG 파이프라인 초기화
        """
        try:
            self._setup_vector_store(settings)
            self._setup_retrieval_chain()
        except Exception as e:
            logger.error(f"LangChain RAG 파이프라인 초기화 실패: {e}", exc_info=True)
            raise

    def _setup_vector_store(self, settings: Settings):
        """
        벡터 저장소 초기화
        """
        if os.path.exists(settings.VECTOR_STORE_PATH):
            logger.info("벡터 저장소를 로드합니다.")
            self.vector_store = Annoy.load_local(
                settings.VECTOR_STORE_PATH,
                self.llm_factory.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("벡터 저장소를 성공적으로 로드했습니다.")
        else:
            comments = self._load_comments(settings)
            if not comments:
                logger.warning("댓글을 찾을 수 없어 벡터 저장소를 생성하지 않습니다.")
                return

            logger.info("텍스트를 분할합니다.")
            texts = self.text_splitter.split_documents(comments)
            logger.info(f"텍스트를 {len(texts)}개의 청크로 분할했습니다.")

            logger.info("벡터 저장소를 생성합니다.")
            self.vector_store = Annoy.from_documents(
                texts, self.llm_factory.embeddings
            )

            self.vector_store.save_local(settings.VECTOR_STORE_PATH)
            logger.info("벡터 저장소를 성공적으로 생성했습니다.")

    def _load_comments(self, settings: Settings) -> List[Document]:
        """
        인스타그램 댓글 로드
        """
        if not self.instagram:
            logger.error("InstagramComponent가 초기화되지 않았습니다.")
            return []

        try:
            logger.info(f"'{settings.TARGET_INSTAGRAM_USERNAME}' 프로필 정보를 가져옵니다.")
            profile = Profile.from_username(
                self.instagram.L.context, settings.TARGET_INSTAGRAM_USERNAME
            )

            posts = profile.get_posts()
            comments = []

            logger.info(f"'{settings.TARGET_INSTAGRAM_USERNAME}'의 최근 게시물 15개에서 댓글을 수집합니다.")
            for i, post in enumerate(posts):
                if i >= 15:
                    break
                logger.info(f"게시물 {i+1}에서 댓글을 가져옵니다: {post.url}")
                for comment in post.get_comments():
                    comments.append(Document(page_content=comment.text))

            logger.info(f"총 {len(comments)}개의 댓글을 수집했습니다.")
            return comments

        except InstaloaderException as e:
            logger.error(f"프로필을 가져오는 중 오류 발생: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"댓글을 로드하는 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return []
        
    def _setup_retrieval_chain(self):
        """
        Retriever 및 체인 설정
        """
        if not self.vector_store:
            logger.warning("벡터 저장소가 준비되지 않았습니다. Retriever 및 체인을 설정할 수 없습니다.")
            return

        logger.info("Retriever 및 체인을 설정합니다.")
        retriever = self.vector_store.as_retriever()

        document_chain = create_stuff_documents_chain(self.llm_factory.llm, self.comment_prompt)
        self.comment_retrieval_chain = create_retrieval_chain(retriever, document_chain)
        document_chain = create_stuff_documents_chain(self.llm_factory.llm, self.re_comment_prompt)
        self.re_comment_retrieval_chain = create_retrieval_chain(retriever, document_chain)
        logger.info("Retriever 및 체인을 성공적으로 설정했습니다.")

    def comment_ask(self, text: str, amount: int) -> List[str]:
        """
        댓글 생성
        """
        if not self.comment_retrieval_chain:
            logger.error("RAG 파이프라인이 준비되지 않아서 질문에 답변할 수 없습니다.")
            raise RuntimeError("RAG 파이프라인이 초기화되지 않았습니다.")

        logger.info(f"질문 처리 시작: {text}")
        chain_with_parser = self.comment_retrieval_chain | itemgetter("answer") | self.parser
        response = chain_with_parser.invoke(
            {"input": text, "amount": amount}
        )

        logger.info("댓글을 성공적으로 생성했습니다.")
        return response["comments"]
    
    def re_comment_ask(self, text: str, amount: int) -> List[str]:
        """
        대댓글 생성
        """
        if not self.re_comment_retrieval_chain:
            logger.error("RAG 파이프라인이 준비되지 않아서 질문에 답변할 수 없습니다.")
            raise RuntimeError("RAG 파이프라인이 초기화되지 않았습니다.")

        logger.info(f"질문 처리 시작: {text}")
        chain_with_parser = self.re_comment_retrieval_chain | itemgetter("answer") | self.parser
        response = chain_with_parser.invoke(
            {"input": text, "amount": amount}
        )

        logger.info("대댓글을 성공적으로 생성했습니다.")
        return response["comments"]


def create_rag_service(settings: Settings) -> RAGService:
    """
    RAGService 인스턴스를 생성하고 초기화
    """
    instagram_component = InstagramComponent(username=settings.MY_INSTAGRAM_USERNAME)
    instagram_component.login()

    llm_factory = get_llm_factory()

    service = RAGService(llm_factory=llm_factory, instagram_component=instagram_component)
    return service
