from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_rag_service
from app.service import RAGService


@pytest.fixture
def mock_rag_service() -> MagicMock:
    """RAGService의 MagicMock을 생성하는 Fixture"""
    mock = MagicMock(spec=RAGService)
    # lifespan에서 호출되는 initialize()가 아무것도 하지 않도록 설정
    mock.initialize.return_value = None
    return mock


@pytest.fixture
def client(mock_rag_service: MagicMock) -> TestClient:
    """의존성이 재정의된 TestClient를 생성하는 Fixture"""
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    # TestClient가 app을 로드할 때 lifespan이 실행되므로,
    # RAGService 생성을 모킹하여 실제 서비스가 초기화되지 않도록 함
    with patch("app.main.create_rag_service", return_value=mock_rag_service), TestClient(
        app
    ) as c:
        yield c
    del app.dependency_overrides[get_rag_service]


def test_read_main(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Autogram LLM Backend is running."}


def test_ask_question_success(client: TestClient, mock_rag_service: MagicMock):
    """질문/답변 API 성공 케이스 테스트"""
    text = "모킹모킹"
    mock_rag_service.ask.return_value = "응답 모킹"
    response = client.post("/api/comments/query", json={"text": text})

    assert response.status_code == 200
    assert response.json() == {"answer": "응답 모킹"}
    mock_rag_service.ask.assert_called_once_with(text)


def test_ask_question_with_no_question_provided(client: TestClient):
    """질문이 없는 경우의 API 테스트"""
    response = client.post("/api/comments/query", json={"text": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "질문이 없습니다."}


def test_ask_question_when_rag_is_not_ready(
    client: TestClient, mock_rag_service: MagicMock
):
    """RAG 파이프라인이 준비되지 않았을 때"""
    text = "질문 실패"
    mock_rag_service.ask.side_effect = RuntimeError("RAG pipeline is not ready.")
    response = client.post("/api/comments/query", json={"text": text})

    assert response.status_code == 503
    assert response.json() == {"detail": "RAG pipeline is not ready."}
    mock_rag_service.ask.assert_called_once_with(text)


def test_ask_question_with_unexpected_error(
    client: TestClient, mock_rag_service: MagicMock
):
    """예상치 못한 오류 발생 시"""
    text = "생성 실패"
    mock_rag_service.ask.side_effect = Exception("An unexpected error occurred.")
    response = client.post("/api/comments/query", json={"text": text})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "답변 생성 중 오류 발생: An unexpected error occurred."
    }
    mock_rag_service.ask.assert_called_once_with(question)
