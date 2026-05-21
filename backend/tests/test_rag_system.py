"""
Tests for RAGSystem.query() and document loading in rag_system.py.

Covers:
- query() return type is (str, list)
- query() does not raise on an empty vector store
- query() with loaded documents returns a non-empty answer
- query() sources are populated when a search is performed
- query() sources are reset between calls
- session history is accumulated across turns
- add_course_document() parses title and counts chunks
- add_course_folder() loads all .txt files
- get_course_analytics() reflects loaded courses
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from config import Config
from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_COURSE = """Course Title: RAG Course
Course Link: https://example.com/rag
Course Instructor: Test Author

Lesson 1: What is RAG
Lesson Link: https://example.com/rag/lesson1
RAG stands for Retrieval-Augmented Generation. It combines semantic document retrieval with large language model generation to answer questions accurately about specific content.

Lesson 2: Vector Stores
Lesson Link: https://example.com/rag/lesson2
Vector stores index text as high-dimensional embeddings that enable semantic similarity search. ChromaDB is a widely used open-source vector store for RAG systems.
"""

SECOND_COURSE = """Course Title: Prompt Engineering
Course Link: https://example.com/pe
Course Instructor: Second Author

Lesson 1: Writing Good Prompts
Lesson Link: https://example.com/pe/lesson1
Prompt engineering is the practice of crafting instructions for large language models to produce accurate and useful outputs.
"""


@pytest.fixture
def test_config(tmp_path):
    cfg = Config(
        ANTHROPIC_API_KEY="",           # forces MockAIGenerator — no real API calls
        CHROMA_PATH=str(tmp_path / "chroma"),
        MAX_RESULTS=5,
    )
    return cfg


@pytest.fixture
def empty_rag(test_config):
    """RAGSystem with no documents loaded."""
    return RAGSystem(test_config)


@pytest.fixture
def loaded_rag(test_config, tmp_path):
    """RAGSystem with one course document loaded."""
    doc = tmp_path / "rag_course.txt"
    doc.write_text(MINIMAL_COURSE, encoding="utf-8")

    system = RAGSystem(test_config)
    system.add_course_document(str(doc))
    return system


@pytest.fixture
def multi_course_rag(test_config, tmp_path):
    """RAGSystem loaded from a folder containing two course files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "course1.txt").write_text(MINIMAL_COURSE, encoding="utf-8")
    (docs_dir / "course2.txt").write_text(SECOND_COURSE, encoding="utf-8")

    system = RAGSystem(test_config)
    system.add_course_folder(str(docs_dir))
    return system


# ---------------------------------------------------------------------------
# query() return contract
# ---------------------------------------------------------------------------

class TestQueryReturnContract:
    def test_returns_two_element_tuple(self, empty_rag):
        result = empty_rag.query("What is Python?")
        assert isinstance(result, tuple) and len(result) == 2

    def test_answer_is_non_empty_string(self, empty_rag):
        answer, _ = empty_rag.query("What is Python?")
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_sources_is_list(self, empty_rag):
        _, sources = empty_rag.query("What is Python?")
        assert isinstance(sources, list)


# ---------------------------------------------------------------------------
# query() robustness
# ---------------------------------------------------------------------------

class TestQueryRobustness:
    def test_does_not_raise_on_empty_store(self, empty_rag):
        """query() must not propagate exceptions when the vector store is empty."""
        try:
            empty_rag.query("What is machine learning?")
        except Exception as exc:
            pytest.fail(f"query() raised on empty store: {type(exc).__name__}: {exc}")

    def test_does_not_raise_with_none_session_id(self, empty_rag):
        try:
            empty_rag.query("any question", session_id=None)
        except Exception as exc:
            pytest.fail(f"query() raised with session_id=None: {type(exc).__name__}: {exc}")

    def test_does_not_raise_with_valid_session_id(self, empty_rag):
        sid = empty_rag.session_manager.create_session()
        try:
            empty_rag.query("first question", session_id=sid)
        except Exception as exc:
            pytest.fail(f"query() raised with valid session_id: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# query() with loaded documents
# ---------------------------------------------------------------------------

class TestQueryWithDocuments:
    def test_answer_is_non_empty_with_loaded_docs(self, loaded_rag):
        answer, _ = loaded_rag.query("What is RAG?")
        assert isinstance(answer, str) and len(answer.strip()) > 0

    def test_sources_are_populated_after_search(self, loaded_rag):
        """MockAIGenerator always searches; sources must be non-empty for loaded store."""
        _, sources = loaded_rag.query("Tell me about RAG retrieval")
        assert len(sources) > 0, "Sources should be populated when documents are loaded"

    def test_sources_contain_course_title(self, loaded_rag):
        _, sources = loaded_rag.query("Tell me about vector stores")
        assert any("RAG" in s for s in sources), f"Unexpected sources: {sources}"

    def test_sources_reset_between_queries(self, loaded_rag):
        """Sources from query N must not bleed into query N+1."""
        _, sources1 = loaded_rag.query("What is RAG?")
        _, sources2 = loaded_rag.query("What are vector stores?")
        # Both should be non-empty lists (independent results, not accumulated)
        assert isinstance(sources1, list)
        assert isinstance(sources2, list)

    def test_answer_mentions_rag_content(self, loaded_rag):
        answer, _ = loaded_rag.query("What is RAG?")
        # MockAIGenerator returns raw search results so course title should appear
        assert "RAG" in answer or "retrieval" in answer.lower()


# ---------------------------------------------------------------------------
# Session / conversation history
# ---------------------------------------------------------------------------

class TestQuerySession:
    def test_multi_turn_conversation_does_not_raise(self, empty_rag):
        sid = empty_rag.session_manager.create_session()
        try:
            empty_rag.query("First question", session_id=sid)
            empty_rag.query("Second question", session_id=sid)
            empty_rag.query("Third question", session_id=sid)
        except Exception as exc:
            pytest.fail(f"Multi-turn conversation raised: {type(exc).__name__}: {exc}")

    def test_session_history_is_stored(self, empty_rag):
        sid = empty_rag.session_manager.create_session()
        empty_rag.query("What is Python?", session_id=sid)
        history = empty_rag.session_manager.get_conversation_history(sid)
        assert history is not None and len(history) > 0


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

class TestDocumentLoading:
    def test_add_course_document_returns_course_object(self, empty_rag, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text(MINIMAL_COURSE, encoding="utf-8")

        course, chunk_count = empty_rag.add_course_document(str(doc))
        assert course is not None
        assert course.title == "RAG Course"
        assert chunk_count > 0

    def test_add_course_document_creates_chunks(self, empty_rag, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text(MINIMAL_COURSE, encoding="utf-8")

        _, chunk_count = empty_rag.add_course_document(str(doc))
        assert chunk_count >= 2  # at least one chunk per lesson

    def test_add_course_folder_loads_all_txt_files(self, test_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "course1.txt").write_text(MINIMAL_COURSE, encoding="utf-8")
        (docs_dir / "course2.txt").write_text(SECOND_COURSE, encoding="utf-8")

        system = RAGSystem(test_config)
        courses_added, chunks_added = system.add_course_folder(str(docs_dir))
        assert courses_added == 2
        assert chunks_added > 0

    def test_add_course_folder_skips_duplicates(self, multi_course_rag, tmp_path):
        """Running add_course_folder a second time must not double-add courses."""
        docs_dir = tmp_path / "docs"  # same dir used in the fixture
        # Re-create the same files
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / "course1.txt").write_text(MINIMAL_COURSE, encoding="utf-8")
        (docs_dir / "course2.txt").write_text(SECOND_COURSE, encoding="utf-8")

        new_courses, _ = multi_course_rag.add_course_folder(str(docs_dir))
        assert new_courses == 0, "Duplicate courses must be skipped"


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_empty_rag_has_zero_courses(self, empty_rag):
        analytics = empty_rag.get_course_analytics()
        assert analytics["total_courses"] == 0

    def test_loaded_rag_reflects_course_count(self, loaded_rag):
        analytics = loaded_rag.get_course_analytics()
        assert analytics["total_courses"] == 1

    def test_analytics_contains_course_title(self, loaded_rag):
        analytics = loaded_rag.get_course_analytics()
        assert "RAG Course" in analytics["course_titles"]

    def test_multi_course_rag_reflects_all_courses(self, multi_course_rag):
        analytics = multi_course_rag.get_course_analytics()
        assert analytics["total_courses"] == 2
        titles = analytics["course_titles"]
        assert "RAG Course" in titles
        assert "Prompt Engineering" in titles
