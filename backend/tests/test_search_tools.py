"""
Tests for CourseSearchTool.execute() in search_tools.py.

Covers:
- Basic query against a populated store
- Empty-store behaviour
- course_name filter
- lesson_number filter
- Combined course + lesson filter
- Unknown course name
- n_results > collection size (ChromaDB edge-case)
- Source tracking after search
- Filter operator syntax (chromadb 1.0 requires explicit $eq)
"""

import pytest
from models import Course, Lesson, CourseChunk
from search_tools import CourseSearchTool
from vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers / extra fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_tool(populated_store):
    return CourseSearchTool(populated_store)


@pytest.fixture
def empty_search_tool(empty_store):
    return CourseSearchTool(empty_store)


# ---------------------------------------------------------------------------
# Basic execute() behaviour
# ---------------------------------------------------------------------------

class TestExecuteBasic:
    def test_returns_non_empty_string_for_relevant_query(self, search_tool):
        result = search_tool.execute(query="Python programming language")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_bracket_headers(self, search_tool):
        """_format_results should wrap each hit in [Course - Lesson N] headers."""
        result = search_tool.execute(query="Python programming language")
        assert "[" in result and "]" in result

    def test_result_mentions_matched_course(self, search_tool):
        result = search_tool.execute(query="Python programming language")
        assert "Python" in result

    def test_ml_query_returns_ml_content(self, search_tool):
        result = search_tool.execute(query="machine learning artificial intelligence")
        assert "Machine Learning" in result or "machine learning" in result.lower()

    def test_empty_collection_returns_no_content_message(self, empty_search_tool):
        result = empty_search_tool.execute(query="Python")
        assert "No relevant content found" in result


# ---------------------------------------------------------------------------
# Filter behaviour
# ---------------------------------------------------------------------------

class TestExecuteFilters:
    def test_course_name_filter_restricts_results(self, search_tool):
        """Filtering by course name must not raise and must return a string."""
        result = search_tool.execute(query="learning", course_name="Python")
        assert isinstance(result, str), "execute() must return str, not raise"

    def test_lesson_number_filter_restricts_results(self, search_tool):
        """Filtering by lesson_number must not raise and must return a string."""
        result = search_tool.execute(query="variables data types", lesson_number=2)
        assert isinstance(result, str), "execute() must return str, not raise"

    def test_lesson_filter_result_references_correct_lesson(self, search_tool):
        result = search_tool.execute(query="variables data types", lesson_number=2)
        if "No relevant content found" not in result:
            assert "Lesson 2" in result

    def test_combined_course_and_lesson_filter(self, search_tool):
        """Both filters applied together must not raise."""
        result = search_tool.execute(
            query="variables",
            course_name="Python Fundamentals",
            lesson_number=2,
        )
        assert isinstance(result, str)

    def test_unknown_course_name_returns_descriptive_message(self, search_tool):
        result = search_tool.execute(query="anything", course_name="NonExistentCourseXYZ999")
        assert isinstance(result, str)
        assert "No course found" in result or "No relevant content" in result

    def test_no_filter_does_not_raise(self, search_tool):
        """Plain query with no optional filters must never raise."""
        try:
            result = search_tool.execute(query="introduction overview")
        except Exception as exc:
            pytest.fail(f"execute() raised {type(exc).__name__}: {exc}")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ChromaDB edge-cases
# ---------------------------------------------------------------------------

class TestExecuteEdgeCases:
    def test_n_results_exceeds_collection_size_does_not_raise(self, tmp_path, embedding_model_name):
        """
        ChromaDB used to raise when n_results > number of stored documents.
        VectorStore.search() must handle this gracefully (returns results or empty).
        """
        store = VectorStore(str(tmp_path / "chroma"), embedding_model_name, max_results=10)
        course = Course(
            title="Tiny Course",
            instructor="Alice",
            lessons=[Lesson(lesson_number=1, title="Only Lesson")],
        )
        store.add_course_metadata(course)
        store.add_course_content([
            CourseChunk(
                content="Just one single chunk of content here.",
                course_title="Tiny Course",
                lesson_number=1,
                chunk_index=0,
            )
        ])
        tool = CourseSearchTool(store)
        try:
            result = tool.execute(query="content")
        except Exception as exc:
            pytest.fail(
                f"execute() raised when n_results ({store.max_results}) > "
                f"collection size (1): {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, str)

    def test_none_lesson_number_in_metadata_does_not_raise_on_add(self, tmp_path, embedding_model_name):
        """
        ChromaDB 1.x rejects None metadata values.
        Adding a chunk with lesson_number=None must not silently corrupt the store.
        """
        store = VectorStore(str(tmp_path / "chroma"), embedding_model_name, max_results=5)
        course = Course(title="No-Lesson Course", instructor="Bob", lessons=[])
        store.add_course_metadata(course)
        try:
            store.add_course_content([
                CourseChunk(
                    content="Content with no lesson number.",
                    course_title="No-Lesson Course",
                    lesson_number=None,
                    chunk_index=0,
                )
            ])
        except Exception as exc:
            pytest.fail(
                f"add_course_content() raised with lesson_number=None: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_filter_operator_syntax_accepted_by_chromadb(self, search_tool):
        """
        ChromaDB 1.0 removed implicit equality (dict value shorthand).
        Applying a course_name filter must return results, not a ChromaDB error string.
        """
        result = search_tool.execute(query="python language", course_name="Python Fundamentals")
        # A ChromaDB filter-syntax error would surface as "Search error: ..." in the result
        assert not result.startswith("Search error:"), (
            f"ChromaDB rejected the filter syntax. Got: {result[:200]}"
        )


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------

class TestSourceTracking:
    def test_last_sources_populated_after_search(self, search_tool):
        search_tool.execute(query="Python programming")
        assert isinstance(search_tool.last_sources, list)
        assert len(search_tool.last_sources) > 0

    def test_last_sources_cleared_on_empty_result(self, empty_search_tool):
        empty_search_tool.execute(query="Python")
        # With no results, _format_results is never called, so last_sources stays []
        assert empty_search_tool.last_sources == []

    def test_sources_contain_course_title(self, search_tool):
        search_tool.execute(query="Python programming language")
        assert any("Python" in src for src in search_tool.last_sources)
