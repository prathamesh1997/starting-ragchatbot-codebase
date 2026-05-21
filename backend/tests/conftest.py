import sys
import os

# Put backend/ on sys.path so all backend modules are importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore


SAMPLE_COURSE_DOC = """Course Title: Python Fundamentals
Course Link: https://example.com/python
Course Instructor: Jane Doe

Lesson 1: Introduction to Python
Lesson Link: https://example.com/python/lesson1
Python is a high-level interpreted programming language created by Guido van Rossum in 1991. Python emphasizes code readability and simplicity above all else.

Lesson 2: Variables and Data Types
Lesson Link: https://example.com/python/lesson2
In Python variables are created when you assign a value to them using the equals sign. Python supports integers, floats, strings, and booleans as core data types.
"""

SAMPLE_COURSE_DOC_2 = """Course Title: Machine Learning Basics
Course Link: https://example.com/ml
Course Instructor: John Smith

Lesson 1: What is Machine Learning
Lesson Link: https://example.com/ml/lesson1
Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed.

Lesson 2: Supervised Learning
Lesson Link: https://example.com/ml/lesson2
Supervised learning involves training a model on labeled data where the correct output is already known for each training example.
"""


@pytest.fixture(scope="session")
def embedding_model_name():
    return "all-MiniLM-L6-v2"


@pytest.fixture
def populated_store(tmp_path, embedding_model_name):
    """VectorStore pre-loaded with two courses (Python + ML)."""
    store = VectorStore(str(tmp_path / "chroma"), embedding_model_name, max_results=5)

    course1 = Course(
        title="Python Fundamentals",
        course_link="https://example.com/python",
        instructor="Jane Doe",
        lessons=[
            Lesson(lesson_number=1, title="Introduction to Python"),
            Lesson(lesson_number=2, title="Variables and Data Types"),
        ],
    )
    store.add_course_metadata(course1)
    store.add_course_content([
        CourseChunk(
            content="Lesson 1 content: Python is a high-level interpreted programming language created by Guido van Rossum.",
            course_title="Python Fundamentals",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Python emphasizes code readability and simplicity above all else.",
            course_title="Python Fundamentals",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Lesson 2 content: Variables in Python are created when you assign a value using the equals sign.",
            course_title="Python Fundamentals",
            lesson_number=2,
            chunk_index=2,
        ),
        CourseChunk(
            content="Python supports integers, floats, strings, and booleans as core data types.",
            course_title="Python Fundamentals",
            lesson_number=2,
            chunk_index=3,
        ),
    ])

    course2 = Course(
        title="Machine Learning Basics",
        course_link="https://example.com/ml",
        instructor="John Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is Machine Learning"),
            Lesson(lesson_number=2, title="Supervised Learning"),
        ],
    )
    store.add_course_metadata(course2)
    store.add_course_content([
        CourseChunk(
            content="Lesson 1 content: Machine learning enables computers to learn patterns from data without explicit programming.",
            course_title="Machine Learning Basics",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Lesson 2 content: Supervised learning trains a model on labeled data where the correct output is known.",
            course_title="Machine Learning Basics",
            lesson_number=2,
            chunk_index=1,
        ),
    ])

    return store


@pytest.fixture
def empty_store(tmp_path, embedding_model_name):
    """VectorStore with no documents."""
    return VectorStore(str(tmp_path / "chroma"), embedding_model_name, max_results=5)
