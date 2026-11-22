"""Перевірки функцій іспиту без LLM: python -m pytest -q"""

import json

import pytest

from exam_tools import TOPICS, Exam, ExamError

EMAIL, NAME = "olena.koval@lpnu.ua", "Олена Коваль"


@pytest.fixture
def exam(tmp_path):
    return Exam(results_dir=tmp_path, seed=1)


def test_unknown_student(exam):
    with pytest.raises(ExamError, match="немає в списку"):
        exam.start_exam("nobody@lpnu.ua", NAME)


def test_wrong_name(exam):
    with pytest.raises(ExamError, match="не збігається"):
        exam.start_exam(EMAIL, "Олена Петренко")


def test_name_order_and_case(exam):
    topics = exam.start_exam(" Olena.Koval@LPNU.ua ", "коваль олена")
    assert 2 <= len(topics) <= 3
    assert len(set(topics)) == len(topics)
    assert all(t in [x["title"] for x in TOPICS] for t in topics)


def test_topics_then_empty(exam):
    topics = exam.start_exam(EMAIL, NAME)
    assert [exam.get_next_topic() for _ in topics] == topics
    assert exam.get_next_topic() == ""


def test_end_exam_requires_all_topics(exam):
    exam.start_exam(EMAIL, NAME)
    exam.get_next_topic()
    with pytest.raises(ExamError, match="не всі теми"):
        exam.end_exam(EMAIL, 10, [])


def test_end_exam_writes_result(exam, tmp_path):
    topics = exam.start_exam(EMAIL, NAME)
    for _ in topics:
        exam.get_next_topic()
    history = [{"role": "user", "content": "привіт", "datetime": "2026-11-22T10:00:00"}]
    with pytest.raises(ExamError, match="0..10"):
        exam.end_exam(EMAIL, 11, history)
    with pytest.raises(ExamError, match="Email"):
        exam.end_exam("andrii.melnyk@lpnu.ua", 7, history)
    exam.end_exam(EMAIL, 7.5, history)
    saved = json.loads(next(tmp_path.glob("*_olena.koval@lpnu.ua.json")).read_text(encoding="utf-8"))
    assert saved["score"] == 7.5 and saved["history"] == history and saved["topics"] == topics
    events = [json.loads(line)["event"] for line in (tmp_path / "exams.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["start", "end"]
    with pytest.raises(ExamError, match="уже завершено"):
        exam.end_exam(EMAIL, 7.5, history)


def test_tools_before_start(exam):
    with pytest.raises(ExamError):
        exam.get_next_topic()
    with pytest.raises(ExamError):
        exam.end_exam(EMAIL, 5, [])


def test_rubric_only_for_exam_topics(exam):
    topics = exam.start_exam(EMAIL, NAME)
    assert exam.get_topic_rubric(topics[0])
    other = next(t["title"] for t in TOPICS if t["title"] not in topics)
    with pytest.raises(ExamError):
        exam.get_topic_rubric(other)
