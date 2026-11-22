"""Функції іспиту, які викликає агент. Замість бази даних JSON-файли в папці data/."""

import json
import random
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TOPICS = json.loads((DATA_DIR / "topics.json").read_text(encoding="utf-8"))
STUDENTS = json.loads((DATA_DIR / "students.json").read_text(encoding="utf-8"))


class ExamError(Exception):
    """Помилка, яку агент отримує як результат інструмента і пояснює студенту."""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def message(role: str, content: str) -> dict:
    """Повідомлення історії у форматі з методички: role, content, datetime."""
    return {"role": role, "content": content, "datetime": now()}


def name_words(name: str) -> set[str]:
    """Слова імені без регістру: «Коваль  Олена» і «олена коваль» дають ту саму множину."""
    name = name.lower().replace("’", "'").replace("ʼ", "'")
    return set(re.findall(r"[\w']+", name))


class Exam:
    """Стан одного іспиту (одна сесія чату). Методи мають сигнатури з методички."""

    def __init__(self, results_dir: Path | None = None, seed: int | None = None):
        self.results_dir = Path(results_dir or DATA_DIR / "results")
        self.rng = random.Random(seed)
        self.student: dict | None = None
        self.topics: list[str] = []       # усі теми іспиту
        self.remaining: list[str] = []    # ще не видані агенту
        self.current: str | None = None
        self.result: dict | None = None

    # --- інструменти з методички ------------------------------------------------------------
    def start_exam(self, email: str, name: str) -> list[str]:
        """Перевіряє студента, записує початок іспиту і повертає 2–3 випадкові теми."""
        if self.student:
            raise ExamError(f"Іспит уже розпочато для {self.student['email']}.")
        email = (email or "").strip().lower()
        student = next((s for s in STUDENTS if s["email"].lower() == email), None)
        if student is None:
            raise ExamError(f"Студента з email {email or '(порожній)'} немає в списку. Попросіть студента "
                            "перевірити email і, коли він його напише, знову викличте start_exam.")
        # Ім'я й прізвище з бази мають бути серед слів name: «Добрий день, Олена Коваль» теж підходить
        if not name_words(student["name"]) <= name_words(name or ""):
            raise ExamError("Ім'я не збігається з іменем, записаним для цього email. Попросіть студента "
                            "написати ім'я та прізвище повністю і знову викличте start_exam.")
        self.student = student
        self.topics = self.rng.sample([t["title"] for t in TOPICS], k=self.rng.randint(2, 3))
        self.remaining = list(self.topics)
        self._append("exams.jsonl", {"event": "start", "email": student["email"], "name": student["name"],
                                     "topics": self.topics, "datetime": now()})
        return list(self.topics)

    def get_next_topic(self) -> str:
        """Повертає наступну тему; порожній рядок, коли теми закінчилися."""
        if not self.student:
            raise ExamError("Іспит ще не розпочато: спершу start_exam.")
        self.current = self.remaining.pop(0) if self.remaining else None
        return self.current or ""

    def end_exam(self, email: str, score: float, history: list[dict]) -> None:
        """Записує результат іспиту: оцінку 0..10 і повну історію чату."""
        if not self.student:
            raise ExamError("Іспит ще не розпочато.")
        if self.result:
            raise ExamError("Іспит уже завершено.")
        if (email or "").strip().lower() != self.student["email"].lower():
            raise ExamError(f"Email не збігається зі студентом іспиту ({self.student['email']}).")
        if self.remaining:
            raise ExamError(f"Ще не всі теми опитано: лишилось {len(self.remaining)}. "
                            "Викличте get_next_topic і продовжуйте іспит.")
        score = float(score)
        if not 0 <= score <= 10:
            raise ExamError("Оцінка має бути в межах 0..10.")
        self.result = {"email": self.student["email"], "name": self.student["name"],
                       "topics": self.topics, "score": score, "finished": now(), "history": list(history)}
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.results_dir / f"{stamp}_{self.student['email']}.json"
        path.write_text(json.dumps(self.result, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append("exams.jsonl", {"event": "end", "email": self.student["email"], "score": score,
                                     "file": path.name, "datetime": now()})

    # --- додатковий інструмент -------------------------------------------------------------
    def get_topic_rubric(self, topic: str) -> list[str]:
        """Ключові пункти, які має розкрити повна відповідь з теми (для питань і оцінювання)."""
        if topic not in self.topics:
            raise ExamError(f"Теми «{topic}» немає серед тем цього іспиту: {self.topics}.")
        return next(t["key_points"] for t in TOPICS if t["title"] == topic)

    @property
    def finished(self) -> bool:
        return self.result is not None

    def _append(self, filename: str, record: dict) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
