"""Вопрос в формате платформы: разбор диалога, состояние вопроса и поисковые запросы из него."""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import CHECK_FIXES


TURN_RE = re.compile(r"^\s*(Пользователь|Ассистент|Клиент|Юрист|Сотрудник|User|Assistant|Бот|Система)\s*:\s*",
                     re.I | re.M)
ASSISTANT_ROLES = {"ассистент", "assistant", "бот", "система"}


def parse_dialog(q: str) -> tuple[str, list[tuple[str, str]]]:
    """Возвращает (текущий вопрос, история [(роль, текст)])."""
    marks = list(TURN_RE.finditer(q))
    if not marks or marks[0].start() > 0 and q[: marks[0].start()].strip():
        if len(marks) < 2:
            return q.strip(), []
    turns = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(q)
        role = "assistant" if m.group(1).lower() in ASSISTANT_ROLES else "user"
        turns.append((role, q[m.end():end].strip()))
    if len(turns) < 2:
        return (turns[0][1] if turns else q.strip()), []
    last_user = max((i for i, (r, _) in enumerate(turns) if r == "user"), default=-1)
    if last_user < 0:
        return q.strip(), []
    return turns[last_user][1], turns[:last_user]


@dataclass
class Prepared:
    current: str
    history: list
    qvec: list | None = None
    # R7: переформулирование (один раз на вопрос; финальный проход B8 берёт готовое)
    rewrite_done: bool = False
    standalone: str | None = None
    extra: list | None = None      # [(поисковый запрос, его эмбеддинг или None)]
    soft4xx_counted: bool = False  # B5: вопрос уже учтён в серии отказов (один раз на вопрос, не на каждую ступень)


def user_text(prep: Prepared) -> str:
    """Реплики пользователя: история + текущий вопрос (без реплик ассистента)."""
    return " ".join([t for r, t in prep.history if r == "user"] + [prep.current])


def event_text(qtext: str) -> str:
    """Текст, из которого берутся даты события: только реплики пользователя (пакет 2), без флага — весь вопрос."""
    return user_text(Prepared(*parse_dialog(qtext))) if CHECK_FIXES else qtext


def search_query(prep: Prepared, assistant: bool = False) -> str:
    """Поисковый запрос: текущий вопрос и реплики пользователя из истории; assistant=True — и реплики ассистента.
    Единственное место, где из вопроса собирается запрос (поиск, реранкер, эмбеддинг, запасной ответ)."""
    return prep.current + " " + " ".join(t for r, t in prep.history if assistant or r == "user")
