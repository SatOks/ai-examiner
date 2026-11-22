"""Вебінтерфейс ШІ-екзаменатора на Streamlit. Запуск: streamlit run streamlit_app.py"""

import json
import os

import streamlit as st
from groq import Groq

from agent import GREETING, MODELS, AgentError, ExaminerAgent
from exam_tools import STUDENTS

REPO_URL = "https://github.com/SatOks/ai-examiner"
TOOL_ICONS = {"start_exam": "📝", "get_next_topic": "📚", "end_exam": "🏁"}


def md(text: str) -> str:
    """Знак $ у Streamlit вмикає LaTeX; екрануємо, щоб формули й ціни показувались як текст."""
    return text.replace("$", "\\$")


def short(value, limit=90) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def show_tool(event: dict) -> None:
    icon = "⚠️" if "error" in event["result"] else TOOL_ICONS.get(event["name"], "🔧")
    with st.expander(f"{icon} `{event['name']}({short(event['arguments'], 60)[1:-1]})`"):
        st.caption("Аргументи")
        st.json(event["arguments"])
        st.caption("Результат")
        st.json(event["result"])


def show_event(event: dict, show_tools: bool) -> None:
    if event["type"] == "tool":
        if show_tools:
            show_tool(event)
    elif event.get("final"):
        with st.container(border=True):
            st.markdown(f"**Результат іспиту: {event['score']:g} / 10**")
            st.markdown(md(event["content"]))
    else:
        st.markdown(md(event["content"]))


def reset() -> None:
    for key in ("agent", "chat"):
        st.session_state.pop(key, None)


st.set_page_config(page_title="ШІ-екзаменатор з NLP", page_icon="🎓")
st.session_state.setdefault("chat", [{"role": "assistant", "events": [{"type": "assistant", "content": GREETING}]}])

# --- бічна панель: ключ, стан іспиту ------------------------------------------------------------
with st.sidebar:
    st.header("Налаштування")
    api_key = st.text_input("API-ключ Groq", type="password", placeholder="gsk_...",
                            help="Безкоштовно на console.groq.com/keys. Ключ ніде не зберігається.")
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")   # змінна середовища лише для локального запуску
    show_tools = st.toggle("Показувати виклики інструментів", value=True)

    st.header("Стан іспиту")
    agent: ExaminerAgent | None = st.session_state.get("agent")
    exam = agent.exam if agent else None
    if not exam or not exam.student:
        st.caption("Іспит ще не розпочато.")
    else:
        st.markdown(f"**{exam.student['name']}**  \n{exam.student['email']}")
        for topic in exam.topics:
            if topic in exam.remaining:
                mark = "⬜"
            elif topic == exam.current and not exam.finished:
                mark = "▶️"
            else:
                mark = "✅"
            st.markdown(f"{mark} {topic}")
        if exam.finished:
            st.metric("Оцінка", f"{exam.result['score']:g} / 10")
            st.download_button("Завантажити результат (JSON)", file_name="exam_result.json",
                               mime="application/json", width="stretch",
                               data=json.dumps(exam.result, ensure_ascii=False, indent=2))
    st.button("Новий іспит", on_click=reset, width="stretch")

# --- головна сторінка -------------------------------------------------------------------------
st.title("🎓 ШІ-екзаменатор з NLP")
st.markdown(
    f"Чат-бот проводить усний іспит з курсу «Основи великих мовних моделей». Це LLM-агент: мовна модель "
    f"`{next(iter(MODELS))}` (Groq API) сама вирішує, коли викликати інструменти "
    f"`start_exam` (перевірити студента і отримати 2–3 теми), `get_next_topic` (перейти до наступної теми) "
    f"і `end_exam` (виставити оцінку 0–10 і записати історію іспиту).\n\n"
    f"Введіть API-ключ Groq у бічній панелі (безкоштовно на "
    f"[console.groq.com](https://console.groq.com/keys)) і представтеся в чаті."
)
with st.expander("Тестові студенти"):
    st.caption("Іспит можуть складати лише студенти зі списку:")
    st.markdown("\n".join(f"- {s['name']}, `{s['email']}`" for s in STUDENTS))

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        for event in item["events"]:
            show_event(event, show_tools)

finished = bool(exam and exam.finished)
prompt = st.chat_input("Іспит завершено. Натисніть «Новий іспит», щоб почати знову." if finished
                       else "Ваша відповідь", disabled=finished)
if prompt:
    with st.chat_message("user"):
        st.markdown(md(prompt))
    if not api_key.strip():
        st.error("Введіть API-ключ Groq у бічній панелі.")
        st.stop()
    if agent is None:
        agent = st.session_state.agent = ExaminerAgent(api_key)
    elif agent.client.api_key != api_key.strip():
        agent.client = Groq(api_key=api_key.strip(), max_retries=0)

    with st.chat_message("assistant"):
        waiting = st.empty()
        agent.on_wait = lambda s: waiting.info(f"Ліміт запитів Groq: чекаю {s:.0f} с…", icon="⏳")
        try:
            with st.spinner("Екзаменатор думає…"):
                events = agent.respond(prompt)
        except AgentError as e:
            waiting.empty()
            st.error(str(e))
            st.stop()
        waiting.empty()
        for event in events:
            show_event(event, show_tools)
    st.session_state.chat += [{"role": "user", "events": [{"type": "assistant", "content": prompt}]},
                              {"role": "assistant", "events": events}]
    st.rerun()   # оновити бічну панель (теми, оцінку)

st.divider()
st.caption(f"Лабораторна робота 9, «Основи великих мовних моделей». Код: [{REPO_URL[8:]}]({REPO_URL})")
