import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from pydantic_ai import Agent
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Label, Markdown, TextArea

load_dotenv()

HIST_PATH = "friday_history.json"
MODEL = "openrouter:openai/gpt-5.1-codex-mini"

agent = Agent(
    MODEL,
    system_prompt="Você é o Friday, um assistente de linha de comando. Responda de forma concisa e técnica.",
)


class ChatMessage(Markdown):
    def __init__(self, who: str, text: str, dt: str | None = None, **kwargs) -> None:
        label = "👤 Your" if who == "user" else "🤖 Friday"
        when = dt or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        super().__init__(f"{text}", **kwargs)
        self.border_title = Text(label, style="white on green")
        self.border_subtitle = Text(when, style="white on gray")


class ChatView(VerticalScroll):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._messages = []

    async def add_message(self, who: str, text: str, dt: str | None = None) -> None:
        entry = {
            "who": who,
            "text": text,
            "dt": dt or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._messages.append(entry)
        await self.mount(ChatMessage(**entry))
        self.save_history()
        self.call_after_refresh(self.scroll_to_latest)

    def save_history(self) -> None:
        with open(HIST_PATH, "w", encoding="utf-8") as file:
            json.dump(self._messages, file, ensure_ascii=False, indent=2)

    async def load_history(self) -> None:
        if os.path.exists(HIST_PATH):
            with open(HIST_PATH, "r", encoding="utf-8") as file:
                raw_messages = json.load(file)
        else:
            raw_messages = []
        for entry in raw_messages:
            await self.add_message(**entry)

    async def clear_history(self) -> None:
        self._messages.clear()
        if os.path.exists(HIST_PATH):
            os.remove(HIST_PATH)
        for child in list(self.children):
            await child.remove()
        self.call_after_refresh(self.scroll_to_latest)

    def scroll_to_latest(self) -> None:
        self.scroll_end(
            animate=True,
            x_axis=False,
            y_axis=True,
            immediate=True,
            force=True,
        )

    async def on_mount(self) -> None:
        await self.load_history()
        self.scroll_to_latest()


class ChatInput(TextArea):
    """Input multi-linha que envia com Enter (ou outra tecla)."""

    BINDINGS = [
        Binding("ctrl+enter", "submit", "Submit"),
        Binding("ctrl+j", "submit", "Submit", show=False),
    ]

    @dataclass
    class Submitted(Message, bubble=True):
        chat_input: "ChatInput"
        text: str

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self, self.text.strip()))


class Chat(Widget):
    def compose(self) -> ComposeResult:
        yield ChatView()
        yield Label(id="thinking")
        yield ChatInput(
            placeholder="Type your question...",
        )


class FridayChatApp(App):
    CSS_PATH = "style.tcss"
    INLINE_PADDING = 0
    BINDINGS = [
        Binding("ctrl+k", "clear_history", "Clear History"),
        Binding("ctrl+i", "toggle_input_mode", "Toggle Input Mode"),
        Binding("ctrl+l", "toggle_llm", "Toggle LLM Model"),
    ]

    def compose(self) -> ComposeResult:
        yield Chat(id="chat")
        yield Footer()

    def on_mount(self) -> None:
        self._chat = self.query_one(Chat)
        self._chat_view = self._chat.query_one(ChatView)
        self._chat_input = self._chat.query_one(ChatInput)
        self._loading = self._chat.query_one(Label)
        self._loading.content = MODEL
        self.set_focus(self._chat_input)
        self._chat_view.scroll_to_latest()

    @on(ChatInput.Submitted)
    async def handle_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()
        self._chat_input.text = ""
        if not text:
            self.notify("Message cannot be empty", title="Error", severity="error")
            return
        await self._chat_view.add_message("user", text)
        self._loading.loading = True
        try:
            reply = await asyncio.to_thread(agent.run_sync, text)
            output = reply.output
        except Exception as exc:  # pylint: disable=broad-except
            output = f"Erro ao chamar o agente: {exc}"
        finally:
            self._loading.loading = False
        await self._chat_view.add_message("agent", output)

    async def action_clear_history(self) -> None:
        if hasattr(self, "_chat_view"):
            await self._chat_view.clear_history()


if __name__ == "__main__":
    FridayChatApp().run(inline=True)
