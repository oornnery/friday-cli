import uuid
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    TabbedContent,
    TabPane,
    TextArea,
)


class Modal(Widget):
    pass


class AskModal(Modal):
    pass


def _generate_id():
    return str(uuid.uuid4())


@dataclass
class MessageModel:
    text: str
    who: str
    dt: str | None = None
    id: str = field(default_factory=_generate_id)

    def __post_init__(self):
        self.dt = self.dt or datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MessageBubble(Markdown):
    DEFAULT_CSS = """

    """

    def __init__(self, message: MessageModel):
        super().__init__(message.text)
        self.message = message
        self.border_title = Text(message.who, style="white on gray")
        self.border_subtitle = Text(message.dt, style="white on gray")


class ChatView(VerticalScroll):
    DEFAULT_CSS = """

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._messages = []

    def add_message(self, text: str, who: str, dt: str | None = None):
        message = MessageModel(text=text, who=who, dt=dt)
        self._messages.append(message)
        self.mount(MessageBubble(message))

    def save_history(self):
        pass

    def load_history(self):
        pass

    def clear_history(self):
        pass


class ChatInformation(Widget):
    DEFAULT_CSS = """

    """


class ChatInput(TextArea):
    DEFAULT_CSS = """
    """
    DEFAULT_PLACEHOLDER = "Type your message here"
    BINDINGS = [
        Binding("ctrl+enter", "submit", "Submit"),
        Binding("ctrl+j", "submit", "Submit", show=False),
    ]

    @dataclass
    class Submitted(Message, bubble=True):
        chat_input: "ChatInput"
        text: str = ""
        timestamp: datetime = field(default_factory=datetime.now)

    def action_submit(self):
        self.post_message(self.Submitted(self, text=self.text))


class ChatContainer(Container):
    DEFAULT_CSS = """
    """

    def compose(self) -> ComposeResult:
        yield ChatView()
        yield ChatInformation(Label("Model: openai:gpt-5.1-mini-codex"))
        yield ChatInput(placeholder=ChatInput.DEFAULT_PLACEHOLDER, compact=True)


class HistoryChatContainer(Widget):
    pass


class SettingsContainer(Widget):
    pass


class FridayApp(App):
    CSS = """
    """
    CSS_PATH = "styles.tcss"
    AUTO_FOCUS = "ChatInput"

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Chat"):
                yield ChatContainer()
            with TabPane("History"):
                yield HistoryChatContainer()
            with TabPane("Settings"):
                yield SettingsContainer()
        yield Footer()

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return
        chat_view = self.query_one(ChatView)
        chat_view.add_message(text, who="user")
        self.query_one(ChatInput).text = ""
        self.call_after_refresh(chat_view.scroll_end)
        self.run_agent_request(text)

    @work(exclusive=True)
    async def run_agent_request(self, prompt: str):
        pass


def main():
    app = FridayApp()
    app.run()


if __name__ == "__main__":
    main()
