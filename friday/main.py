# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "httpx>=0.28.1",
#     "openai>=2.8.0",
#     "pydantic-ai>=1.18.0",
#     "python-dotenv>=1.2.1",
#     "rich>=14.2.0",
#     "textual>=6.6.0",
#     "textual-dev>=1.8.0",
#     "typer>=0.20.0",
# ]
# ///
# pylint: disable=too-many-instance-attributes,too-many-arguments
import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from dotenv import load_dotenv
from pydantic_ai import Agent
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Label, Markdown, OptionList, TextArea
from textual.widgets.option_list import Option

load_dotenv()
load_dotenv(os.path.expanduser("~/.env"))

FRIDAY_PATH = os.path.expanduser("~/.friday")
HIST_PATH = os.path.join(FRIDAY_PATH, "history.json")
LOG_DIR = os.path.join(FRIDAY_PATH, ".logs")
LOG_FILE = os.path.join(LOG_DIR, "friday.log")
EXEC_CMD_PATH = os.path.join(FRIDAY_PATH, "exec_cmd")

os.makedirs(FRIDAY_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
LOGGER = logging.getLogger("friday")

MODEL = "openrouter:openai/gpt-5.1-codex-mini"

agent = Agent(
    MODEL,
    deps_type=str,
    system_prompt="Você é o Friday, um assistente de linha de comando. Responda de forma concisa e técnica.",
)

TOOL_NAMES = [
    "list_directory",
    "read_file",
    "write_file",
    "run_command",
    "suggest_command",
]


@agent.tool_plain
def list_directory(path: str) -> str:
    """Lista os arquivos do diretório informado."""
    LOGGER.info("Tool list_directory called path=%s", path)
    return "\n".join(sorted(os.listdir(path)))


@agent.tool_plain
def read_file(path: str) -> str:
    """Lê e retorna o conteúdo de um arquivo de texto."""
    LOGGER.info("Tool read_file called path=%s", path)
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


@agent.tool_plain
def write_file(path: str, content: str) -> str:
    """Escreve o conteúdo em um arquivo."""
    LOGGER.info("Tool write_file called path=%s length=%d", path, len(content))
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    return f"File {path} written successfully."


@agent.tool_plain
def run_command(command: list[str]) -> str:
    """
    Executa um comando shell seguro e retorna o output.
    O comando deve ser uma lista de strings.
    Desabilita ('sudo', '-f', '-rf', 'rm').
    """
    LOGGER.info("Tool run_command called command=%s", command)
    commands_disabled = ("sudo", "-f", "-rf", "rm")
    for token in command:
        if token in commands_disabled:
            LOGGER.warning("run_command blocked unsafe token=%s", token)
            return f"Command {token} is disabled."
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("run_command failed")
        return str(exc)
    LOGGER.info("run_command finished exit_code=%s", result.returncode)
    return result.stdout.decode("utf-8") + result.stderr.decode("utf-8")


@agent.tool_plain
def suggest_command(command: str) -> str:
    """Sugere um comando com base na entrada do usuário."""
    LOGGER.info("Tool suggest_command storing command=%s", command)
    with open(EXEC_CMD_PATH, "w", encoding="utf-8") as file:
        file.write(command)
    return "Command suggested successfully."


class ChatMessage(Markdown):
    """Renderiza uma mensagem do chat com moldura."""

    def __init__(self, who: str, text: str, dt: str | None = None, **kwargs) -> None:
        label = "👤 You" if who == "user" else "🤖 Friday"
        when = dt or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        super().__init__(text, **kwargs)
        self.border_title = Text(label, style="white on green")
        self.border_subtitle = Text(when, style="white on gray")


class ChatView(VerticalScroll):
    """Container que mantém o histórico, com persistência em disco."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._messages: list[dict[str, str]] = []

    async def add_message(self, who: str, text: str, dt: str | None = None) -> None:
        entry = {
            "who": who,
            "text": text,
            "dt": dt or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        await self._mount_entry(entry, persist=True)

    async def _mount_entry(self, entry: dict[str, str], persist: bool) -> None:
        self._messages.append(entry)
        await self.mount(ChatMessage(**entry))
        if persist:
            self.save_history()
            LOGGER.info("Persisted %d chat messages", len(self._messages))
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
        LOGGER.info("Loaded %d historical chat messages", len(raw_messages))
        self._messages.clear()
        for entry in raw_messages:
            await self._mount_entry(entry, persist=False)

    async def clear_history(self) -> None:
        self._messages.clear()
        if os.path.exists(HIST_PATH):
            os.remove(HIST_PATH)
        LOGGER.info("History cleared and file removed")
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


class ChatTextArea(TextArea):
    """TextArea especializado para envio via Ctrl+Enter."""

    TOKEN_STYLE_NAME = "friday.token"

    BINDINGS = [
        Binding("ctrl+enter", "submit", "Submit"),
        Binding("ctrl+j", "submit", "Submit", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ensure_token_highlight_style()

    @dataclass
    class Submitted(Message, bubble=True):
        textarea: "ChatTextArea"
        text: str

    def _ensure_token_highlight_style(self) -> None:
        theme = getattr(self, "_theme", None)
        if not theme:
            return
        highlight_style = Style(color="#aee8ff", bgcolor="#1b1f24", bold=True, frame=True)
        theme.syntax_styles.setdefault(self.TOKEN_STYLE_NAME, highlight_style)

    def _watch_theme(self, theme: str) -> None:
        super()._watch_theme(theme)
        self._ensure_token_highlight_style()

    def _build_highlight_map(self) -> None:
        super()._build_highlight_map()
        tokens = self._iter_trigger_tokens()
        if not tokens:
            return
        highlights = self._highlights
        lines = self.document.lines
        for row, start_col, end_col in tokens:
            line = lines[row]
            start_byte = len(line[:start_col].encode("utf-8"))
            end_byte = len(line[:end_col].encode("utf-8"))
            highlights[row].append((start_byte, end_byte, self.TOKEN_STYLE_NAME))

    def _iter_trigger_tokens(self) -> list[tuple[int, int, int]]:
        tokens: list[tuple[int, int, int]] = []
        lines = self.document.lines
        for row, line in enumerate(lines):
            idx = 0
            length = len(line)
            while idx < length:
                char = line[idx]
                if char in ("@", "#"):
                    if idx > 0 and not line[idx - 1].isspace():
                        idx += 1
                        continue
                    end = idx + 1
                    while end < length and not line[end].isspace():
                        end += 1
                    if end - idx > 1:
                        tokens.append((row, idx, end))
                    idx = end
                else:
                    idx += 1
        return tokens

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self, self.text.strip()))



class ChatInput(Widget):
    """Widget composto que inclui TextArea + OptionList para autocomplete."""

    DEFAULT_CSS = """
    ChatInput {
        layout: vertical;
        overflow: visible visible;
    }
    ChatInput OptionList {
        max-height: 10;
        border: solid #1f1f1f;
        layer: suggestions;
        display: none;
    }
    """

    MAX_SUGGESTIONS = 20
    MAX_VISIBLE_ROWS = 8

    @dataclass
    class Submitted(Message, bubble=True):
        chat_input: "ChatInput"
        text: str

    def __init__(self, placeholder: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.placeholder = placeholder
        self._input: ChatTextArea | None = None
        self._options: OptionList | None = None
        self._suggestion_meta: Tuple[int, int, str] | None = None
        self._suggestions_visible = False
        self._visible_row_count = 0

    def compose(self) -> ComposeResult:
        yield ChatTextArea(id="chat_textarea", placeholder=self.placeholder)
        yield OptionList(id="chat_suggestions")

    @property
    def text(self) -> str:
        return self._input.text if self._input else ""

    @text.setter
    def text(self, value: str) -> None:
        if self._input:
            self._input.text = value
            self._input.move_cursor(self._cursor_from_offset(len(value)))
            self._update_suggestions()

    async def on_mount(self) -> None:
        self._input = self.query_one(ChatTextArea)
        self._options = self.query_one(OptionList)
        self._options.display = "none"
        self._options.styles.layer = "suggestions"
        self._options.styles.overflow_y = "auto"

    def on_resize(self, event: events.Resize) -> None:
        del event
        self._position_suggestions()

    @on(ChatTextArea.Submitted)
    def handle_textarea_submitted(self, event: ChatTextArea.Submitted) -> None:
        if event.textarea is not self._input:
            return
        event.stop()
        self._hide_suggestions()
        self.post_message(self.Submitted(self, event.text))

    @on(TextArea.Changed)
    def handle_textarea_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self._input:
            self._update_suggestions()

    @on(TextArea.SelectionChanged)
    def handle_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area is self._input:
            self._update_suggestions()

    @on(OptionList.OptionSelected)
    def handle_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list is self._options:
            event.stop()
            self._apply_suggestion(event.option.id or str(event.option.prompt))
            self._hide_suggestions()

    def on_key(self, event: events.Key) -> None:
        if not self._suggestions_visible or not self._options:
            return
        if event.key == "up":
            self._options.action_cursor_up()
            event.stop()
        elif event.key == "down":
            self._options.action_cursor_down()
            event.stop()
        elif event.key == "escape":
            self._hide_suggestions()
            event.stop()
        elif event.key in ("tab",) or (event.key == "space" and event.ctrl):
            self._accept_highlighted()
            event.stop()

    def focus_input(self) -> None:
        if self._input:
            self._input.focus()

    def _position_suggestions(self) -> None:
        if not (self._suggestions_visible and self._options and self._input):
            return
        region = self._input.region
        if region.width == 0:
            self.call_after_refresh(self._position_suggestions)
            return
        screen_region = self._input.screen_region
        screen = self.screen or self.app.screen
        screen_height = screen.size.height if screen else self.app.size.height
        available_below = max(0, screen_height - (screen_region.y + screen_region.height))
        available_above = max(0, screen_region.y)
        option_rows = max(1, min(self._visible_row_count or 1, self.MAX_VISIBLE_ROWS))
        estimated_height = option_rows + 2
        show_below = available_below >= estimated_height or available_below >= available_above
        if show_below:
            offset_y = region.y + region.height
            max_height = min(self.MAX_VISIBLE_ROWS, max(option_rows, available_below))
        else:
            max_height = min(self.MAX_VISIBLE_ROWS, max(option_rows, available_above))
            offset_y = region.y - max_height
        self._options.styles.offset = (region.x, offset_y)
        self._options.styles.width = region.width
        self._options.styles.max_height = max_height
        self._options.set_class(show_below, "-below")
        self._options.set_class(not show_below, "-above")

    def _cursor_from_offset(self, offset: int) -> tuple[int, int]:
        lines = self.text.split("
")
        running = 0
        for idx, line in enumerate(lines):
            line_len = len(line)
            if offset <= running + line_len:
                return idx, offset - running
            running += line_len + 1
        return (len(lines) - 1, len(lines[-1])) if lines else (0, 0)

    def _accept_highlighted(self) -> None:
        if not self._options:
            return
        option = self._options.highlighted_option
        if option is None:
            return
        self._apply_suggestion(option.id or str(option.prompt))
        self._hide_suggestions()

    def _hide_suggestions(self) -> None:
        if not self._options:
            return
        self._options.display = "none"
        self._options.clear_options()
        self._options.styles.offset = (0, 0)
        self._suggestion_meta = None
        self._suggestions_visible = False
        self._visible_row_count = 0

    def _update_suggestions(self) -> None:
        if not self._input or not self._options:
            return
        meta = self._detect_trigger()
        if not meta:
            self._hide_suggestions()
            return
        trigger, start, end, partial = meta
        options = (
            self._path_suggestions(partial)
            if trigger == "@"
            else self._tool_suggestions(partial)
        )
        if not options:
            self._hide_suggestions()
            return
        display_options = options[: self.MAX_SUGGESTIONS]
        self._options.clear_options()
        for option in display_options:
            self._options.add_option(Option(option, id=option))
        self._options.highlighted = 0
        self._options.display = "block"
        self._suggestion_meta = (start, end, trigger)
        self._suggestions_visible = True
        self._visible_row_count = len(display_options)
        self._position_suggestions()
        self.call_after_refresh(self._position_suggestions)

    def _detect_trigger(self) -> Tuple[str, int, int, str] | None:
        if not self._input:
            return None
        offset = self._cursor_offset()
        before = self._input.text[:offset]
        last_idx = -1
        trigger_char = ""
        for candidate in ("@", "#"):
            idx = before.rfind(candidate)
            if idx > last_idx:
                last_idx = idx
                trigger_char = candidate
        if last_idx == -1:
            return None
        if last_idx > 0 and not before[last_idx - 1].isspace():
            return None
        partial = before[last_idx + 1 :]
        if any(delim in partial for delim in ("
", "
", " ", "	")):
            return None
        return trigger_char, last_idx + 1, offset, partial

    def _cursor_offset(self) -> int:
        if not self._input:
            return 0
        row, col = self._input.cursor_location
        lines = self._input.text.split("
")
        row = min(row, len(lines) - 1)
        col = min(col, len(lines[row])) if lines else 0
        return sum(len(line) + 1 for line in lines[:row]) + col

    def _apply_suggestion(self, value: str) -> None:
        if not self._input or not self._suggestion_meta:
            return
        start, end, trigger = self._suggestion_meta
        original = self._input.text
        replacement = value if trigger == "#" else value
        trailing = "" if replacement.endswith("/") else " "
        new_text = original[:start] + replacement + trailing + original[end:]
        self._input.text = new_text
        new_offset = start + len(replacement) + len(trailing)
        self._input.move_cursor(self._cursor_from_offset(new_offset))
        self._hide_suggestions()

    def _path_suggestions(self, partial: str) -> list[str]:
        try:
            entries = sorted(os.listdir("."))
        except OSError:
            return []
        partial_lower = partial.lower()
        suggestions: list[str] = []
        for name in entries:
            display = name + ("/" if os.path.isdir(name) else "")
            if not partial or name.lower().startswith(partial_lower):
                suggestions.append(display)
            if len(suggestions) >= self.MAX_SUGGESTIONS:
                break
        return suggestions

    def _tool_suggestions(self, partial: str) -> list[str]:
        partial_lower = partial.lower()
        matches = [tool for tool in TOOL_NAMES if tool.startswith(partial_lower)]
        return matches if matches else TOOL_NAMES



class Chat(Widget):
    """Widget principal composto por ChatView, indicador e ChatInput."""

    def compose(self) -> ComposeResult:
        yield ChatView()
        yield Label(id="thinking")
        yield ChatInput(placeholder="Type your question...")


class FridayChatApp(App):
    """Aplicação Textual principal."""

    CSS_PATH = "style.tcss"
    INLINE = True
    BINDINGS = [
        Binding("ctrl+k", "clear_history", "Clear History"),
        Binding("ctrl+i", "toggle_input_mode", "Toggle Input Mode"),
        Binding("ctrl+l", "toggle_llm", "Toggle LLM Model"),
    ]

    def compose(self) -> ComposeResult:
        yield Chat(id="chat")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one(Chat)
        self._chat_view = chat.query_one(ChatView)
        self._chat_input: ChatInput = chat.query_one(ChatInput)
        self._thinking = chat.query_one(Label)
        self._thinking.content = MODEL
        self._chat_input.focus_input()
        self._chat_view.scroll_to_latest()

    @on(ChatInput.Submitted)
    async def handle_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()
        self._chat_input.text = ""
        if not text:
            self.notify("Message cannot be empty", title="Error", severity="error")
            return
        LOGGER.info("User submitted prompt: %s", text)
        await self._chat_view.add_message("user", text)
        self._thinking.loading = True
        try:
            LOGGER.info("Dispatching prompt to agent")
            reply = await asyncio.to_thread(agent.run_sync, text)
            output = reply.output
            LOGGER.info("Agent response: %s", output)
        except Exception as exc:  # pylint: disable=broad-except
            output = f"Erro ao chamar o agente: {exc}"
            LOGGER.exception("Agent execution failed")
        finally:
            self._thinking.loading = False
        await self._chat_view.add_message("agent", output)

    async def action_clear_history(self) -> None:
        await self._chat_view.clear_history()


if __name__ == "__main__":
    FridayChatApp().run(inline=True)
