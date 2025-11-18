"""
LLM Chat App - Textual TUI com Pydantic AI
Estrutura pytônica seguindo boas práticas do Textual
"""

import asyncio

from rich.markdown import Markdown
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Label, TextArea

# ============================================================================
# WIDGETS CUSTOMIZADOS
# ============================================================================


class MessageBubble(Widget):
    """Widget para renderizar uma mensagem individual."""

    DEFAULT_CSS = """
    MessageBubble {
        height: auto;
        width: 100%;
        padding: 1 2;
        margin-bottom: 1;
        border-left: solid;
    }

    MessageBubble.user {
        background: $accent;
    }

    MessageBubble.assistant {
        background: $panel;
    }

    MessageBubble.system {
        background: $accent-muted;
    }
    """

    def __init__(self, text: str, role: str = "user", **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.role = role

    def on_mount(self):
        """Aplica classe CSS baseado no role."""
        self.add_class(self.role)

    def render(self) -> Markdown:
        """Renderiza com Markdown."""
        return Markdown(self.text)


class ChatView(ScrollableContainer):
    """Container scrollável para mensagens."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        border: solid $border-blurred;
        padding: 1;
        background: $panel;
        overflow-y: auto;
    }
    """

    def add_message(self, text: str, role: str = "user"):
        """Adiciona mensagem e faz scroll para o final."""
        bubble = MessageBubble(text, role=role)
        self.mount(bubble)
        self.scroll_visible()

    def clear_messages(self):
        """Limpa todas as mensagens."""
        self.query(MessageBubble).remove()


class ChatInput(Container):
    """Input com TextArea multi-linha."""

    DEFAULT_CSS = """
    ChatInput {
        height: 8;
        border-top: solid $accent-muted;
        padding: 1;
        layout: vertical;
    }

    ChatInput #input-area {
        height: 5;
        width: 1fr;
        border: solid $border-blurred;
    }

    ChatInput .button-row {
        height: 2;
        margin-top: 1;
        align: left middle;
    }

    ChatInput Button {
        margin-right: 1;
    }
    """

    def compose(self):
        yield TextArea(id="input-area", soft_wrap=True, tab_behavior="indent")
        with Horizontal(classes="button-row"):
            yield Button("Send [Ctrl+Enter]", id="send-btn", variant="primary")
            yield Button("Clear [Ctrl+K]", id="clear-btn")
            yield Label(id="status", expand=False)


# ============================================================================
# APP PRINCIPAL
# ============================================================================


class LLMChatApp(App):
    """Chat LLM com Textual - Pytônico e Reativo."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Sair", show=False),
        Binding("ctrl+enter", "send_message", "Enviar", show=True),
        Binding("ctrl+j", "send_message", "Enviar", show=False),
        Binding("ctrl+k", "clear_chat", "Limpar", show=True),
    ]

    CSS_PATH = "styles.tcss"

    def compose(self) -> ComposeResult:
        """Layout principal."""
        yield Header(show_clock=True)

        with Vertical(id="main-container"):
            yield Label("🤖 LLM Chat Agent", id="title")
            yield ChatView(id="chat-view")
            yield ChatInput(id="input-box")

        yield Footer()

    # ===== LIFECYCLE =====

    def on_mount(self):
        """Inicializa a app."""
        self.title = "LLM Chat Textual"
        self.sub_title = "Powered by Pydantic AI"
        self.messages_history = []  # Histórico de mensagens

        # Mensagem de boas-vindas
        chat_view = self.query_one("#chat-view", ChatView)
        chat_view.add_message(
            "👋 Bem-vindo ao LLM Chat Agent!\n\n"
            "**Comandos:**\n"
            "- `Ctrl+Enter` para enviar\n"
            "- `Ctrl+K` para limpar chat\n"
            "- `Ctrl+C` para sair",
            role="system",
        )

    # ===== EVENT HANDLERS =====

    @on(Button.Pressed, "#send-btn")
    def _handle_send(self, event: Button.Pressed):
        """Evento: botão Send clicado."""
        self.action_send_message()

    @on(Button.Pressed, "#clear-btn")
    def _handle_clear(self, event: Button.Pressed):
        """Evento: botão Clear clicado."""
        self.action_clear_chat()

    # ===== ACTIONS (Keyboard + Button) =====

    async def action_send_message(self):
        """Ação: Envia mensagem para o LLM."""
        input_area = self.query_one("#input-area", TextArea)
        chat_view = self.query_one("#chat-view", ChatView)
        status_label = self.query_one("#status", Label)

        user_text = input_area.text.strip()
        if not user_text:
            status_label.update("❌ Digite uma mensagem!")
            return

        # 1. Mostra mensagem do usuário
        chat_view.add_message(user_text, role="user")
        self.messages_history.append({"role": "user", "content": user_text})

        # 2. Limpa input
        input_area.clear()

        # 3. Mostra status de "pensando"
        status_label.update("⏳ Processando...")

        # 4. Chama LLM em background (Worker)
        self.run_worker(self._llm_request(user_text), exclusive=True)

    def action_clear_chat(self):
        """Ação: Limpa histórico de chat."""
        chat_view = self.query_one("#chat-view", ChatView)
        chat_view.clear_messages()
        self.messages_history = []
        status_label = self.query_one("#status", Label)
        status_label.update("✨ Chat limpo!")

    # ===== WORKER PARA LLM (Async) =====

    async def _llm_request(self, message: str):
        """Worker: Simula requisição ao LLM.

        Substituir por chamada real a Pydantic AI/OpenRouter
        """
        chat_view = self.query_one("#chat-view", ChatView)
        status_label = self.query_one("#status", Label)

        try:
            # Simula delay de API
            await asyncio.sleep(1)

            # AQUI: Integrar com Pydantic AI
            # response = await self.agent.run(message)

            # Resposta simulada
            response = (
                f"## Resposta do Agent\n\n"
                f"Você disse: **{message}**\n\n"
                f"```python\n"
                f"# Este é um exemplo de resposta\n"
                f"# Integre com Pydantic AI aqui\n"
                f"response = agent.run('{message}')\n"
                f"```"
            )

            chat_view.add_message(response, role="assistant")
            self.messages_history.append({"role": "assistant", "content": response})
            status_label.update("✅ Pronto!")

        except Exception as e:
            error_msg = f"❌ Erro: {str(e)}"
            chat_view.add_message(error_msg, role="system")
            status_label.update(error_msg)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = LLMChatApp()
    app.run()
