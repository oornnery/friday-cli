# Friday CLI

Friday CLI é uma TUI em Textual que integra um agente Pydantic AI. Ela mantém um histórico de conversas (`friday_history.json`) e oferece atalhos para limpar e navegar rapidamente.

## Requisitos

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- zsh (para usar o script auxiliar)
- Dependências listadas em `pyproject.toml`

## Executando com uv

### 1. Via uv diretamente

```bash
uv run python src/main.py
```

### 2. Script zsh pronto

O projeto inclui `scripts/run_friday.zsh`, que garante que a execução ocorra na raiz e repassa argumentos ao app.

1. Dê permissão de execução (já feito no repositório, mas use se necessário):
   ```bash
   chmod +x scripts/run_friday.zsh
   ```
2. Rode:
   ```bash
   ./scripts/run_friday.zsh
   ```

### 3. Atalho no zsh

Adicione ao seu `~/.zshrc`:

```zsh
alias fridaychat="/caminho/para/o/repo/scripts/run_friday.zsh"
```

Recarregue o shell (`source ~/.zshrc`) e basta digitar `fridaychat`.

#### Atalho com combinação de teclas

Se usa o plugin `zsh-autosuggestions` ou um gerenciador como `zle`, crie um widget:

```zsh
function fridaychat() {
  zle -I
  uv run python ~/proj/friday-cli/src/main.py &
}
zle -N fridaychat
bindkey '^F' fridaychat  # Ctrl+F para abrir o Friday
```

Ajuste `'^F'` para outra combinação caso já esteja em uso.

## Atalhos dentro do app

- `Ctrl+K`: limpa histórico (remove `friday_history.json` e limpa a UI).
- `Ctrl+Shift+K`: mesma ação, pensado como “limpeza rápida”.
- Input principal recebe foco automaticamente ao abrir.

## Limpeza de histórico

O histórico é salvo como JSON na raiz. Os atalhos acima limpam tanto o arquivo quanto os widgets renderizados. Se quiser deletar manualmente:

```bash
rm friday_history.json
```

## Dicas

- O app rola automaticamente até a última mensagem na inicialização e quando novas mensagens chegam.
- Mensagens antigas em formatos variados são normalizadas para um dicionário `{who, text, dt}` antes de serem exibidas.
- Para confirmar se todas as dependências estão OK:
  ```bash
  uv sync
  uvx ty check
  ```
