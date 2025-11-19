import csv
import json
import os
import xml.etree.ElementTree
from pathlib import Path

import toml
import yaml
from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv(Path.home() / ".env")

FRIDAY_PATH = Path().home() / ".config/friday"
FRIDAY_CONFIG_PATH = FRIDAY_PATH / "config.toml"
FRIDAY_HIST_PATH = FRIDAY_PATH / "history.json"
FRIDAY_COMM_PATH = FRIDAY_PATH / "exec_cmd"
MODEL = "openrouter:openai/gpt-5.1-codex-mini"

agent = Agent(
    model=MODEL,
    instructions="Your name is Friday. You are a helpful assistant on the command line.",
)


@agent.tool_plain
def ls(path: str = "."):
    """List files and directories in the given path"""
    return os.listdir(path)


@agent.tool_plain
def reader(path: str, typer: str | None = None):
    """Read the contents of a file

    Args:
        path (str): Path to the file to read.
        typer (str | None): Optional file type to read exemple: json.

    Returns:
        str: Contents of the file.
    """
    with open(path, "r") as file:
        if typer == "json":
            return json.load(file)
        elif typer == "yaml":
            return yaml.safe_load(file)
        elif typer == "toml":
            return toml.load(file)
        elif typer == "csv":
            return csv.reader(file)
        elif typer == "xml":
            return xml.etree.ElementTree.parse(file).getroot()
        return file.read()


@agent.tool_plain
def writer(path: str, content: str):
    """Write content to a file"""
    with open(path, "w") as file:
        file.write(content)


@agent.tool_plain
def find(path: str, pattern: str):
    """Find files matching a pattern"""
    return glob.glob(os.path.join(path, pattern))


@agent.tool_plain
def suggest_command(command: str):
    """Suggest a command to the user in prompt"""
    with open(FRIDAY_COMM_PATH, "w") as file:
        file.write(command)
