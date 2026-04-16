"""Model management helpers for CLI commands."""

import json
import os
import platform
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.prompt import Confirm, Prompt

from deeppresenter.utils.config import DeepPresenterConfig

from .common import (
    LOCAL_BASE_URL,
    LOCAL_MODEL,
    PACKAGE_DIR,
    REQUIRED_LLM_KEYS,
    console,
)


def is_local_model_server_running() -> bool:
    """Check whether Ollama server responds on /v1/models."""
    try:
        # Ollama uses port 11434
        models_url = f"{LOCAL_BASE_URL}/models"
        print(f"models_url = {models_url}")
        req = Request(models_url, method="GET")
        with urlopen(req) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8") or "{}")
            # OpenAI-compatible API returns {"data": [...]}
            return isinstance(payload, dict) and isinstance(payload.get("data"), list)
    except (
        HTTPError,
        URLError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ):
        return False


def _build_inference_command() -> list[str]:
    system = platform.system().lower()
    print(f"system = {system}")
    # For Ollama, we don't need to start a server command
    # The ensure_llamacpp() function will check and start Ollama if needed
    console.print(
        f"[cyan]Local model service is not running. Please ensure Ollama is running with model '{LOCAL_MODEL}'[/cyan]"
    )
    return None


def setup_inference() -> int | None:
    """Start local inference if needed and return the PID we should later stop."""
    if is_local_model_server_running():
        return None

    # For Ollama, we don't start a process, just check if server is running
    # The ensure_llamacpp() function should have started Ollama if needed
    console.print("[yellow]Ollama server is not running. Please start Ollama with `ollama serve`[/yellow]")
    
    # Wait for server to start
    import time
    for _ in range(30):
        time.sleep(1)
        if is_local_model_server_running():
            console.print("[green]✓[/green] Ollama server is now running")
            return None  # No PID to return since we didn't start a process
    
    raise RuntimeError(
        "Ollama server did not start. Please ensure Ollama is installed and running."
    )


def is_onboarded() -> bool:
    """Check if user has completed onboarding."""
    from .common import CONFIG_FILE, MCP_FILE

    return CONFIG_FILE.exists() and MCP_FILE.exists()


def prompt_llm_config(
    name: str,
    optional: bool = False,
    existing: dict | None = None,
    previous_config: tuple[str, dict] | None = None,
    reuse_previous_default: bool = True,
) -> dict | None:
    """Prompt user for LLM configuration."""
    if optional and not Confirm.ask(f"Configure {name}?", default=False):
        return None

    console.print(f"\n[bold cyan]Configuring {name}[/bold cyan]")

    if existing:
        console.print(
            f"[dim]Previous: {existing.get('model', 'N/A')} @ {existing.get('base_url', 'N/A')}[/dim]"
        )
        if Confirm.ask(f"Reuse previous {name} configuration?", default=True):
            return existing

    if previous_config:
        prev_name, prev_cfg = previous_config
        console.print(
            f"[dim]Last configured: {prev_name} - {prev_cfg.get('model', 'N/A')} @ {prev_cfg.get('base_url', 'N/A')}[/dim]"
        )
        if Confirm.ask(
            f"Reuse {prev_name} configuration?", default=reuse_previous_default
        ):
            return prev_cfg

    return {
        "base_url": Prompt.ask("Base URL"),
        "model": Prompt.ask("Model name"),
        "api_key": Prompt.ask("API key", password=True),
    }


def has_complete_model_config(existing_config: dict | None) -> bool:
    return isinstance(existing_config, dict) and all(
        isinstance(existing_config.get(key), dict)
        and existing_config.get(key, {}).get("base_url")
        and existing_config.get(key, {}).get("model")
        for key in REQUIRED_LLM_KEYS
    )


def uses_local_model(config: DeepPresenterConfig) -> bool:
    dumped = config.model_dump()
    return any(
        "127.0.0.1" in str(dumped.get(key, {}).get("base_url", ""))
        or "localhost" in str(dumped.get(key, {}).get("base_url", ""))
        for key in REQUIRED_LLM_KEYS
    )
